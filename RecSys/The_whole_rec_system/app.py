import pickle
from typing import List
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from schema import PostGet
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
from sqlalchemy import Column, Integer, String
from loguru import logger
import lightgbm as lgb
from catboost import CatBoostClassifier

app = FastAPI()

# Load environment variables from a .env file
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# строка для подключения к базе данных:
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# прослойка SQLAlchemy, которая уничтожает все различия между БД
engine = create_engine(SQLALCHEMY_DATABASE_URL,
                       pool_size=20,
                       max_overflow=30,
                       pool_timeout=60,
                       pool_recycle=1800)


# специальный код выгрузки признаков постов из базы данных для снижения использования памяти+++++
def batch_load_sql(query: str, used_engine=engine, parse_dates=None):
    with used_engine.connect() as connection:
        dbapi_conn = connection.connection

        chunks = []
        for chunk_dataframe in pd.read_sql(query, con=dbapi_conn, chunksize=200000, parse_dates=parse_dates):
            chunks.append(chunk_dataframe)

    df = pd.concat(chunks, ignore_index=True)

    return df


# код загрузки модели===
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


# код загрузки фич===
def load_features_for_control_model(engine):
    # загружаем уникальные записи post_id и user_id с лайками
    logger.info("loading like posts...")
    liked_posts_query = """
    SELECT DISTINCT post_id, user_id
    FROM public.feed_data
    WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query, engine, parse_dates=['timestamp'])

    # загружаем фичи постов, которые сгенерировали заранее
    logger.info('loading pregenerated posts features...')
    posts_features_query = "SELECT * FROM starodubovav_post_info_control"

    posts_features = batch_load_sql(posts_features_query, engine)

    # загружаем фичи юзеров
    logger.info("loading user features...")
    user_features_query = "SELECT * FROM public.user_data"
    user_features = batch_load_sql(user_features_query, engine)

    return [liked_posts, posts_features, user_features]

def load_features_for_test_model(engine):
    # загружаем уникальные записи post_id и user_id с лайками
    logger.info("loading like posts...")
    liked_posts_query = """
    SELECT DISTINCT post_id, user_id
    FROM public.feed_data
    WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query, engine, parse_dates=['timestamp'])

    # загружаем фичи постов, которые сгенерировали заранее
    logger.info('loading pregenerated posts features...')
    posts_features_query = "SELECT * FROM starodubovav_modified_post_info_lesson_22"

    posts_features = batch_load_sql(posts_features_query, engine)

    # загружаем фичи юзеров
    logger.info("loading user features...")
    user_features_query = "SELECT * FROM starodubovav_modified_user_info_lesson_22"
    user_features = batch_load_sql(user_features_query, engine)

    return [liked_posts, posts_features, user_features]


# Загрузка модели LightGBM
def load_models_by_pickle():
    model_path = get_model_path("model_control.pkl")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Загрузка модели CatBoost
def load_models_by_CatBoost():
    model_path = get_model_path("catboost_test_model")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

print(f"pandas version: {pd.__version__}")
print(f"SQLAlchemy version: {sqlalchemy.__version__}")
#===========
# Когда поднимается сервис делаем следующее:
logger.info("Loading model...")
my_model = load_models_by_CatBoost()  # загружаем модель

logger.info("Loading features...")
#features = load_features_for_control_model(engine)
features = load_features_for_test_model(engine)
logger.info("Service is up and running...")


def get_recommended_feed(id: int, time: datetime, limit: int):
    # Загрузим фичи по пользователям
    logger.info(f"user_id:{id}")
    logger.info("reading features ...")
    user_features = features[2].loc[features[2].user_id == id]

    # загрузим фичи по постам
    logger.info("dropping columns ...")
    post_features = features[1].drop(columns=['text'])
    content = features[1][['post_id', 'text', 'topic']]

    # объединяем фичи
    logger.info("zipping everything ...")
    add_user_features = dict(
        zip(user_features.columns, user_features.values[0]))  # user_features.values[0] -[0] так как это список списка
    user_post_features = post_features.assign(
        **add_user_features)  # дабавляем фичи выбранного юзера ко всем фичам всех постов
    user_post_features = user_post_features.set_index('post_id')

    # добавляем инфу о дате рекомедации
    logger.info("add time info ...")
    # Добавим час дня
    user_post_features['hour'] = time.hour
    user_post_features['month'] = time.month
    user_post_features['day_of_week'] = time.weekday()

    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'

    user_post_features['time_of_day'] = user_post_features['hour'].apply(get_time_of_day)

    # Добавим цикличность в временную шкалу: Модели лучше улавливают
    # циклические зависимости, когда время представлено таким образом.
    user_post_features['hour_cos'] = np.cos(2 * np.pi * user_post_features['hour'] / 24)
    user_post_features = user_post_features.drop(columns='user_id')

    # предсказываем вероятность лайкнуть пост для всех постов
    logger.info('predicting')
    cat_cols = user_post_features.select_dtypes(include='object')
    for col in cat_cols:
        user_post_features[col] = user_post_features[col].astype('category')
    user_post_features['predicts'] = my_model.predict_proba(user_post_features)[:,
                                     1]  # вероятность лайка, то есть вероятность положительного класса

    # убираем записи, где пользователь уже ставил лайк
    logger.info('deleting posts that have already been liked')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_post_features[~user_post_features.index.isin(liked_posts)]

    # рекомендуем топ-5 постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [PostGet(**{
        "id": i,
        "text": content[content.post_id == i].text.values[0],
        "topic": content[content.post_id == i].topic.values[0]
    }) for i in recommended_posts]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5,
) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)
