from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from a .env file
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# строка для подключения к базе данных:
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# прослойка SQLAlchemy, которая уничтожает все различия между БД
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# фабрика для создания экземпляров Session с заданными параметрами
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# bind=engine в аргументах sessionmaker() означает, что сессии, которые мы создаем, должны быть привязаны к движку (
# который ранее создали)
Base = declarative_base()
