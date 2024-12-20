import sqlalchemy as db
from sqlalchemy import Column, Integer, String, Boolean, func, desc, asc

from database import Base, SessionLocal


class User(Base):
    __tablename__ = "user"
    # уникальный идентификатор пользователя (primary key):
    id = Column(Integer, primary_key=True)
    # пол пользоователя:
    gender = Column(Integer)
    # возраст пользоователя:
    age = Column(Integer)
    # страна пользоователя:
    country = Column(String)
    # город пользоователя:
    city = Column(String)
    # экспериментальная группа пользоователя:
    exp_group = Column(Integer)
    # операционная система пользоователя:
    os = Column(String)
    # источник трафика пользоователя:
    source = Column(String)


if __name__ == "__main__":
    session = SessionLocal()
    result_of_query = (
        session.query(User.country, User.os, func.count(User.country).label("count"))
        .filter(User.exp_group == 3)
        .order_by(func.count(User.country).desc())
        .group_by(User.country, User.os)
        .having(func.count(User.country) > 100)
    )

    print([(x.country, x.os, x.count) for x in result_of_query])
