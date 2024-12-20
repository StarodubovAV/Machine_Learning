from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, select
from sqlalchemy.orm import relationship

from database import Base, SessionLocal
from table_post import Post
from table_user import User


class Feed(Base):
    __tablename__ = "feed_action"
    # идентификатор пользователя:
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    # идентификатор поста:
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    # совершенное в сети действие:
    action = Column(String)
    # время действия:
    time = Column(DateTime)
    # мостик до таблицы User:
    user = relationship(User)
    # мостик до таблицы Post:
    post = relationship(Post)


if __name__ == "__main__":
    session = SessionLocal()
    #    result = (
    #       session.query(Feed)
    #        .join(User)
    #        .filter(User.age < 16)
    #        .filter(User.id == 257)
    #        .join(Post)
    #        .filter(Post.topic == "business")
    #        .limit(10)
    #        .all()
    #    )

    subquery = (
        select(Feed.post_id)
        .filter(Feed.action == 'like')
        .group_by(Feed.post_id)
        .order_by(func.count(Feed.post_id).desc())
        .limit(10)
        .subquery()
    )
    query = session.query(Post).filter(Post.id.in_(select(subquery))).all()

    subquery = (
        session.query(Feed.post_id, func.count(Feed.post_id).label("like_count"))
        .filter(Feed.action == 'like')
        .group_by(Feed.post_id)
        .order_by(func.count(Feed.post_id).desc())
        .limit(2)
        .subquery()
    )
    query = session.query(Post).join(subquery, Post.id == subquery.c.post_id).all()


    for x in query:
        print(x.id, x.text, x.topic)
