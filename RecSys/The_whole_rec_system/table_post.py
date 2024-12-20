from sqlalchemy import Column, Integer, String

from database import Base, SessionLocal


class Post(Base):
    __tablename__ = "post"
    # уникальный идентификатор поста (primary key):
    id = Column(Integer, primary_key=True)
    # текст поста:
    text = Column(String)
    # тема поста:
    topic = Column(String)


if __name__ == "__main__":
    session = SessionLocal()
    result_of_query = (
        session.query(Post)
        .filter(Post.topic == "business")
        .order_by(Post.id.desc())
        .limit(10)
        .all()
    )
    print([x.id for x in result_of_query])
