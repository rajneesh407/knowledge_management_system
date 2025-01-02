from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.collection import Base

engine = create_engine("sqlite:///code//database//knowledge_database.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
