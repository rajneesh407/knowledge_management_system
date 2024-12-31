from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.collection import Base

engine = create_engine('sqlite:///knowledge_database.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()