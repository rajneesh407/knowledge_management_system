from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    collection_name = Column(String(100), nullable=False)
    file_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    text_summarization_model = Column(String(50))
    image_summarization_model = Column(String(50))
    file_path = Column(String(255), nullable=False)
    summarize_content = Column(Boolean, default=False)
