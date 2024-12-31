import streamlit as st
from pathlib import Path
from pdf_parser import PDFParser
from docx_parser import WordDocParser
from retriever import RetrieverModel
from model import response_model, initialize_client
from config import PERSISTANT_DIRECTORY, PDF_DIRECTORY, ALLOWED_FILE_TYPES
import chromadb
from chromadb.config import Settings
import os 
import pytesseract
from utils import display_base64_image
from gtts import gTTS
from io import BytesIO
import streamlit as st
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.collection import Collection
from datetime import datetime


st.set_page_config(page_title="Resources List")

os.environ["PATH"] += os.pathsep + 'C:\\Users\\rajneesh.jha\\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rajneesh.jha\\Tesseract-OCR\\tesseract.exe'

chroma_db = chromadb.Client(Settings(is_persistent=True,persist_directory=PERSISTANT_DIRECTORY))


engine = create_engine('sqlite:///knowledge_database.db')
Session = sessionmaker(bind=engine)
session = Session()

# Input fields
st.subheader("Resource Description:")
collection_name = st.text_input("Collection Name")
description = st.text_area("Description")
st.subheader("Upload Resources:")
file_type = st.selectbox(
    "Choose File Type",
    options=["PDF", "DOCX"],
    key="file_type"
)

uploaded_file = st.file_uploader(
    f"Upload a {file_type} File",
    type=ALLOWED_FILE_TYPES[file_type],
    key="file_uploader"
)
st.subheader("Choose Models:")
text_summarization_model = st.selectbox("Text Summarization Model", ["llama_8b"])
image_summarization_model = st.selectbox("Image Summarization Model", ["llama_11b"])

if (uploaded_file and collection_name):
    # Create a folder with the collection name under PDF_DIRECTORY
    if uploaded_file and collection_name:
        collection_path = Path(PDF_DIRECTORY) / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file as a PDF in the collection folder
        if file_type == 'PDF':
            path  = f"{collection_path}\\{collection_name}.pdf"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            parser = PDFParser(path)
        elif file_type=="DOCX":
            path  = f"{collection_path}\\{collection_name}.docx"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            parser=WordDocParser(path)
        
        st.sidebar.success(f"'{uploaded_file.name}' uploaded successfully.")
        st.sidebar.info("Parsing document ... This may take a few moments.")
        texts_list, tables_list, images_list = parser.parse()
        st.sidebar.success("Sucessfully Parsed Document.")
        client_model = initialize_client()
        retriever_class = RetrieverModel(collection_name=collection_name,client_model=client_model, persistant_directory=PERSISTANT_DIRECTORY)
        retriever_class.add_documents(texts_list, tables_list, images_list)

        if st.button("Submit"):
            new_collection = Collection(
                collection_name=collection_name,
                description=description,
                text_summarization_model=text_summarization_model,
                image_summarization_model=image_summarization_model,
                file_path=path,
                file_type=file_type
            )
            session.add(new_collection)
            session.commit()
            st.success("Collection saved successfully!")
        
