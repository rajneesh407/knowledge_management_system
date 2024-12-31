# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import streamlit as st
from pathlib import Path
from pdf_parser import PDFParser
from docx_parser import WordDocParser
from retriever import RetrieverModel
from model import initialize_client
from config import PERSISTANT_DIRECTORY, PDF_DIRECTORY, ALLOWED_FILE_TYPES
import chromadb
from chromadb.config import Settings
import os 
import pytesseract
import streamlit as st
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.collection import Collection
import pandas as pd 

st.set_page_config(page_title="Resources List", layout='wide', page_icon="📁")


os.environ["PATH"] += os.pathsep + 'C:\\Users\\rajneesh.jha\\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rajneesh.jha\\Tesseract-OCR\\tesseract.exe'

chroma_db = chromadb.Client(Settings(is_persistent=True,persist_directory=PERSISTANT_DIRECTORY))

engine = create_engine(f'sqlite:///knowledge_database.db')
Session = sessionmaker(bind=engine)
session = Session()

st.subheader("Knowledge Database")

collections = session.query(Collection).all()

if collections:
    data = [
        {
            "Created Date": collection.created_date,
            "Collection Name": collection.collection_name,
            "Description": collection.description,
            "Text Model": collection.text_summarization_model,
            "Image Model": collection.image_summarization_model,
        }
        for collection in collections
    ]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

st.subheader("Add New Resource:")
with st.expander("Click to add new resource"):
    col1, col2 = st.columns([2, 1])
    with col1:
        collection_name = st.text_input("Collection Name")
    with col2:
        file_type = st.selectbox(
            "Choose File Type",
            options=ALLOWED_FILE_TYPES.keys(),
            key="file_type"
        )
    
    uploaded_file = st.file_uploader(
        f"Upload a {file_type} File",
        type=ALLOWED_FILE_TYPES[file_type],
        key="file_uploader"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        text_summarization_model = st.selectbox("Text Summarization Model", ["llama_8b"])
    with col2:
        image_summarization_model = st.selectbox("Image Summarization Model", ["llama_11b"])
    
    # Fourth row: description
    description = st.text_area("Description")

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

            if st.button("Submit", key="submit_resource"):
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
            
