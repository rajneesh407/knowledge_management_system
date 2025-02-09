import streamlit as st
from pathlib import Path
from backend.pdf_parser import PDFParser
from backend.docx_parser import WordDocParser
from backend.retriever import RetrieverModel
from backend.model import initialize_client
from backend.config import PERSISTANT_DIRECTORY, PDF_DIRECTORY, ALLOWED_FILE_TYPES
import chromadb
from chromadb.config import Settings
import os
import pytesseract
import streamlit as st
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.models.collection import Collection
import pandas as pd
from backend.utils import get_video_transcript, transcribe_audio_file

st.set_page_config(page_title="Resources List", layout="wide", page_icon="📁")


os.environ["PATH"] += os.pathsep + "C:\\Users\\rajneesh.jha\\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Users\\rajneesh.jha\\Tesseract-OCR\\tesseract.exe"
)
os.environ["PATH"] += (
    os.pathsep
    + "C:\\Users\\rajneesh.jha\\ffmpeg-2025-02-06-git-6da82b4485-full_build\\bin"
)
chroma_db = chromadb.Client(
    Settings(is_persistent=True, persist_directory=PERSISTANT_DIRECTORY)
)

engine = create_engine(f"sqlite:///database/knowledge_database.db")
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
else:
    st.write("No resources to show. Please add resources below.")
if st.button("Query Existing Resource"):
    st.switch_page("pages/2_📖_Answer_Vault.py")
st.subheader("Add New Resource:")
with st.expander("Click to add new resource"):
    col1, col2 = st.columns([2, 1])
    with col1:
        collection_name = st.text_input("Collection Name")
    with col2:
        file_type = st.selectbox(
            "Choose File Type", options=ALLOWED_FILE_TYPES.keys(), key="file_type"
        )

    summarize_content = st.checkbox("Summarize Content")
    video_link = None
    uploaded_file = None
    if file_type != "YOUTUBE":
        uploaded_file = st.file_uploader(
            f"Upload a {file_type} File",
            type=ALLOWED_FILE_TYPES[file_type],
            key="file_uploader",
        )
    else:
        video_link = st.text_input("Youtube Link")

    text_summarization_model = None
    if summarize_content:
        col1, col2 = st.columns(2)
        with col1:
            text_summarization_model = st.selectbox(
                "Text Summarization Model", ["llama_8b"]
            )
        with col2:
            image_summarization_model = st.selectbox(
                "Image Summarization Model", ["llama_11b"]
            )
    else:
        image_summarization_model = st.selectbox(
            "Image Summarization Model", ["llama_11b"]
        )

    description = st.text_area("Description")

    if (uploaded_file or video_link) and collection_name:
        # Create a folder with the collection name under PDF_DIRECTORY
        collection_path = Path(PDF_DIRECTORY) / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)

        if file_type in ["PDF", "DOCX"]:
            if file_type == "PDF":
                path = f"{collection_path}\\{collection_name}.pdf"
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                parser = PDFParser(path)
            elif file_type == "DOCX":
                path = f"{collection_path}\\{collection_name}.docx"
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                parser = WordDocParser(path)

            with st.sidebar:
                from streamlit_pdf_viewer import pdf_viewer

                st.subheader("Preview Document")
                pdf_viewer(uploaded_file.getvalue(), height=500)
            st.sidebar.success(f"'{uploaded_file.name}' uploaded successfully.")
            st.sidebar.info("Parsing document ... This may take a few moments.")
            texts_list, tables_list, images_list = parser.parse()
            st.sidebar.success("Sucessfully Parsed Document.")

        elif file_type == "AUDIO":
            path = f"{collection_path}\\{collection_name}.wav"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())  # Save the file
            texts_list = transcribe_audio_file(path).split(". ")
            tables_list, images_list = [], []
            with st.sidebar:
                st.subheader("Listen Audio")
                st.audio(path, format="audio/wav")
            st.sidebar.success(f"Audio uploaded and parsed successfully.")

        elif file_type == "YOUTUBE":
            path = f"{collection_path}\\video.mp4"
            transcript = get_video_transcript(video_link, collection_path)
            texts_list = transcript.split(". ")
            tables_list, images_list = [], []
            st.write(texts_list)
            with st.sidebar:
                st.subheader("Watch Video")
                st.video(video_link)
            st.sidebar.success(f"'{video_link}' uploaded and parsed successfully.")

    if st.button("Submit", key="submit_resource"):
        client_model = initialize_client()

        retriever_class = RetrieverModel(
            collection_name=collection_name,
            client_model=client_model,
            persistant_directory=PERSISTANT_DIRECTORY,
            model_name_text=text_summarization_model,
            model_name_image=image_summarization_model,
        )
        retriever_class.add_documents(
            texts_list,
            tables_list,
            images_list,
            summarize_content=summarize_content,
            add_metadata=file_type not in ["YOUTUBE", "AUDIO"],
        )

        new_collection = Collection(
            collection_name=collection_name,
            description=description,
            text_summarization_model=text_summarization_model,
            image_summarization_model=image_summarization_model,
            file_path=path,
            file_type=file_type,
            summarize_content=summarize_content,
        )
        session.add(new_collection)
        session.commit()
        st.success("Collection saved successfully!")
