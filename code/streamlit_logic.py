import streamlit as st
from pathlib import Path
from pdf_parser import PDFParser
from docx_parser import WordDocParser
from retriever import RetrieverModel
from model import response_model, initialize_client
from config import PERSISTANT_DIRECTORY, PDF_DIRECTORY
import chromadb
from chromadb.config import Settings
import os 
import pytesseract
from utils import display_base64_image
from gtts import gTTS
from io import BytesIO

os.environ["PATH"] += os.pathsep + 'C:\\Users\\rajneesh.jha\\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rajneesh.jha\\Tesseract-OCR\\tesseract.exe'

chroma_db = chromadb.Client(Settings(is_persistent=True,persist_directory=PERSISTANT_DIRECTORY))


st.title("Knowledge Retriever")

# Section 1: Choose from Predefined Collection
st.sidebar.subheader("CHOOSE A PREDEFINED RESOURCE:")
predefined_collection_name = st.sidebar.selectbox(
    "Choose a Predefined Collection",
    options=[""]+[c.name for c in chroma_db.list_collections()],
    key="predefined_collection_name"
)

st.sidebar.markdown("---")  

# Section 2: Upload File and Configure
st.sidebar.subheader("UPLOAD A NEW RESOURCE:")
file_type = st.sidebar.selectbox(
    "Choose File Type",
    options=["PDF", "DOCX"],
    key="file_type"
)

allowed_file_types = {
    "PDF": ["pdf"],
    "TEXT": ["txt"],
    "DOCX":['docx']
}
uploaded_file = st.sidebar.file_uploader(
    f"Upload a {file_type} File",
    type=allowed_file_types[file_type],
    key="file_uploader"
)
collection_name = st.sidebar.text_input("Collection Name", key="collection_name")

if (uploaded_file and collection_name) or predefined_collection_name:
    # Create a folder with the collection name under PDF_DIRECTORY
    if uploaded_file and collection_name:
        collection_path = Path(PDF_DIRECTORY) / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file as a PDF in the collection folder
        if file_type == 'PDF':
            path  = f"{collection_path}/{collection_name}.pdf"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            parser = PDFParser(path)
        elif file_type=="DOCX":
            path  = f"{collection_path}/{collection_name}.docx"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            parser=WordDocParser(path)
        
    client_model = initialize_client()

    if predefined_collection_name is not None and len(predefined_collection_name)>1:
        collection_name=predefined_collection_name
    
    if collection_name in [c.name for c in chroma_db.list_collections()]:
        st.sidebar.info(f"Collection '{collection_name}' already exists. Loading the model.")
        retriever_class = RetrieverModel(collection_name=collection_name,client_model=client_model, persistant_directory=PERSISTANT_DIRECTORY)
    
        retriever = retriever_class.get_retriever()
    else:
        st.sidebar.success(f"'{uploaded_file.name}' uploaded successfully.")
        st.sidebar.info("Parsing document ... This may take a few moments.")
        
        texts_list, tables_list, images_list = parser.parse()
        st.sidebar.info("Sucessfully Parsed Document.")
    
        retriever_class = RetrieverModel(collection_name=collection_name,client_model=client_model, persistant_directory=PERSISTANT_DIRECTORY)
        retriever = retriever_class.create_retriever(texts_list, tables_list, images_list)
        st.sidebar.success("Retriever is ready for questions.")
    st.subheader("Ask Question")
    question = st.text_input("Enter your question:")
    if question and retriever:
        response = response_model(client_model, retriever).invoke(question)
        if response:
            with st.expander("Answer"):
                if st.button("Play"):
                    language = 'en'
                    tts = gTTS(text=response["response"], lang=language, slow=False)
                    audio_stream = BytesIO()
                    tts.write_to_fp(audio_stream)
                    audio_stream.seek(0) 
                    st.audio(audio_stream, format="audio/mp3")
                st.write(response["response"])
                

            if st.button("Show Context"):
                st.subheader("Context")
                context = response["context"]

                if isinstance(context, dict):
                    texts = context.get("texts", [])
                    images = context.get("images", [])
                    for idx in range(len(texts)):
                        with st.expander(f"Text Source {idx + 1}"):
                            st.write(texts[idx].page_content)

                    for idx in range(len(images)):
                        with st.expander(f"Image Source {idx + 1}"):
                            img = display_base64_image(images[idx])
                            st.image(img)
                else:
                    st.write("No detailed context available.")
        else:
            st.subheader("Answer")
            st.write("No relevant information found.")


