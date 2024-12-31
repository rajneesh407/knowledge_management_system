import streamlit as st
from retriever import RetrieverModel
from model import response_model, initialize_client
from config import PERSISTANT_DIRECTORY
import chromadb
from chromadb.config import Settings
import os 
import pytesseract
from utils import display_base64_image
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="Knowledge Retriever", layout='wide', page_icon="📖")




os.environ["PATH"] += os.pathsep + 'C:\\Users\\rajneesh.jha\\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rajneesh.jha\\Tesseract-OCR\\tesseract.exe'

chroma_db = chromadb.Client(Settings(is_persistent=True,persist_directory=PERSISTANT_DIRECTORY))


st.title("Knowledge Retriever")
collection_name = st.sidebar.selectbox(
    "Choose a Pre-configured Collection",
    options=[""]+[c.name for c in chroma_db.list_collections()],
    key="collection_name"
)
if collection_name!="":
    client_model = initialize_client()
    st.sidebar.markdown("---")  
    st.sidebar.info("Loading the model.")
    retriever_class = RetrieverModel(collection_name=collection_name,client_model=client_model, persistant_directory=PERSISTANT_DIRECTORY)
    retriever = retriever_class.get_retriever()
    st.sidebar.success("Retriever is ready for questions.")    
    st.subheader("Ask Question")
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Enter your question:")
    with col2:
        response_model_name = st.selectbox("Response Model", options=["llama_11b", "llama_8b"], index=0)
    if question and retriever:
        response = response_model(client_model, retriever, response_model_name).invoke(question)
        if response:
            with st.expander("Answer"):
                if st.button("▶️"):
                    language = 'en'
                    tts = gTTS(text=response["response"], lang=language, slow=False)
                    audio_stream = BytesIO()
                    tts.write_to_fp(audio_stream)
                    audio_stream.seek(0) 
                    st.audio(audio_stream, format="audio/mp3")
                st.write(response["response"])

            if st.button("Show Summarized Context"):
                st.subheader("Context")
                context = response["context"]
                if isinstance(context, dict):
                    texts = context.get("texts", [])
                    images = context.get("images", [])
                    for idx in range(len(texts)):
                        with st.expander(f"Text Source {idx + 1}"):
                            # st.write(texts[idx].page_content)
                            st.write(retriever.vectorstore.get(where={"doc_id":texts[idx].metadata['doc_id']})['documents'][0])
                    for idx in range(len(images)):
                        with st.expander(f"Image Source {idx + 1}"):
                            img = display_base64_image(images[idx])
                            st.image(img)
                else:
                    st.write("No detailed context available.")
        else:
            st.subheader("Answer")
            st.write("No relevant information found.")


