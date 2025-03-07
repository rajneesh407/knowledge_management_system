import streamlit as st

st.set_page_config(page_title="Home", page_icon="👋", layout="wide")
import streamlit as st

from PIL import Image
import streamlit as st

st.title("🎓 AI-Powered Academic Resource Management")

st.write(
    """
    Unlock the future of academic resource management with our **AI-driven platform** that transforms how students store, access, and interact with learning materials. Designed to streamline research and study workflows, our system leverages **Generative AI, Retrieval-Augmented Generation (RAG), and Vector Embeddings** to provide seamless access to academic content across multiple formats, including **PDFs, documents, images, audio, and video**.  
    
    ## 🔑 Key Features  
    - ✅ **Intelligent Search & Summarization** – Retrieve relevant information instantly with **semantic search** and AI-powered document summaries.  
    - ✅ **Multimodal Resource Integration** – Manage all types of academic materials—text, video, audio, and images—in one place.  
    - ✅ **Context-Aware Responses** – Get precise answers and insights using **LLMs** and advanced retrieval techniques.  
    - ✅ **Automated Metadata Extraction** – Organize and categorize resources efficiently with AI-driven indexing.  
    - ✅ **User-Friendly Interface** – A simple and intuitive UI that makes uploading, querying, and navigating academic resources effortless.  
    - ✅ **Secure & Scalable** – Powered by **Vector Databases** (Chroma DB, Qdrant, Weaviate, Pinecone) for fast, efficient, and privacy-focused information retrieval.  
    
    ## ⚙️ How It Works  
    - 📌 **Upload Resources** – Add PDFs, videos, audio files, or handwritten notes.  
    - 📌 **AI Processing** – Our system extracts, summarizes, and indexes content using **vector embeddings and chunking techniques**.  
    - 📌 **Query Your Knowledgebase** – Ask questions and receive **contextually accurate answers** in real time.  
    - 📌 **Retrieve & Summarize Video/Audio** – Extract key takeaways from lectures, podcasts, or research talks with **AI-powered transcription and analysis**.  
    
    Our platform **reduces search time, enhances learning efficiency, and ensures students get the right information when they need it.**  
    
    🚀 **Experience the power of AI-driven academic assistance today!**  
    """
)
