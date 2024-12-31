import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from model import get_embedding_function, summarization_model_image, summarization_model_text
from config import EMBEDDING_FUNCTION_NAME, ID_KEY, PERSISTANT_DIRECTORY
from langchain.storage._lc_store import create_kv_docstore
from custom_chroma_docstore import ChromaStore
import chromadb
from chromadb.config import Settings
from prompt_library import summarization_prompt_text_or_table


class RetrieverModel:
    def __init__(self,  
                 collection_name, 
                 client_model, 
                 id_key=None, 
                 persistant_directory=None):
        persistant_directory=persistant_directory or PERSISTANT_DIRECTORY
        self.client_model=client_model
        self.vectorstore = Chroma(
            collection_name=collection_name, 
            embedding_function=get_embedding_function(model_name=EMBEDDING_FUNCTION_NAME), 
            persist_directory=persistant_directory
        )
        self.id_key = id_key or ID_KEY
        self.collection_name = collection_name
        self.chroma_db = chromadb.Client(Settings(is_persistent=True,persist_directory=persistant_directory))
        self.store = create_kv_docstore(ChromaStore(persistant_directory, collection_name))

    def collection_exists(self):
        if self.collection_name in [c.name for c in self.chroma_db.list_collections()]:
            return True 
        return False
    
    def get_retriever(self):
        if self.collection_name in [c.name for c in self.chroma_db.list_collections()]:
            retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    docstore=self.store,
                    id_key=self.id_key,
                )
            return retriever
        else:
            raise AssertionError(f"The {self.collection_name} does not exists.")
        
    

    def summarize_document_and_image(self, doc, is_image=False, model_name_image="llama_11b", model_name_text="llama_8b"):
        from langchain_core.output_parsers import StrOutputParser
        if is_image:
            model_output=summarization_model_image(self.client_model, doc, model_name_image)
            return StrOutputParser().parse(model_output)
        model_output = summarization_model_text(self.client_model, [{"role":"user","content":summarization_prompt_text_or_table.format(text_or_table=doc)}],model_name_text)
        return StrOutputParser().parse(model_output)


    def add_documents(self, texts_list, tables_list=[], images_list=[]):
        text_summaries = [self.summarize_document_and_image(i) for i in texts_list]
        table_summaries = [self.summarize_document_and_image(i.metadata.text_as_html) for i in tables_list]
        image_summaries = [self.summarize_document_and_image(i, is_image=True) for i in images_list]

        if len(texts_list) > 0:
            doc_ids = [str(uuid.uuid4()) for _ in texts_list]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
            ]
            self.vectorstore.add_documents(summary_texts)
            self.store.mset(list(zip(doc_ids, [
                Document(
                    page_content=str(text), 
                    metadata={self.id_key: doc_ids[i], 'page_number': text.metadata.page_number}
                ) for i, text in enumerate(texts_list)
            ])))

        if len(tables_list) > 0:
            table_ids = [str(uuid.uuid4()) for _ in tables_list]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
            ]
            self.vectorstore.add_documents(summary_tables)
            self.store.mset(list(zip(table_ids, [
                Document(
                    page_content=str(table), 
                    metadata={self.id_key: table_ids[i], 'page_number': table.metadata.page_number}
                ) for i, table in enumerate(tables_list)
            ])))

        if len(images_list) > 0:
            img_ids = [str(uuid.uuid4()) for _ in images_list]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
            ]
            self.vectorstore.add_documents(summary_img)
            self.store.mset(list(zip(img_ids, [
                Document(
                    page_content=image_data, 
                    metadata={self.id_key: img_ids[i]}
                ) for i, image_data in enumerate(images_list)
            ])))