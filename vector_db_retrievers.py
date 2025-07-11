### vector_db_retrievers.py
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
import pickle
import os


pickle_path = os.path.join('data', 'docs', 'new_docs.pkl')
with open(pickle_path, 'rb') as file:
    new_docs = pickle.load(file)

model_path = os.path.join('models', 'embedding_model', 'bge-m3')
#model_path = os.getenv('EMBEDDING_MODEL_PATH')

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
    
)

faiss_index_path = os.path.join('data', 'faiss_index')
vectorstore = FAISS.load_local(
    faiss_index_path,
    hf_embeddings,
    allow_dangerous_deserialization=True
)


faiss_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 9}
)

bm25_retriever = BM25Retriever.from_documents(new_docs)
bm25_retriever.k = 2


ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4],
    c=60,
    id_key="id"
)
