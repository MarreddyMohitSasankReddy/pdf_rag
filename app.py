import os
import tempfile
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyBm2i6wTD9IuPFDvfuOnYc3SBzpo7R2Zjc")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY as an environment variable.")
    st.stop()

st.set_page_config(page_title="PDF RAG with Gemini 2.0 Flash", layout="wide")
st.title("📄🤖 PDF RAG with Gemini 2.0 Flash")

# Sidebar
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 200, 1500, 500)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 50)
    model_name = "models/gemini-1.5-flash"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load and split PDF
    st.info("Processing PDF...")
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)

    # Embedding and Vector DB
    st.info("Creating vector index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embeddings)

    retriever = vectordb.as_retriever()

    # Gemini 2.0 Flash LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask a question based on the PDF:")

    if query:
        result = qa_chain({"query": query})
        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("Source Chunks"):
            for doc in result["source_documents"]:
                st.markdown(doc.page_content)

    os.unlink(tmp_path)
