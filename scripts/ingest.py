import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def ingest():
    print("Loading PDF...")
    loader = PyPDFLoader("data/eu_ai_act.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Storing in PostgreSQL...")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="eu_ai_act",
        connection=DATABASE_URL,
    )
    vectorstore.add_documents(chunks)
    print("Done!")

if __name__ == "__main__":
    ingest()