import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="eu_ai_act",
    connection=DATABASE_URL,
)

llm = ChatOllama(model="llama3.2", temperature=0)

prompt = PromptTemplate.from_template("""You are an expert on the EU AI Act. 
Use the following context to answer the question.
If you don't know the answer based on the context, say so clearly.
Always cite which articles or sections you are referencing.

Context: {context}

Question: {question}

Answer:""")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def query_eu_ai_act(question: str) -> dict:
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    
    sources = list(set([
        f"Page {doc.metadata.get('page', 'unknown')}" for doc in source_docs
    ]))
    
    return {
        "answer": answer,
        "sources": sources,
        "source_docs": source_docs
    }