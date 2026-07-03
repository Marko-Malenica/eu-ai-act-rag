import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-5.4", temperature=0)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="eu_ai_act_openai",
    connection=DATABASE_URL,
)

prompt = PromptTemplate.from_template("""You are an expert on the EU AI Act. 
Use the following context to answer the question.
If you don't know the answer based on the context, say so clearly.
Always cite which articles or sections you are referencing.

Context: {context}

Question: {question}

Answer:""")

chain = prompt | llm | StrOutputParser()

def query_eu_ai_act(question: str) -> dict:
    results = vectorstore.similarity_search_with_relevance_scores(question, k=5)
    source_docs = [doc for doc, _ in results]
    similarities = [score for _, score in results]

    context = "\n\n".join(doc.page_content for doc in source_docs)
    answer = chain.invoke({"context": context, "question": question})

    sources = list(set([
        f"Page {doc.metadata.get('page', 'unknown')}" for doc in source_docs
    ]))

    return {
        "answer": answer,
        "sources": sources,
        "source_docs": source_docs,
        "similarities": similarities
    }