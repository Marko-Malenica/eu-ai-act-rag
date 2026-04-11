import os
import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

WEIGHTS = {
    "retrieval": 0.35,
    "grounding": 0.35,
    "diversity": 0.15,
    "semantic": 0.15
}

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

#Embedding model nomic-embed-text ensures that all vector components are >= 0,
#thus making the result always positive (where cosine similarity usually has
#an output [-1, 1]).
def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieval_score(source_docs, question: str) -> float:
    if not source_docs:
        return 0.0
    question_embedding = embeddings.embed_query(question)
    similarities = []
    for doc in source_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarities.append(cosine_similarity(question_embedding, doc_embedding))
    return float(np.mean(similarities))


def grounding_score(answer: str, context: str) -> float:
    prompt = PromptTemplate.from_template("""Given this context from the EU AI Act:
    {context}

    And this answer:
    {answer}

    Rate from 0.0 to 1.0 (the average being 0.5) how well the answer is grounded in the context.
    A score of 1.0 means every claim in the answer is directly supported by the context.
    A score of 0.0 means the answer contains information not present in the context.
    Return only a decimal number between 0.0 and 1.0, nothing else.""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})
    
    try:
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5


def source_domination_score(source_docs, question) -> float:
    if not source_docs:
        return 0.0
    if len(source_docs) == 1:
        return 1.0
    
    embeddings_list = [embeddings.embed_query(doc.page_content) for doc in source_docs]
    question_embedding = embeddings.embed_query(question)
    
    similarities = [cosine_similarity(v, question_embedding) for v in embeddings_list]
    n = len(similarities)

    #Score between 1/n - 1.0
    d_score = max(similarities) / np.sum(similarities)
    
    #Normalised to 0.0-1.0
    d_score = (d_score - 1 / n) / (1 - 1 / n)
    return d_score


def semantic_answer_similarity(question: str, answer: str) -> float:
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)
    return cosine_similarity(question_embedding, answer_embedding)


def compute_confidence(question: str, answer: str, source_docs: list) -> float:
    context = "\n\n".join([doc.page_content for doc in source_docs])
    
    r_score = retrieval_score(source_docs, question)
    g_score = grounding_score(answer, context)
    d_score = source_domination_score(source_docs, question)
    s_score = semantic_answer_similarity(question, answer)
    
    confidence = (
        WEIGHTS["retrieval"] * r_score +
        WEIGHTS["grounding"] * g_score +
        WEIGHTS["diversity"] * d_score +
        WEIGHTS["semantic"] * s_score
    )
    
    return round(confidence, 2)