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
    "consistency": 0.15,
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

#Put grounding and consistency score into the same function to increase performance,
#having to make only 1 API call instead of 2.
def grounding_and_consistency_score(answer: str, context: str) -> tuple[float, float]:
    prompt = PromptTemplate.from_template("""Given this context from the EU AI Act:
    {context}

    And this answer:
    {answer}

    Task 1 - Grounding: Rate how well the answer is supported by the context above.
    - 1.0 = every statement in the answer comes directly from the context
    - 0.5 = some statements are supported, some are not
    - 0.0 = answer contains information not present in the context

    Task 2 - Consistency: Rate how consistent the context chunks are with each other.
    - 1.0 = all chunks discuss the same topic consistently
    - 0.5 = chunks are mostly consistent with minor differences
    - 0.0 = chunks contradict each other

    Respond with exactly two decimal numbers, rounded to the third decimal, separated by a comma.
    Grounding score first, consistency score second.
    Example format: 0.823, 0.914""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})
    
    try:
        parts = result.strip().split(",")
        g_score = max(0.0, min(1.0, float(parts[0].strip())))
        c_score = max(0.0, min(1.0, float(parts[1].strip())))
        return g_score, c_score
    except (ValueError, IndexError):
        return 0.5, 0.5

# source_domination_score was considered as a confidence metric to measure
# whether one chunk dominates the retrieval results (indicating a direct answer
# exists in the document). However, for legal documents like the EU AI Act,
# information is naturally distributed across multiple articles, oftentimes using the same words,
# causing all chunks to score similarly and the metric to systematically underperform.
# Still a viable metric to be used for other data types (eg. FAQs, API documentation, medical handbooks...)
# Replaced with consistency_score which better captures document coherence.

# def source_domination_score(source_docs, question) -> float:
#     if not source_docs:
#         return 0.0
#     if len(source_docs) == 1:
#         return 1.0
    
#     embeddings_list = [embeddings.embed_query(doc.page_content) for doc in source_docs]
#     question_embedding = embeddings.embed_query(question)
    
#     similarities = [cosine_similarity(v, question_embedding) for v in embeddings_list]
#     n = len(similarities)

#     #Score between 1/n - 1.0
#     d_score = max(similarities) / np.sum(similarities)
    
#     #Normalised to 0.0-1.0
#     d_score = (d_score - 1 / n) / (1 - 1 / n)
#     return d_score


def semantic_answer_similarity(question: str, answer: str) -> float:
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)
    return cosine_similarity(question_embedding, answer_embedding)


def compute_confidence(question: str, answer: str, source_docs: list) -> float:
    context = "\n\n".join([doc.page_content for doc in source_docs])
    
    r_score = retrieval_score(source_docs, question)
    g_score, c_score = grounding_and_consistency_score(answer, context)
    s_score = semantic_answer_similarity(question, answer)
    
    confidence = (
        WEIGHTS["retrieval"] * r_score +
        WEIGHTS["grounding"] * g_score +
        WEIGHTS["consistency"] * c_score +
        WEIGHTS["semantic"] * s_score
    )
    
    return round(confidence, 2)