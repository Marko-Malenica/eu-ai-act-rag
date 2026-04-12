import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from backend.rag import query_eu_ai_act
from backend.confidence import compute_confidence
from backend.database import init_db, get_db, Conversation, Feedback

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="EU AI Act RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str

class QueryResponse(BaseModel):
    id: int
    answer: str
    confidence: float
    sources: list[str]
    flag: str | None

class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    result = query_eu_ai_act(request.question)
    
    confidence = float(compute_confidence(
        question=request.question,
        answer=result["answer"],
        source_docs=result["source_docs"]
    ))
    
    flag = None
    if confidence < 0.4:
        flag = "Low confidence — consult official EU AI Act source"
    elif confidence < 0.6:
        flag = "Medium confidence — verify with official source"
    
    sources_str = ", ".join(result["sources"])
    
    conversation = Conversation(
        session_id=request.session_id,
        question=request.question,
        answer=result["answer"],
        confidence=confidence,
        sources=sources_str
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return QueryResponse(
        id=conversation.id,
        answer=result["answer"],
        confidence=confidence,
        sources=result["sources"],
        flag=flag
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    if request.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    feedback = Feedback(
        conversation_id=request.conversation_id,
        rating=request.rating
    )
    db.add(feedback)
    db.commit()
    
    return {"status": "ok"}


@app.get("/history/{session_id}")
def history(session_id: str, db: Session = Depends(get_db)):
    conversations = db.query(Conversation).filter(Conversation.session_id == session_id).order_by(
        Conversation.timestamp.desc()
    ).limit(20).all()
    
    return [
        {
            "id": c.id,
            "question": c.question,
            "answer": c.answer,
            "confidence": c.confidence,
            "sources": c.sources.split(", "),
            "timestamp": c.timestamp
        }
        for c in conversations
    ]