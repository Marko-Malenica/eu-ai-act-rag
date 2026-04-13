import uuid
import requests
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="EU AI Act Assistant", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

st.title("EU AI Act Assistant")
st.caption("Ask questions about the EU AI Act. Responses include confidence scores.")


def display_assistant_message(message: dict):
    st.write(message["content"])
    confidence = message["confidence"]
    
    if confidence == 0.0:
        st.error(message.get("flag", "Question out of scope"))
    else:
        if confidence >= 0.8:
            st.success(f"Confidence: {confidence}")
            st.success("High confidence ✅")
        elif confidence >= 0.6:
            st.warning(f"Confidence: {confidence}")
            st.warning("Medium confidence — verify with official source")
        else:
            st.error(f"Confidence: {confidence}")
            st.error("Low confidence — consult official EU AI Act source")
    st.caption(f"Sources: {', '.join(message['sources'])}")
    col1, col2 = st.columns([1, 1])
    if message["id"] in st.session_state.feedback_given:
        st.caption("✓ Feedback submitted")
    else:
        with col1:
            if st.button("👍", key=f"up_{message['id']}"):
                requests.post(f"{API_URL}/feedback", json={
                    "conversation_id": message["id"],
                    "rating": 1
                })
                st.session_state.feedback_given.add(message["id"])
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{message['id']}"):
                requests.post(f"{API_URL}/feedback", json={
                    "conversation_id": message["id"],
                    "rating": -1
                })
                st.session_state.feedback_given.add(message["id"])
                st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_assistant_message(message)
        else:
            st.write(message["content"])


if question := st.chat_input("Ask about the EU AI Act..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Searching EU AI Act..."):
        response = requests.post(f"{API_URL}/query", json={
            "question": question,
            "session_id": st.session_state.session_id
        })
        data = response.json()

    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
        "confidence": data["confidence"],
        "sources": data["sources"],
        "flag": data.get("flag"),
        "id": data["id"]
    })
    
    st.rerun()