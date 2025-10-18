import streamlit as st
from RAG.loader import load_pdf, chunk_text
from RAG.database import add_to_chroma
from RAG.ask import query_rag
import os

st.title("📚 Lecture Notes Q&A (RAG System)")

uploaded_file = st.file_uploader("Upload your lecture notes", type=["pdf"])
question = st.text_input("Ask a question about your notes")

if uploaded_file:
    save_path = os.path.join("data", "uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    text = load_pdf(save_path)
    chunks = chunk_text(text)
    add_to_chroma(chunks)
    st.success("✅ Notes uploaded and processed!")

if question and st.button("Ask"):
    answer = query_rag(question)
    st.subheader("💬 Answer")
    st.write(answer)
