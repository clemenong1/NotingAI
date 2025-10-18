# NotingAI
RAG (Retrieval-Augmented Generation) combines:
- Retrieval — find relevant context (e.g. lecture notes) from your knowledge base.
- Generation — feed that context to an LLM (like OpenAI’s GPT) to answer the question.

Flow:
User question → Embed → Retrieve relevant chunks → Send context + question to GPT → Answer
