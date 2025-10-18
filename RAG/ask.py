import os
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

load_dotenv()


def _get_openai_client():
    try:
        from openai import OpenAI

        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAI client. Make sure the `openai` package is installed "
            "and OPENAI_API_KEY is set. Original error: " + str(e)
        )


def _get_chroma_collection(name: str = "notes"):
    try:
        import chromadb

        chroma_client = chromadb.Client()
        return chroma_client.get_or_create_collection(name=name)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize chromadb collection. "
            "See RAG/database.py for more details. Original error: "
            f"{e}"
        )


def query_rag(question: str) -> str:
    collection = _get_chroma_collection()

    client = _get_openai_client()

    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3
    )

    context = "\n\n".join(results['documents'][0])

    prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
