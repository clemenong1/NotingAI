
import os
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv is optional — if it's not installed, provide a no-op so imports
    # don't fail in environments where environment variables are set externally.
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
    """Create and return a chromadb collection.

    This defers importing and creating the chromadb client until it's actually
    needed at runtime so that importing this module (for example by Streamlit
    on app startup) doesn't trigger chromadb's dependency chain which may
    import `posthog` and fail on older Python versions (TypeAlias/PEP 585
    related issues).
    """
    try:
        import chromadb

        chroma_client = chromadb.Client()
        return chroma_client.get_or_create_collection(name=name)
    except Exception as e:
        # Provide a helpful error that suggests likely fixes without raising
        # the raw chromadb/posthog stack trace during import-time.
        raise RuntimeError(
            "Failed to initialize chromadb collection. "
            "This commonly happens if `posthog` (a chromadb dependency) uses "
            "type annotations not supported by older Python versions (e.g. <3.9). "
            "Recommended fixes: (1) upgrade your Python to 3.9+, or "
            "(2) pin `posthog` to a compatible version in your environment, "
            "or (3) install chromadb in a clean virtualenv. Original error: "
            f"{e}"
        )


def add_to_chroma(chunks):
    collection = _get_chroma_collection()

    for i, chunk in enumerate(chunks):
        client = _get_openai_client()
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk-{i}"]
        )
