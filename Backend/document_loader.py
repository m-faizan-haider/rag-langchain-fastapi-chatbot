# Backend/document_loader.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
load_dotenv()
from Backend.config import DATA_DIR

def load_documents() -> List:
    """
    Load documents from DATA_DIR using LangChain loaders.
    Returns a list of LangChain Document objects.
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    loaders = [
        DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader),
        # Add more loaders here if required (pptx, md, docx)
    ]

    docs = []
    for loader in loaders:
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"⚠️ Skipped some files due to loader error: {e}")

    return docs
