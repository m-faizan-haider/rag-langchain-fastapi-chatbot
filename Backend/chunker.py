# Backend/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from Backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter with
    immutable chunk params from config.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    return splits
