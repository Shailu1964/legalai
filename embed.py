import utils as Utils
import os as OS
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


def pdf_to_text(url):
    """Downloads a PDF from a URL and extracts its text."""
    try:
        response = requests.get(url)
        pdf_data = response.content
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred during PDF conversion: {e}")
        return ""


def split_text_into_sections(text, min_chars_per_section):
    """Splits raw text into manageable chunks for the vector database."""
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2

    if current_section:
        sections.append(current_section.strip())

    return sections


def embed_text_in_chromadb(text, document_name, document_description, persist_directory=Utils.DB_FOLDER):
    """
    Creates embeddings using a free HuggingFace model and stores them in ChromaDB.
    """
    # 1. Initialize Free Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Prepare Chunks
    raw_sections = split_text_into_sections(text, 1000)

    # 3. Convert strings to LangChain Document objects
    documents = [
        Document(
            page_content=section,
            metadata={"name": document_name, "description": document_description}
        )
        for section in raw_sections
    ]

    print(f"Generating embeddings for {len(documents)} chunks...")

    # 4. Initialize and Populate ChromaDB
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name='collection_1'
    )

    print(f"Successfully added documents to {persist_directory}")


if __name__ == "__main__":
    document_name = "Artificial Intelligence Act"
    document_description = "Artificial Intelligence Act"

    print(f"Downloading and parsing: {Utils.EUROPEAN_ACT_URL}")
    text = pdf_to_text(Utils.EUROPEAN_ACT_URL)

    if text:
        embed_text_in_chromadb(text, document_name, document_description)
    else:
        print("Failed to extract text from the PDF.")