from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configuración de parámetros
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PDF_PATH = 'BOT_CIADET EDITADO HERMES 94 PAG..pdf'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_PATH = "chroma_db"  # Carpeta donde se guardará la base de datos de Chroma

def csv_loader(path):
    """Carga datos de un archivo CSV y devuelve los documentos procesados."""
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(
        file_path=path,
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['Index', 'Height', 'Weight']
        }
    )
    return loader.load()
def txt_loader(path):
    """Carga datos de un archivo de texto y devuelve el contenido como una cadena."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
def pdf_loader(path):
    """Carga datos de un archivo PDF y devuelve los documentos procesados."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Divide el texto en fragmentos para el procesamiento de RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([text])

# Cargar y procesar el PDF
pdf_docs = pdf_loader(PDF_PATH)
text = " ".join([doc.page_content for doc in pdf_docs])

# Dividir el texto en fragmentos
docs = chunk_text(text)

# Crear embeddings y base de datos vectorial
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)

print("Base de datos vectorial creada y guardada.")