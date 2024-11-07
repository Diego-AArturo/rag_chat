import chainlit as cl
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Configuración de variables de entorno y API de Google Gemini
load_dotenv()
api = os.getenv('api_gemini')
genai.configure(api_key=api)

# Configuración de Google Generative Model
PROYECT_ID = "chatmultimodal"
LOCATION = "us-central1"
model = genai.GenerativeModel("models/gemini-1.5-flash")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DB_PATH = "chroma_db"

# Crear los embeddings
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Cargar la base de datos vectorial desde el almacenamiento y asignar la función de embeddings
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr")

@cl.on_chat_start
async def start():
    """
    Inicia el chat cuando la aplicación se lanza.
    """
    message = cl.Message(content="Welcome to the Chat CIADET!")
    await message.send()

@cl.on_message
async def message(new_message: cl.Message):
    """
    Maneja los mensajes de los usuarios: primero recupera documentos relevantes de Chroma,
    luego genera una respuesta usando Google Generative AI.
    """
    user_query = new_message.content

    # Recuperar documentos relevantes usando Chroma
    retrieved_docs = retriever.invoke(user_query)
    context = " ".join([doc.page_content for doc in retrieved_docs])  # Combina el contenido recuperado

    # Generar respuesta con Google Generative Model, utilizando el contexto recuperado
    response = await model.generate_content_async(f'''contesta las preguntas basado solo en esta informacion: {context}
                                                    pregunta: {user_query}''')
    
    # Enviar respuesta al usuario
    message = cl.Message(content=response.text)
    await message.send()
