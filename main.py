import chainlit as cl
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api = os.getenv('api_gemini')
genai.configure(api_key=api)

PROYECT_ID ="chatmultimodal"
LOCATION = "us-central1"
model = genai.GenerativeModel(
    "models/gemini-1.5-flash"
)

@cl.on_chat_start
async def start():
    '''
    Star the chat when the application launches.
    '''
    menssage = cl.Message(content="Welcome Chat")
    await menssage.send()

@cl.on_message
async def message(new_message: cl.Message):
    '''
    handle messages from user

    '''
    content_message = new_message.content
    content_elements = new_message.elements
    response = await model.generate_content_async(content_message)
    message = cl.Message(content=response.text)
    await message.send()