import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from bot import Models, Chatbot, recommend_eng, recommend_kr
import json

### Loading the Huggingface Token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key= hf_token,
)

bot = Chatbot(model=Models.LLAMA)
bot.review_response()
# recommend()

chat_history = {}
json.dumps(chat_history)

with open(bot.review_response(), mode="w", encoding="utf-8") as write_file:
    json.dump(chat_history, write_file)

### Prompt that asks in Korean and English which language you want to choose

### Function that saves data, e.g. prompt to translate then saves the prompt and response
# def save_data():
    
