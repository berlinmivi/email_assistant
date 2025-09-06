import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gmail_funcs import get_user_email

from create_cvector_db import extract_important_words, get_embeddings

import requests
import asyncio

async def get_access_token():
    url = "http://0.0.0.0:5050/refresh_access_token/"

    response = requests.get(url,)
    if response.status_code == 200:
        #print(response.json())
        return response.json()
    else:
        print("Failed to get response:", response.text)
        return None

access_token = get_access_token()
user_email = get_user_email(access_token)

# === CONFIG ===
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# === Load embedding model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(MODEL_NAME,)#.to(DEVICE)
model.eval()
print("#####################EmbedModel loaded########################")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 1
# QUERY = """Dr_Willis:  i wonder if their irc client is misconfigured.\nActionParsnip: sounds like misconfigured user\nDr_Willis:  from what i read of backtrack4 most users in here asking about it.. really should not be  using it.. :P  my wife will not touch coke its pepsi or nothing  for her. :)\nActionParsnip: exactly they just care about users\nManDay: I'll check it out, thank you!","""
SAVE_DIR = "cdb_index"

# === Load CDB and metadata ===
import chromadb

SAVE_DIR = "cdb_index"
chroma_client = chromadb.PersistentClient(SAVE_DIR)
collection = chroma_client.get_collection(name="my_collection")


with open(f"{SAVE_DIR}/texts.pkl", "rb") as f:
    texts = pickle.load(f)
print("########################VectorDB loaded########################")

# # === Load chat model (instruct-tuned) ===
# chat_tokenizer = ChatTokenizer.from_pretrained(MODEL_NAME)
chat_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ).to(DEVICE)
chat_model.eval()
print("########################ChatModel loaded########################")

#define input model for chat endpoint
class GenerateRequest(BaseModel):
    conv_thread: str

class RAGResponse(BaseModel):
    reply_suggestion:str
    history: str


app = FastAPI()

@app.post("/generate_response", response_model = RAGResponse)
async def generate_response(request:GenerateRequest):
    """endpoint for generating a response based on the current conversation"""
    try:
        start = time.process_time()    
        
        results = collection.query(
            query_embeddings = get_embeddings(MODEL_NAME, extract_important_words([request.conv_thread]), ), # Chroma will embed this for you
            n_results= 1 # how many results to return
        )
        
        context = results['documents'][0][0]
        
        # === Format prompt for chat model ===
        prompt = f"""You are an expert assistant. The following context contains the thread conversation that took place earlier as input and the response given by
        {user_email} as reponse. Analyse the context and reply to the query below as {user_email}. Do not include the ideas that lead to the result in the response

        Context:
        {context}

        Question: {request.conv_thread}
        Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt",truncation=True, padding=True)
        with torch.no_grad():
            outputs = chat_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.3)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end = time.process_time()
        print("Response time: ", end - start)

        # return RAGResponse(reply_suggestion = response.split("Answer:")[1].split("\n")[0],
        #                    history = context)
        return RAGResponse(reply_suggestion = response,
                           history = context)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)