from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import subprocess
import requests
import time
from gmail_funcs import get_sub_message_body,  get_thread_conversation, get_recent_thread_ids, create_draft_message
import asyncio


async def get_access_token():
    url = "http://0.0.0.0:5050/refresh_access_token/"

    
    response = requests.get(url, )
    if response.status_code == 200:
        #print(response.json())
        return response.json()
    else:
        print("Failed to get response:", response.text)
    return None

async def get_response(conv_thread):
    url = "http://0.0.0.0:5000/generate_response/"

    params = {"conv_thread": conv_thread}
    response = requests.post(url, json = params,)
    if response.status_code == 200:
        #print(response.json())
        return response.json()['reply_suggestion'].split("Answer:")[-1].split("Analysis:")[0]
    else:
        print("Failed to get response:", response.text)
    return None

app = FastAPI()

# Create a global lock
request_lock = asyncio.Lock()

class Request(BaseModel):
    content: dict
@app.post("/secure-webhook",)
async def secure_webhook(request:dict):
    
    async with request_lock:
        data = request
        access_token = await get_access_token()


        history_id = int(eval(get_sub_message_body(request))["historyId"])
        time.sleep(30)

        thread_ids = await get_recent_thread_ids(access_token, history_id)

        for thread_id in thread_ids:

            conv_details = await get_thread_conversation(access_token, thread_id)

            conv_thread = conv_details.get("conv_thread","")
            to = conv_details.get("to","")
            
            response = await get_response(conv_thread)
            print("Response Generated")

            await create_draft_message(access_token, to, response, thread_id, SUBJECT = "Re: Existing Conversation" )
            print("Draft Created")
        #print("working")
        return 200
    
# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)