import requests
import base64
from html import unescape
from email.message import EmailMessage
from email.utils import parseaddr
from typing import List, Dict
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override = True)
# Replace these with your actual credentials and refresh token
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
TOKEN_URL = 'https://oauth2.googleapis.com/token'

app = FastAPI()

@app.get("/refresh_access_token", response_model = str)
async def refresh_access_token():
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'refresh_token': REFRESH_TOKEN,
        'grant_type': 'refresh_token'
    }

    response = requests.post(TOKEN_URL, data=payload,)
    if response.status_code == 200:
        access_token = response.json()['access_token']
        print("New access token:", access_token)
        return access_token
    else:
        print("Failed to refresh token:", response.text)
        return None
async def get_user_email(access_token: str) -> str:
    headers = {"Authorization": f"Bearer {access_token}"}
    url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get profile: {response.text}")

    return response.json()["emailAddress"]


def decode_message_body(payload):
    """Decode base64 text/plain body from a Gmail message payload."""
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain':
                data = part['body'].get('data', '')
                return base64.urlsafe_b64decode(data.encode()).decode(errors='ignore')
    elif 'body' in payload:
        data = payload['body'].get('data', '')
        if data:
            return base64.urlsafe_b64decode(data.encode()).decode(errors='ignore')
    return ""

async def get_sent_message_pairs(access_token: str, max_messages: int = 100) -> List[Dict[str, str]]:
    """
    Builds input-response pairs using only the Sent Mail folder.
    
    Each pair:
      - Input: All prior messages in the thread before your sent message
      - Response: Your sent message

    :param access_token: Gmail OAuth token
    :param max_messages: Number of sent messages to process
    :return: List of input-response pairs
    """
    
    user_email = get_user_email(access_token)
    headers = {"Authorization": f"Bearer {access_token}"}
    pairs = []

    # 1. List messages in Sent folder
    sent_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    params = {
        "labelIds": "SENT",
        "maxResults": max_messages
    }
    sent_resp = requests.get(sent_url, headers=headers, params=params)
    
    if sent_resp.status_code != 200:
        raise Exception(f"Failed to list sent messages: {sent_resp.text}")

    sent_messages = sent_resp.json().get("messages", [])

    for msg_meta in sent_messages:
        msg_id = msg_meta["id"]

        # # 2. Fetch the full message to get thread ID
        # msg_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
        # msg_resp = requests.get(msg_url, headers=headers)
        # if msg_resp.status_code != 200:
        #     continue

        # msg_data = msg_resp.json()
        thread_id = msg_meta["threadId"]

        # 3. Fetch the full thread
        thread_url = f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}"
        thread_resp = requests.get(thread_url, headers=headers)
        if thread_resp.status_code != 200:
            continue

        messages = thread_resp.json().get("messages", [])
        # print(messages)
        conversation = []
        response_text = ""

        for m in messages:
            payload = m.get("payload", {})
            headers_list = payload.get("headers", [])
            from_header = next((h["value"] for h in headers_list if h["name"] == "From"), "Unknown")
            from_email = parseaddr(from_header)[1].lower()
            body = decode_message_body(payload) or m.get("snippet", "")
    
            line = f"{from_email}: {body.strip()}"

            if m["id"] == msg_id:
                response_text = user_email + ": " + body.strip()  # This is the current sent message (your reply)
                break  # Stop here; earlier messages form the input
            else:
                conversation.append(line)

        if conversation and response_text:
            pairs.append({
                "input": "\n".join(conversation),
                "response": response_text
            })

    return pairs


def get_sub_message_body(sub_json):

    # Fallback if no parts
    if 'data' in sub_json['message']:
        data = sub_json['message']['data']
        decoded_bytes = base64.urlsafe_b64decode(data + '==')
        return decoded_bytes.decode('utf-8')

    return "No plain text body found."

async def get_recent_thread_ids(access_token, history_id):
    url = 'https://gmail.googleapis.com/gmail/v1/users/me/history/'
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "startHistoryId": history_id,
        "historyTypes": "messageAdded"

    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        history = response.json().get("history", [])
        thread_ids = []
        for h in history:
            for msg in h.get("messagesAdded", []):
                thread_ids.append(msg["message"]["threadId"])
        return thread_ids
    else:
        raise Exception(f"Failed to get history: {response.text}")
    

async def get_thread_conversation(access_token, thread_id):
    url = f'https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}'
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to get thread: {response.text}")

    thread = response.json()
    messages = thread.get("messages", [])
    conversation = []

    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")

        # Try to get the plain text part of the message body
        parts = msg.get("payload", {}).get("parts", [])
        body = ""
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data", "")
                body = base64.urlsafe_b64decode(data.encode()).decode(errors="ignore")
                break
        else:
            # Fallback to snippet if no body found
            body = msg.get("snippet", "")

        # Clean up formatting
        sender = sender.split('<')[-1].strip('>')
        conversation.append(f"\"{sender}\":: {unescape(body.strip())}")

    conv_thread = "; ".join(conversation)
    to = sender
    
    return {"conv_thread":conv_thread,
            "to":to}
    
async def create_draft_message(ACCESS_TOKEN, TO, BODY, THREAD_ID, SUBJECT = "Re: Existing Conversation" ):
    # 1. Create the email message
    message = EmailMessage()
    message['To'] = TO
    # message['From'] = FROM
    message['Subject'] = SUBJECT
    message.set_content(BODY)

    # 2. Encode the message
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    # 3. Build the payload with thread ID
    payload = {
        "message": {
            "raw": encoded_message,
            "threadId": THREAD_ID
        }
    }

    # 4. Send request to create draft
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
        headers=headers,
        json=payload
    )

    # 5. Handle response
    if response.status_code == 200:
        print("✅ Draft created successfully.")
        print("Draft ID:", response.json()['id'])
    else:
        print("❌ Failed to create draft:")
        print(response.status_code, response.text)
        
        

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5050)      
    