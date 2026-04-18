# local_llm.py

import requests
from utils.message_blocks import debug_block

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2"  # or "llama3.2" or whatever you created

def ask_local_llm(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Send a prompt to the local Ollama model and return the full reply as a string.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    return data["message"]["content"]

def answer_user(user_message: str) -> str:
    # You can augment the prompt here with context, KG info, etc.
    response = ask_local_llm(user_message)
    with debug_block():
        print(response)
    return response

def is_recommendation(user_message: str) -> bool:
    prompt = f"Is the following sentence asking for movie recommendations, reply only 'yes' or 'no'? {user_message}"
    response = answer_user(prompt)
    if "yes" in response.lower():
        return True
    return False
