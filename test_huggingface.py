# test_huggingface.py - Run this first to test your token
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")
print(f"API Key found: {api_key[:10]}..." if api_key else "No API key found")

if api_key:
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Test simple text generation to verify token works
    response = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers=headers,
        json={"inputs": "Hello, how are you?"}
    )
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Token works!")
    else:
        print(f"❌ Token failed: {response.text}")