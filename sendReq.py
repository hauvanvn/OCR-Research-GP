import requests
import json

with open('ocr_result.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    requests.post("http://localhost:3001/video-generation/1c938d01-fdd1-4ec5-a7e5-1f3c9c42137f/generate/callback", json=data)