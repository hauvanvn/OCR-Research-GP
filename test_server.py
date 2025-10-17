import requests

URL = "http://127.0.0.1:5000/api/ocr"

PDF_PATH = "sample.pdf"

# Send POST request
with open(PDF_PATH, "rb") as f:
    files = {'pdf': f}
    print("📤 Sending PDF to OCR API...")
    response = requests.post(URL, files=files)

# Print result
if response.status_code == 200:
    print("✅ OCR result received:")
    print(response.json())
else:
    print(f"❌ Error {response.status_code}: {response.text}")