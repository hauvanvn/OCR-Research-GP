import requests
import json
import time

# --- Configuration ---
BASE_URL = "https://taren-counteractive-overtolerantly.ngrok-free.dev/api/ocr"

# We use httpbin.org as the callback. 
# It accepts the POST from your server and returns 200 OK + the data it received.
CALLBACK_URL = "" 

# A simple, publicly accessible PDF for testing (1 page)
# TEST_PDF_URL = "https://drive.google.com/uc?export=download&id=1qHo2bFLHkobLlwhGqtd5qjZQu-h5K9wz"
TEST_PDF_URL = "https://drive.google.com/uc?export=download&id=1wN23WvOXKVhMm-6WpPVPJjTpXwOAXy_m"

def test_ocr_flow():
    # 1. Define the payload expected by your server
    payload = {
        "slide": TEST_PDF_URL,
        "durations": [5, 10, 15, 5, 20, 12], # Simulation for 3 pages/segments
        "callBackUrl": CALLBACK_URL
    }

    print(f"🚀 Sending request to: {BASE_URL}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    print("-" * 40)

    try:
        # 2. Send the POST request
        start_time = time.time()
        response = requests.post(BASE_URL, json=payload)
        end_time = time.time()

        # 3. Handle the response
        if response.status_code == 200:
            print(f"✅ Success! (Time taken: {end_time - start_time:.2f}s)")
            
            # Since your server returns the response from httpbin, 
            # we can inspect what httpbin received from your server.
            httpbin_data = response.json()
            
            # The data your server sent to the callback is inside 'json'
            server_sent_data = httpbin_data.get('json', {})
            
            print("\n📬 What the Callback URL received:")
            print(f"Type: {server_sent_data.get('type')}")
            print(f"Status: {server_sent_data.get('status')}")
            
            # The actual OCR result string
            ocr_content = server_sent_data.get('ocr_json')
            print(f"OCR Data Length: {len(str(ocr_content))} chars")
            print(f"OCR Data Preview: {str(ocr_content)[:100]}...")
            
        else:
            print(f"❌ Failed with Status Code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Ensure 'server.py' is running on port 5000.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_ocr_flow()