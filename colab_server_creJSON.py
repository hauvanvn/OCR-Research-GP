import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS 
from OCR import ocr_image

app = Flask(__name__)
CORS(app)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR: processing request!")

    data = request.get_json()
    print("Package receive from BE:", data)

    url = data.get("slide")
    audio_arr = data.get("durations", [])
    callbackURL = data.get("callBackUrl")

    result = ocr_image(url, audio_arr)
    ocr_json = json.dumps(result, ensure_ascii=False)

    callbackJSON = {
        "type": "ocr",
        "status": "success",
        "ocr_json": ocr_json   
    }
    print(callbackJSON)

    try:
        # Create a 'logs' directory if it doesn't exist
        log_dir = "ocr_results"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filename = f"{log_dir}/ocr_result.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(callbackJSON, f, ensure_ascii=False, indent=4)
        
        print(f"Callback saved to: {filename}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    return jsonify({"message": "OCR done"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)