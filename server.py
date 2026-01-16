from flask import Flask, request, jsonify
from flask_cors import CORS 
import tempfile
import json
from diff import ocr_image
import requests

app = Flask(__name__)
CORS(app)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR: processing request!")

    data = request.get_json()

    print("Package receive from BE: ",data)

    url = data.get("slide")
    audio_arr = data.get("durations", [])
    
    callbackURL = data.get("callBackUrl")

    result = ocr_image(url, audio_arr)
    #Chỗ này đổi sang string cho phù hợp với kiểu dữ liệu string của ocr_json trong db
    result_string = json.dumps(result, separators=(',', ':'), ensure_ascii=False)

    callbackJSON = {
        "type": "ocr",
        "status": "success",
        "ocr_json": result_string
    }

    with open("result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
    return callbackJSON
    # return requests.post(callbackURL, json=callbackJSON)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)