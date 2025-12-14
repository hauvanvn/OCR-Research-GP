from flask import Flask, request, jsonify
from flask_cors import CORS 
import tempfile
from OCR import ocr_image

app = Flask(__name__)
CORS(app)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR: processing request!")

    data = request.get_json()

    url = data.get("slide")
    audio_arr = data.get("durations", [])

    callbackURL = data.get("callBackUrl")

    result = ocr_image(url, audio_arr)
    ocr_json = jsonify(result)

    callbackJSON = {
        "type": "ocr",
        "status": "success",
        "ocr_json": ocr_json
    }

    return request.post(callbackURL, callbackJSON)

if __name__ == '__main__':
    app.run(port=5000, debug=True)