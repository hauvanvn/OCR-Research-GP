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
    url = data.get("url")
    audio_arr = data.get("numbers", [])

    result = ocr_image(url, audio_arr)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)