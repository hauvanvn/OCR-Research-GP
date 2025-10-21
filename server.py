from flask import Flask, request, jsonify
from flask_cors import CORS 
import tempfile
from OCR import ocr_image

app = Flask(__name__)
CORS(app)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR: processing request!")
    file = request.files['pdf']
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        file.save(tmp.name)
        result = ocr_image(tmp.name)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)