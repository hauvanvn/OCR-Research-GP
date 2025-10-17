from flask import Flask, request, jsonify
import tempfile
from OCR import ocr_pdf
app = Flask(__name__)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    file = request.files['pdf']
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        result = ocr_pdf(tmp.name)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)