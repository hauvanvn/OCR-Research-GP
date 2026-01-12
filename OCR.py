import easyocr
import cv2
import json
import numpy as np
import regex
import os
import urllib.request #Use for downloading files from URL
import fitz
import shutil #Use to clear folder data

import pandas as pd
import string

import nltk
from nltk.corpus import gutenberg, stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
punctuations = set(string.punctuation)

import spacy
from spacy.lang.vi import Vietnamese
from spacy.lang.en import English

MODELS = {
    "en": {"reader": None, "nlp": None},
    "vi": {"reader": None, "nlp": None}
}

def get_resources(lang):
    if MODELS[lang]["reader"] is None:
        if lang == "vi":
            MODELS["vi"]["reader"] = easyocr.Reader(['vi'])
            MODELS["vi"]["nlp"] = Vietnamese()
            MODELS["vi"]["nlp"].add_pipe('sentencizer')
        else:
            MODELS["en"]["reader"] = easyocr.Reader(['en'])
            MODELS["en"]["nlp"] = English()
            MODELS["en"]["nlp"].add_pipe('sentencizer')
            
    return MODELS[lang]["reader"], MODELS[lang]["nlp"]

def load_vietnamese_stopwords(path="vietnamese-stopwords.txt"):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())
    
EN_STOPWORDS = set(stopwords.words('english'))
VI_STOPWORDS = load_vietnamese_stopwords()

def get_imgs(url):
    download_dir = "data"
    img_dir = os.path.join(download_dir, "imgs")

    os.makedirs(img_dir, exist_ok=True)

    pdf_path = os.path.join(download_dir, "lesson.pdf")

    urllib.request.urlretrieve(url, pdf_path)

    image_paths = []

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(img_dir, f"page_{i+1}.jpg")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def sort_words_into_lines(results, y_tolerance=15):
    """
    results: list of dicts with keys: text, x, y, bbox
    y_tolerance: pixel threshold for grouping into the same line
    """
    lines = []

    for word in results:
        placed = False
        for line in lines:
            # Compare this word's baseline y to the mean baseline y of the current line
            mean_y = sum(w["avg_y"] for w in line) / len(line)
            if abs(word["avg_y"] - mean_y) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    # Sort lines top-to-bottom
    lines.sort(key=lambda line: sum(w["avg_y"] for w in line) / len(line))

    # Sort each line left-to-right
    for line in lines:
        line.sort(key=lambda w: w["x"])

    return lines

def merge_bounding_boxes(boxes):
    """Merge multiple bounding boxes into one."""
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def is_stopword(token_text: str) -> bool:
    t = token_text.lower()
    return t in EN_STOPWORDS or t in VI_STOPWORDS

def ocr_image(image_path, audio_arr, lang):
    imgs_path = get_imgs(image_path)

    reader, nlp = get_resources(lang)

    OCRjson = {}
    for i in range(1, len(audio_arr)):
        audio_arr[i] += audio_arr[i - 1]
        
    for image_path, renderTime in zip(imgs_path, audio_arr):
        # Try reading with cv2
        image = cv2.imread(image_path)

        # If cv2 fails, try reading via numpy buffer
        if image is None or image.size == 0:
            print("⚠️ cv2.imread failed, retrying with np.fromfile")
            try:
                data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception as e:
                return {"error": f"Failed to decode image: {str(e)}"}

        # Still None?
        if image is None or image.size == 0:
            return {"error": "Unable to read image file — possibly corrupted or unsupported format."}

        h, w, _ = image.shape
        boxes = reader.readtext(image, paragraph=False, width_ths=0.01)

        # Step 1: Collect per-word boxes and text
        results = []
        for bbox, text, conf in boxes:
            if conf > 0.7 and text.strip():
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                x = min(x_coords)
                y = min(y_coords)
                bw = max(x_coords) - x
                bh = max(y_coords) - y

                results.append({
                    "text": text.strip(),
                    "x": x,
                    "avg_y": (y + max(y_coords)) // 2,
                    "bbox": [x / w, y / h, bw / w, bh / h]
                })

        # Step 2: Srot and clear
        lines = sort_words_into_lines(results, y_tolerance=15)
        ordered_words = [w for line in lines for w in line]  # flatten

        # Step 3: Build string and mapping of char positions
        text_str = ""
        mapping = []
        offset = 0
        for w in ordered_words:
            cleaned = regex.sub(r'[^\p{L}\s]', '', w["text"])
            cleaned = regex.sub(r'\s+', ' ', cleaned).strip()

            if cleaned:
                mapping.append({
                    "bbox": w["bbox"],
                    "start": offset,
                    "end": offset + len(cleaned)
                })
                text_str += cleaned + " "
                offset += len(cleaned) + 1

        # Step 4: Run spaCy
        doc = nlp(text_str)
        tokens_without_punct = [token for token in doc if not token.is_punct and not is_stopword(token.text)]

        # Step 5: Merge by spaCy sentence
        merged_data = []

        for token in tokens_without_punct:
            token_start = token.idx
            token_end = token.idx + len(token.text)

            token_boxes = [
                m["bbox"] for m in mapping
                if m["start"] >= token_start and m["end"] <= token_end
            ]

            if token_boxes:
                merged_bbox = merge_bounding_boxes(token_boxes)
                merged_data.append({
                    "bbox": merged_bbox,
                    "text": token.text.strip()
                })

        #Result json
        renderTime += 1
        OCRjson[str(renderTime)] = merged_data
    
    #Delete data folder
    shutil.rmtree("data/imgs")
    os.remove("data/lesson.pdf")

    # Step 6: Save result for this frame
    print("OCR done!")
    if OCRjson:
        return OCRjson
    return None