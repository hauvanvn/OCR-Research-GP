import easyocr
import cv2
import json
import numpy as np
import regex
import os

import string
import nltk
import spacy
import pandas as pd
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

from spacy.lang.vi import Vietnamese

nlp = Vietnamese()
nlp.add_pipe('sentencizer')

reader = easyocr.Reader(['vi']) # this needs to run only once to load the model into memory

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

def ocr_image(image_path):
    reader = easyocr.Reader(['vi']) # this needs to run only once to load the model into memory

    # Check file exists
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

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

    # if cur_frame >= 840 and cur_frame <=880:
    #     print("###############################################")
    #     print(text_str)

    # Step 4: Run spaCy
    doc = nlp(text_str)
    tokens_without_punct = [token for token in doc if not token.is_punct]

    # Step 5: Merge by spaCy sentence
    merged_data = []

    for token in tokens_without_punct:
        # if cur_frame >= 840 and cur_frame <=880:
        #   print(token.text.strip())
        # start position of the token in text_str
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

    # Step 6: Save result for this frame
    print(f"OCR: {image_path} done!")
    if merged_data:
        return merged_data
    return None