import easyocr
import cv2
import numpy as np
import regex
import os
import urllib.request
import fitz  # PyMuPDF
import shutil
import string
import nltk
from nltk.corpus import stopwords
import langid

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- Global Configuration ---
DOWNLOAD_DIR = "data"
IMG_DIR = os.path.join(DOWNLOAD_DIR, "imgs")
STOPWORDS_FILE = "vietnamese-stopwords.txt"

# --- 1. Robust Import Strategy (Fixes Pydantic/Spacy Crash) ---
USE_SPACY = False
NLP_MODELS = {"en": None, "vi": None}

try:
    # Try importing Spacy. If it crashes due to Pydantic ConfigError, we catch it.
    from spacy.lang.vi import Vietnamese
    from spacy.lang.en import English
    USE_SPACY = True
    print("✅ SpaCy loaded successfully.")
except Exception as e:
    print(f"⚠️ SpaCy import failed ({e}). Switching to NLTK fallback mode.")
    USE_SPACY = False

# --- 2. Fallback Tokenizer (Mimics Spacy API) ---
class MockToken:
    """Mock token object to replace Spacy token if Spacy fails."""
    def __init__(self, text, idx):
        self.text = text
        self.idx = idx
        self.is_punct = text in string.punctuation

def fallback_tokenize(text):
    """
    Uses NLTK/Regex to tokenize text and calculate character indices 
    to match Spacy's API (token.text, token.idx).
    """
    # Simple regex tokenizer that keeps punctuation as separate tokens
    # This pattern matches words OR non-whitespace punctuation
    pattern = r'\w+|[^\w\s]' 
    tokens = []
    
    # We need to find the exact start index of each token in the original string
    for match in regex.finditer(pattern, text):
        tokens.append(MockToken(match.group(), match.start()))
        
    return tokens

# --- 3. Resource Loading ---

# Initialize OCR Reader (Dual language to save reload time)
# set gpu=True if you have a GPU
GLOBAL_READER = easyocr.Reader(['vi', 'en'], gpu=True) 

def load_vietnamese_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

EN_STOPWORDS = set(stopwords.words('english'))
VI_STOPWORDS = load_vietnamese_stopwords(STOPWORDS_FILE)

# Optional: Restrict langid to only consider English and Vietnamese 
# This IMPROVES accuracy significantly because it won't guess 'fr' or 'de'.
langid.set_languages(['en', 'vi'])

def detect_language(text):
    if not text or not text.strip():
        return "en"
        
    try:
        # classify returns a tuple: ('vi', -54.4)
        lang, score = langid.classify(text)
        
        if lang == 'vi':
            return 'vi'
        return 'en'
        
    except Exception:
        return "en"
    
def get_nlp_processor(lang):
    """Returns a processing function/object based on availability."""
    if not USE_SPACY:
        return fallback_tokenize

    # Lazy load Spacy models if allowed
    if NLP_MODELS[lang] is None:
        if lang == "vi":
            nlp = Vietnamese()
        else:
            nlp = English()
        nlp.add_pipe('sentencizer')
        NLP_MODELS[lang] = nlp
    
    return NLP_MODELS[lang]

# --- 4. Helper Functions ---

def get_imgs_from_pdf(url):
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
    
    os.makedirs(IMG_DIR, exist_ok=True)
    pdf_path = os.path.join(DOWNLOAD_DIR, "lesson.pdf")

    try:
        urllib.request.urlretrieve(url, pdf_path)
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return []

    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.join(IMG_DIR, f"page_{i+1}.jpg")
            pix.save(img_path)
            image_paths.append(img_path)
        doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
        
    return image_paths

def sort_words_into_lines(results, y_tolerance=15):
    lines = []
    for word in results:
        placed = False
        for line in lines:
            mean_y = sum(w["avg_y"] for w in line) / len(line)
            if abs(word["avg_y"] - mean_y) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    lines.sort(key=lambda line: sum(w["avg_y"] for w in line) / len(line))
    for line in lines:
        line.sort(key=lambda w: w["x"])
    return lines

def merge_bounding_boxes(boxes):
    if not boxes:
        return [0, 0, 0, 0]
    x_mins = [b[0] for b in boxes]
    y_mins = [b[1] for b in boxes]
    x_maxs = [b[0] + b[2] for b in boxes]
    y_maxs = [b[1] + b[3] for b in boxes]
    
    min_x = min(x_mins)
    min_y = min(y_mins)
    return [min_x, min_y, max(x_maxs) - min_x, max(y_maxs) - min_y]

def is_stopword(token_text):
    t = token_text.lower()
    return t in EN_STOPWORDS or t in VI_STOPWORDS

# --- 5. Main OCR Function ---

def ocr_image(pdf_url, audio_arr):
    # Prepare timing (cumulative sum)
    timing_arr = list(audio_arr)
    for i in range(1, len(timing_arr)):
        timing_arr[i] += timing_arr[i - 1]

    imgs_path = get_imgs_from_pdf(pdf_url)
    if not imgs_path:
        return {"error": "Failed to extract images"}

    OCRjson = {}

    for image_path, renderTime in zip(imgs_path, timing_arr):
        print(f"Processing: {image_path}")
        image = cv2.imread(image_path)
        
        # Fallback reading
        if image is None or image.size == 0:
            try:
                data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                pass
        
        if image is None: continue

        h, w, _ = image.shape
        
        # EasyOCR
        try:
            boxes = GLOBAL_READER.readtext(image, paragraph=False, width_ths=0.01)
        except Exception as e:
            print(f"OCR Error: {e}")
            continue

        # Normalization
        results = []
        for bbox, text, conf in boxes:
            if conf > 0.5 and text.strip():
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, y_min = min(x_coords), min(y_coords)
                box_w, box_h = max(x_coords) - x_min, max(y_coords) - y_min
                
                results.append({
                    "text": text.strip(),
                    "x": x_min,
                    "avg_y": (y_min + max(y_coords)) // 2,
                    "bbox": [x_min / w, y_min / h, box_w / w, box_h / h]
                })

        # Line Sorting
        lines = sort_words_into_lines(results, y_tolerance=15)
        ordered_words = [w for line in lines for w in line]

        # Text Reconstruction
        text_str = ""
        mapping = []
        offset = 0
        
        for w in ordered_words:
            cleaned = regex.sub(r'[^\p{L}\s0-9]', '', w["text"])
            cleaned = regex.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                mapping.append({"bbox": w["bbox"], "start": offset, "end": offset + len(cleaned)})
                text_str += cleaned + " "
                offset += len(cleaned) + 1

        # NLP Processing (With Fallback)
        detected_lang = detect_language(text_str)
        print(f"Language detected: {detected_lang}")
        nlp = get_nlp_processor(detected_lang)
        
        doc = nlp(text_str) # Returns either Spacy Doc or list of MockTokens
        
        # Filter tokens
        tokens_of_interest = [t for t in doc if not t.is_punct and not is_stopword(t.text)]

        # Merge Logic
        merged_data = []
        for token in tokens_of_interest:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            # Find overlapping OCR boxes
            token_boxes = [
                m["bbox"] for m in mapping
                if not (m["end"] <= token_start or m["start"] >= token_end)
            ]
            
            if token_boxes:
                merged_data.append({
                    "bbox": merge_bounding_boxes(token_boxes),
                    "text": token.text.strip()
                })

        OCRjson[str(renderTime + 1)] = merged_data

    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
        
    print("OCR done!")
    return OCRjson if OCRjson else None