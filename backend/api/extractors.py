import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import mimetypes
import os
import platform
from pathlib import Path
from PIL import Image

# ======================================================
# CONFIGURATION — Auto-detect Tesseract on Windows
# ======================================================

if platform.system() == "Windows":
    # Common install paths for Tesseract on Windows
    _tesseract_candidates = [
        r"M:\mahesh\Intenships\Infosys Internship\Tesseract-OCR\tesseract.exe",  # ✅ project-local install
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.environ.get("USERNAME", "")),
    ]
    _found = False
    for _path in _tesseract_candidates:
        if Path(_path).exists():
            pytesseract.pytesseract.tesseract_cmd = _path
            print(f"✅ Tesseract found at: {_path}")
            _found = True
            break
    if not _found:
        print("⚠️  Tesseract not found in common paths. Image OCR will fail.")
        print("    Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    # Check availability by exe existence only — avoids blocking subprocess call at startup
    TESSERACT_AVAILABLE = _found
else:
    # On Linux/Mac, tesseract should be in PATH
    TESSERACT_AVAILABLE = bool(os.environ.get("TESSERACT_AVAILABLE", "1") != "0")

# ======================================================
# FILE TYPE DETECTION
# ======================================================

def detect_file_type(file_path: Path) -> str:
    """
    Detect file type using extension + mimetype fallback.
    Returns: pdf | image | csv | unknown
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return "pdf"

    if ext in [".png", ".jpg", ".jpeg"]:
        return "image"

    if ext == ".csv":
        return "csv"

    # Fallback using mimetype
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        if "pdf" in mime:
            return "pdf"
        if "image" in mime:
            return "image"
        if "csv" in mime:
            return "csv"

    return "unknown"


# ======================================================
# PDF EXTRACTION
# ======================================================

def extract_pdf_text(file_path: Path) -> str:
    """
    Extract text from digital PDFs using PyMuPDF.
    Includes detailed logging for debugging.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            print(f"📄 PDF opened successfully: {file_path.name}")
            print(f"   Total pages: {len(doc)}")
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += page_text
                print(f"   Page {page_num + 1}: {len(page_text)} characters extracted")
            
            print(f"✅ Total text extracted: {len(text)} characters")
            
            # Log first 500 characters for debugging
            if text:
                print(f"\n📋 First 500 characters of extracted text:")
                print("   " + repr(text[:500]))
    except Exception as e:
        print(f"❌ PDF Extraction Error: {e}")

    return text.strip()


# ======================================================
# IMAGE OCR EXTRACTION
# ======================================================

def extract_image_text(file_path: Path) -> str:
    """
    OCR extraction from image files using Tesseract.
    Falls back with a clear error if Tesseract is not installed.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is not installed on this machine. "
            "Please install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "to process image files. For now, please upload a PDF or CSV instead."
        )
    try:
        image = Image.open(file_path)
        # Use lang='eng' for medical reports
        text = pytesseract.image_to_string(image, config="--psm 6")
        if not text.strip():
            raise ValueError("OCR returned empty text. The image may be too dark, blurry, or not a medical report.")
        return text.strip()
    except RuntimeError:
        raise  # Re-raise our custom error
    except Exception as e:
        print(f"❌ Image OCR Error: {e}")
        raise RuntimeError(f"Could not read text from image: {e}")


# ======================================================
# CSV EXTRACTION
# ======================================================

def extract_csv_text(file_path: Path) -> str:
    """
    Convert CSV rows into readable medical sentences for the AI.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            return ""

        lines = []
        for _, row in df.iterrows():
            sentence_parts = [f"{col} is {val}" for col, val in row.items() if pd.notna(val)]
            if sentence_parts:
                lines.append(", ".join(sentence_parts))

        return "\n".join(lines)

    except Exception as e:
        print(f"❌ CSV Extraction Error: {e}")
        return ""


# ======================================================
# PDF → IMAGE CONVERSION FOR OCR FALLBACK
# ======================================================

def convert_pdf_to_images(file_path: Path):
    """
    Convert PDF pages into high-resolution images for OCR fallback.
    Used when a PDF is a 'scanned' image rather than digital text.
    """
    images = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            # Increase resolution (zoom) for better OCR accuracy
            zoom = 2  
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
    except Exception as e:
        print(f"❌ PDF Image Conversion Error: {e}")

    return images


# ======================================================
# UNIFIED EXTRACTION ROUTER (The Main Function)
# ======================================================

def extract_text(file_path: Path) -> str:
    """
    Auto-detect file type and extract text accordingly.
    Includes OCR fallback for scanned PDFs.
    """
    if not file_path.exists():
        return ""

    file_type = detect_file_type(file_path)
    print(f"\n{'='*60}")
    print(f"📂 FILE EXTRACTION ENGINE")
    print(f"{'='*60}")
    print(f"🔍 File: {file_path.name}")
    print(f"🔍 Type: {file_type}")
    print(f"🔍 Size: {file_path.stat().st_size} bytes")

    if file_type == "pdf":
        text = extract_pdf_text(file_path)

        # If the text is empty, the PDF is likely a scanned image.
        if not text or len(text.strip()) < 20:
            print("⚠️ Digital text not found. Starting OCR Fallback...")
            if not TESSERACT_AVAILABLE:
                raise RuntimeError(
                    "This PDF appears to be a scanned image, but Tesseract OCR is not installed. "
                    "Please install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki, "
                    "or upload a digital (text-based) PDF instead."
                )
            try:
                images = convert_pdf_to_images(file_path)
                ocr_results = []
                for img in images:
                    result = pytesseract.image_to_string(img, config="--psm 6")
                    if result.strip():
                        ocr_results.append(result)
                combined = "\n".join(ocr_results).strip()
                if not combined:
                    raise ValueError("OCR fallback returned empty text from scanned PDF.")
                print(f"✅ OCR extraction successful: {len(combined)} characters")
                return combined
            except RuntimeError:
                raise
            except Exception as e:
                print(f"❌ OCR Fallback failed: {e}")
                raise RuntimeError(f"Failed to extract text from scanned PDF: {e}")
        print(f"✅ PDF extraction successful: {len(text)} characters")
        return text

    if file_type == "image":
        result = extract_image_text(file_path)
        print(f"✅ Image extraction successful: {len(result)} characters")
        return result

    if file_type == "csv":
        result = extract_csv_text(file_path)
        print(f"✅ CSV extraction successful: {len(result)} characters")
        return result

    print(f"❌ Unsupported file type detected: {file_type}")
    return ""