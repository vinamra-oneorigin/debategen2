import os
import shutil
from loguru import logger
from PyPDF2 import PdfReader
import pdfplumber

from controller.config import PDF_UPLOAD_DIR

def upload_pdf(src_path):
    """
    Validates and copies a PDF file to the upload directory.
    Returns the destination path or raises an error.
    """
    if not os.path.isfile(src_path):
        logger.error(f"File not found: {src_path}")
        raise FileNotFoundError(f"File not found: {src_path}")
    if not src_path.lower().endswith(".pdf"):
        logger.error("Only PDF files are supported.")
        raise ValueError("Only PDF files are supported.")
    dest_path = os.path.join(PDF_UPLOAD_DIR, os.path.basename(src_path))
    try:
        shutil.copy2(src_path, dest_path)
        logger.info(f"PDF uploaded to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to upload PDF: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2, falls back to pdfplumber if needed.
    Returns the extracted text or raises an error.
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        logger.error("Not a PDF file.")
        raise ValueError("Not a PDF file.")

    # Try PyPDF2 first
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            logger.info("Text extracted using PyPDF2.")
            return text
        else:
            logger.warning("PyPDF2 returned empty text, trying pdfplumber.")
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}, trying pdfplumber.")

    # Fallback to pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            logger.info("Text extracted using pdfplumber.")
            return text
        else:
            logger.error("Both PyPDF2 and pdfplumber returned empty text.")
            raise ValueError("Failed to extract text from PDF.")
    except Exception as e:
        logger.error(f"pdfplumber failed: {e}")
        raise ValueError("Failed to extract text from PDF with both libraries.")

def clean_extracted_text(text):
    """
    Cleans extracted text: removes extra whitespace, fixes encoding issues.
    """
    if not isinstance(text, str):
        logger.error("Input to clean_extracted_text must be a string.")
        raise ValueError("Input must be a string.")
    # Remove extra whitespace and normalize line endings
    cleaned = "\n".join(line.strip() for line in text.splitlines())
    cleaned = "\n".join(line for line in cleaned.splitlines() if line)  # Remove empty lines
    cleaned = cleaned.replace("\u00a0", " ")  # Replace non-breaking spaces
    cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return cleaned

# Alias for notebook compatibility
def extract_pdf_text(pdf_path):
    """
    Alias for extract_text_from_pdf for compatibility with debate.ipynb.
    """
    return extract_text_from_pdf(pdf_path)
