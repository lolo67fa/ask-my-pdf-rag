"""PDF Processing Module for 'Ask My PDF'.

Handles:
- Extracting text page-by-page using PyMuPDF (fitz), preserving 1-based page numbers.
- Detecting empty, scanned, or corrupted PDFs gracefully.
- Splitting pages into overlapping text chunks while retaining page metadata.
"""

from typing import List, Dict, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF


def extract_pages_from_pdf(pdf_source: Union[bytes, str]) -> List[Dict[str, Union[int, str]]]:
    """Extract text from each page of a PDF document.

    Args:
        pdf_source: Raw PDF bytes or a string file path.

    Returns:
        A list of dictionaries, each containing:
            - page_number (int): 1-based page number.
            - text (str): Cleaned extracted text from the page.

    Raises:
        ValueError: If the file is not a valid PDF, has zero pages,
                    or contains no extractable text (e.g., scanned images).
    """
    doc = None
    try:
        if isinstance(pdf_source, bytes):
            if not pdf_source:
                raise ValueError("The uploaded file is empty (0 bytes).")
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        elif isinstance(pdf_source, str):
            doc = fitz.open(pdf_source)
        else:
            raise TypeError("pdf_source must be bytes or a file path string.")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF file: {exc}") from exc

    if len(doc) == 0:
        doc.close()
        raise ValueError("The PDF document contains 0 pages.")

    pages: List[Dict[str, Union[int, str]]] = []
    total_characters = 0

    for index in range(len(doc)):
        page = doc[index]
        page_number = index + 1  # 1-based page number
        raw_text = page.get_text("text") or ""
        cleaned_text = " ".join(raw_text.split())

        pages.append({
            "page_number": page_number,
            "text": cleaned_text
        })
        total_characters += len(cleaned_text)

    doc.close()

    # If the document has pages but virtually no selectable text, it's empty or scanned
    if total_characters < 15:
        raise ValueError(
            "This PDF appears to be empty or contains scanned images without selectable text. "
            "Please upload a document with readable text."
        )

    return pages


def chunk_pages(
    pages: List[Dict[str, Union[int, str]]],
    chunk_size: int = 150,
    chunk_overlap: int = 30
) -> List[Dict[str, Union[str, int]]]:
    """Split page text into manageable overlapping word-based chunks.

    Each chunk retains its source 1-based page number so that answers
    can always cite the correct page.

    Args:
        pages: List of page dicts from extract_pages_from_pdf.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts, each containing:
            - chunk_id: Unique string identifier (e.g. 'p1_c1').
            - page_number: 1-based integer page number.
            - text: Text excerpt string for the chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    chunks: List[Dict[str, Union[str, int]]] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        page_num = int(page["page_number"])
        text = str(page["text"]).strip()

        if not text:
            continue

        words = text.split()
        if len(words) <= chunk_size:
            # Single chunk for this page
            chunks.append({
                "chunk_id": f"p{page_num}_c1",
                "page_number": page_num,
                "text": text
            })
            continue

        # Overlapping sliding window
        chunk_idx = 1
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"p{page_num}_c{chunk_idx}",
                "page_number": page_num,
                "text": chunk_text
            })

            chunk_idx += 1
            if end >= len(words):
                break
            start += step

    return chunks


def compute_document_metrics(pages: List[Dict[str, Union[int, str]]]) -> Dict[str, int]:
    """Calculate total page count and word count across extracted pages.

    Args:
        pages: List of page dicts.

    Returns:
        Dictionary with 'page_count' and 'word_count'.
    """
    total_pages = len(pages)
    total_words = sum(len(str(p["text"]).split()) for p in pages)
    return {
        "page_count": total_pages,
        "word_count": total_words
    }
