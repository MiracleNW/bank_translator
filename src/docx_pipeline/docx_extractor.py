from docx import Document
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "input_docx"


def extract_docx(file_path):

    doc = Document(file_path)

    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)

    return paragraphs


def load_all():

    files = list(INPUT_DIR.glob("*.docx"))

    print(f"FOUND DOCX: {len(files)}")

    data = {}

    for f in files:
        print(f"PROCESSING: {f.name}")
        data[f.name] = extract_docx(f)

    return data