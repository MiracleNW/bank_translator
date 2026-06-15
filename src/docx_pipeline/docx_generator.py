from docx import Document
from pathlib import Path

from src.docx_pipeline.docx_extractor import load_all
from src.docx_pipeline.docx_translator import translate_docx

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output_docx"

def save_docx(original_name, translated_texts):

    doc = Document()

    for text in translated_texts:
        doc.add_paragraph(text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / f"{Path(original_name).stem}_translated.docx"

    doc.save(out_path)

    print(f"SAVED: {out_path}")


def run():

    data = load_all()

    print(f"\nFOUND FILES: {len(data)}")

    for filename, paragraphs in data.items():

        print(f"\n==========================")
        print(f"FILE: {filename}")
        print(f"PARAGRAPHS: {len(paragraphs)}")

        translated = translate_docx(paragraphs)

        save_docx(filename, translated)


if __name__ == "__main__":
    run()