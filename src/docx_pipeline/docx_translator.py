from src.inference import translate_many


def translate_docx(paragraphs):
    return translate_many(list(paragraphs))
