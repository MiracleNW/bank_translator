from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from docx import Document

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "dataset.csv"


# CLEAN TEXT
def clean_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,:;])", r"\1", text)

    return text.strip()


# Проверка смысла
def is_meaningful(text):
    cleaned = re.sub(r"[\d\W_]+", "", text)
    return len(cleaned) > 5

# Мусор
def is_garbage(text):
    text = text.strip()

    if re.fullmatch(r"[0-9]+[.)]?", text):
        return True

    if re.fullmatch(r"[A-Za-zА-Яа-я][.)]?", text):
        return True

    return False


# Парсинг одного DOCX
def parse_docx(file_path):
    doc = Document(file_path)
    data = []

    for table in doc.tables:
        for row in table.rows:
            cells = row.cells

            if len(cells) != 2:
                continue

            ru = clean_text(cells[0].text)
            en = clean_text(cells[1].text)

            if not ru or not en:
                continue

            if is_garbage(ru) or is_garbage(en):
                continue

            if not is_meaningful(ru) or not is_meaningful(en):
                continue

            data.append({
                "instruction": "Ты профессиональный переводчик банковских и юридических документов. Переведи текст с английского на русский, сохраняя официальный стиль и юридическую терминологию.",
                "input": en,
                "output": ru,
            })

    return data


# Обработка всей папки
def parse_all_docx(raw_dir):
    if not raw_dir.exists():
        raise FileNotFoundError(f"Папка с исходными DOCX не найдена: {raw_dir}")

    all_data = []
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".docx")]

    print(f"Найдено файлов: {len(files)}")
    if not files:
        raise FileNotFoundError(f"В папке {raw_dir} нет .docx файлов")

    for file in files:
        path = raw_dir / file
        print(f"Обрабатываю: {file}")
        data = parse_docx(path)
        all_data.extend(data)

    return pd.DataFrame(all_data)


# Сохранение
def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = parse_all_docx(RAW_DIR)

    print(f"\nСобрано строк: {len(df)}")
    if df.empty:
        raise ValueError("Не удалось собрать ни одной строки датасета. Проверьте таблицы DOCX: ожидается 2 колонки RU | EN.")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Сохранено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
