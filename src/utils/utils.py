import yaml

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_example(example):

    return f"""### Instruction:
Ты профессиональный переводчик банковских и юридических документов.
Переведи текст с английского на русский, сохраняя официальный стиль и юридическую терминологию.

### Input:
{example['input']}

### Response:
{example['output']}
"""