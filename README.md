# Promenade

AI-агент для поиска культурного досуга — музеи, выставки, концерты, фестивали и другие мероприятия отдыха. Парсит страницы заведений, складывает в SQLite и векторную базу Qdrant, отвечает на запросы через LLM-пайплайн с retrieval и reranking.

## Структура

```
.
├── pyproject.toml              # метаданные пакета promenade
├── README.md
├── .env.example                # шаблон переменных окружения
├── .gitignore
│
├── src/
│   └── promenade/
│       ├── __init__.py
│       ├── models.py           # ядро: LLM-клиенты, embeddings, reranker, retrieval, агент
│       └── configure_db.ipynb  # SQLAlchemy-схема (Museum, Schedule) — импортируется в models.py
│
├── notebooks/                  # прикладные и демо-ноутбуки
│   ├── configure_vec_db.ipynb  # инициализация Qdrant-коллекции
│   ├── reader_page_parser.ipynb
│   ├── reader_tool.ipynb
│   ├── retriever.ipynb
│   └── example.ipynb
│
├── data/                       # локальное состояние (gitignored): SQLite, Qdrant, дампы
│   └── .gitkeep
│
├── docs/
│   └── samples/                # примеры распарсенных страниц
│       ├── example_page.md
│       └── example_page_2.md
│
└── tests/                      # задел под тесты
    └── .gitkeep
```

### Чем занято `data/`

Папка в репозиторий не коммитится (кроме `.gitkeep`). После прогона ноутбуков там появится:

- `data/museum.db` — SQLite с таблицами `museum` и `schedule`.
- `data/qdrant/` — векторная база Qdrant с коллекцией `museum_collection`.

## Требования

- Python 3.13+
- Доступ к API: OpenAI-совместимый endpoint (cloud.ru foundation models), Telegram bot token.

## Установка

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

pip install -e .
```

`pip install -e .` ставит пакет `promenade` в editable-режиме — ноутбуки и скрипты смогут импортировать его как `from promenade.models import *` независимо от CWD.

## Конфигурация

Скопируй `.env.example` в `.env` и заполни ключи:

```
OPENAI_API_KEY=...
TELEGRAM_TOKEN=...
QWEN_API_KEY=...
```

## Инициализация баз данных

Все команды и ноутбуки запускать **из корня репозитория** (а не из `notebooks/` или `src/promenade/`) — пути к `data/` резолвятся относительно CWD.

1. Запустить `src/promenade/configure_db.ipynb` — создаст `data/museum.db` и схему.
2. Запустить `notebooks/configure_vec_db.ipynb` — создаст `data/qdrant/` и коллекцию.

После этого можно пользоваться `notebooks/reader_page_parser.ipynb`, `reader_tool.ipynb`, `retriever.ipynb` и `example.ipynb`.
