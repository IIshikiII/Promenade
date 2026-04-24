# Museum Agent

Агент для работы с музейными данными: парсинг страниц, наполнение SQLite и векторной базы (Qdrant), retrieval и LLM-пайплайн поверх этого.

## Структура проекта

```
.
├── models.py                    # Основной модуль: модели, клиенты LLM/embeddings/reranker, пайплайны
├── configure_db.ipynb           # Инициализация SQLite-схемы (Museum, Schedule) — импортируется в models.py
├── configure_vec_db.ipynb       # Инициализация векторной базы (Qdrant / vector_base)
├── reader_page_parser.ipynb     # Парсер страниц
├── reader_tool.ipynb            # Reader-инструмент
├── retriever.ipynb              # Retrieval-пайплайн
├── example.ipynb                # Пример использования
├── docs_samples/                # Примеры страниц в markdown
│   ├── example_page.md
│   └── example_page_2.md
├── .env.example                 # Шаблон переменных окружения
├── .gitignore
└── README.md
```

Генерируется локально и в репозиторий не попадает:

- `env/` — виртуальное окружение
- `sqlite_museum_db/` — SQLite БД и векторная база Qdrant
- `res.txt` — локальные дампы результатов
- `__pycache__/`, `.ipynb_checkpoints/`

## Требования

- Python 3.13
- Доступ к API: OpenAI-совместимый endpoint (cloud.ru foundation models), Telegram bot token

## Установка

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

pip install -r requirements.txt   # при наличии файла зависимостей
```

## Конфигурация

Скопируй `.env.example` в `.env` и заполни ключи:

```
OPENAI_API_KEY=...
TELEGRAM_TOKEN=...
QWEN_API_KEY=...
```

## Инициализация баз данных

1. Запустить `configure_db.ipynb` — создаст SQLite-схему.
2. Запустить `configure_vec_db.ipynb` — создаст векторную базу по пути `./sqlite_museum_db/vector_base`.

После этого можно использовать `models.py` и остальные ноутбуки.
