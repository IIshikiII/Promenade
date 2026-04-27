# Promenade

An AI agent for discovering cultural leisure — museums, exhibitions, concerts, festivals and other events. Promenade crawls venue pages, persists structured data into SQLite and a Qdrant vector store, and answers natural-language queries through an LLM pipeline with retrieval and reranking.

## Repository layout

```
.
├── pyproject.toml                  # promenade package metadata
├── README.md
├── .env.example                    # environment variable template
├── .gitignore
│
├── src/
│   ├── promenade/                  # installable library
│   │   ├── __init__.py
│   │   └── models.py               # LLM / embeddings / reranker clients,
│   │                               # retrieval, web tools, ORM schema
│   │
│   └── init_dev_state/             # one-shot dev bootstrap scripts (not a package)
│       ├── init_sqlite_db.py       # creates data/museum.db and the schema
│       └── init_qdrant_db.py       # creates data/qdrant/ and the collection
│
├── notebooks/                      # demo and exploratory notebooks
│   ├── reader_page_parser.ipynb
│   ├── reader_tool.ipynb
│   ├── graph_agent.ipynb
│   └── example.ipynb
│
├── data/                           # local state (gitignored): SQLite, Qdrant
│   └── .gitkeep
│
├── docs/
│   └── samples/                    # examples of parsed pages
│       ├── cosmo_page.md
│       └── tretyakovka_page.md
│
└── tests/
    └── .gitkeep
```

### What lives in `data/`

The `data/` directory is gitignored (apart from `.gitkeep`). After running the bootstrap scripts it will contain:

- `data/museum.db` — SQLite database with the `museum` and `schedule` tables.
- `data/qdrant/` — local Qdrant vector store with the `museum_collection` collection.

## Requirements

- Python 3.13+
- API access: an OpenAI-compatible endpoint (cloud.ru foundation models) and a Telegram bot token.

## Installation

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

pip install -e .
```

`pip install -e .` installs the `promenade` package in editable mode, so notebooks and scripts can import it as `from promenade.models import ...` regardless of the current working directory. The `src/init_dev_state/` directory is intentionally excluded from the install — it contains development scripts, not library code.

## Configuration

Copy `.env.example` to `.env` and fill in the keys:

```
OPENAI_API_KEY=...
TELEGRAM_TOKEN=...
QWEN_API_KEY=...
```

## Bootstrapping the databases

Run all commands and notebooks **from the repository root** (not from `notebooks/` or `src/`) — `init_sqlite_db.py` resolves `data/` relative to the current working directory.

```bash
python src/init_dev_state/init_sqlite_db.py     # creates data/museum.db
python src/init_dev_state/init_qdrant_db.py     # creates data/qdrant/
```

Both scripts are idempotent: running them again wipes existing state and recreates an empty database.

## Usage

Once the databases are initialized, the demo notebooks can be opened in any order:

- `notebooks/reader_page_parser.ipynb` — fetches and parses venue pages.
- `notebooks/reader_tool.ipynb` — agent-facing reader tool wrapping the parser.
- `notebooks/graph_agent.ipynb` — LangGraph agent that orchestrates retrieval, reading and answering.
- `notebooks/example.ipynb` — end-to-end usage example.
