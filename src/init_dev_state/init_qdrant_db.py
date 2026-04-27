
import shutil
from pathlib import Path
from promenade.models import DATA_DIR

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct

import uuid

QDRANT_PATH = DATA_DIR / "qdrant"

if QDRANT_PATH.exists():
    shutil.rmtree(QDRANT_PATH)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

client = QdrantClient(path=str(QDRANT_PATH))


if not client.collection_exists("museum_collection"):
   client.create_collection(
      collection_name="museum_collection",
      vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
   )

client.close()



