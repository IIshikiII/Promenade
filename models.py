# --- Standard library ---
import copy
import inspect
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import time
from typing import Any
from itertools import batched

# --- Third-party ---
import httpx
import requests
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, HttpUrl, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as atqdm

# --- LangChain ---
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- Qdrant ---
from qdrant_client import QdrantClient
from qdrant_client.http.models import Payload
from qdrant_client.models import Distance, PointStruct, VectorParams

# --- Local ---
from ipynb.fs.full.configure_db import Museum, Schedule

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")

MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LLM_BASE_URL = "https://foundation-models.api.cloud.ru/v1"

EMBEDDINGS_MODEL_NAME = "BAAI/bge-m3"
EMBEDDINGS_BASE_URL = "https://foundation-models.api.cloud.ru/v1"

RERANKER_BASE_URL = "https://foundation-models.api.cloud.ru"
# RERANKER_BASE_URL = "http://localhost:8000"
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERABKER_CONTEXT_LENGTH = 80000 # 2 char per tokern for 40k token limit

VECTOR_DATABASE_ADRESS = "./sqlite_museum_db/vector_base"
VECOTOR_DATABASE_COLLECTION = "museum_collection"

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=QWEN_API_KEY,
    base_url=LLM_BASE_URL,
)

embeddings_model = OpenAIEmbeddings(
    model=EMBEDDINGS_MODEL_NAME,
    api_key=QWEN_API_KEY,
    base_url=EMBEDDINGS_BASE_URL,
)

async_reranker_client = AsyncOpenAI(
    api_key=QWEN_API_KEY,
    base_url=RERANKER_BASE_URL
)

# --- Exceptions ---
class ValidationError(Exception):
    pass


# --- Agent models ---
@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: Any = None


class ToolTracer:
    """Collects all tool calls."""
    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def record(self, name: str, args: dict, result: Any = None) -> None:
        self.calls.append(ToolCallRecord(name=name, args=args, result=result))

    def called(self, name: str) -> bool:
        return any(c.name == name for c in self.calls)

    def get_calls(self, name: str) -> list:
        return [c for c in self.calls if c.name == name]

    def print_trace(self) -> None:
        print("=== Tool Call Trace ===")
        for i, c in enumerate(self.calls, 1):
            print(f"  {i}. {c.name}({json.dumps(c.args, ensure_ascii=False)[:80]})")
            if c.result is not None:
                print(f"     -> {json.dumps(c.result, ensure_ascii=False)[:100]}")
        print("=====================")


def llm_chat(messages: list, tools: list | None =  None):
    """Sends the message historu to LLM and returns the model response.

    Parameters:
      messages — list of dialog messages. Each message is a LangChain object:
                   SystemMessage(content="...")   — instruction for the model (agent role)
                   HumanMessage(content="...")    — message from the user
                   AIMessage(...)                 — previous model response
                   ToolMessage(content="...", tool_call_id="...") — tool result

      tools   — list of tool descriptions (OpenAI function calling schema or LangChain tools).

    Returns AIMessage:
      msg.content    — text response (str)
      msg.tool_calls — list of tool calls:
                         "name" — tool name
                         "args" — arguments (already parsed dict)
                         "id"   — unique call identifier
    """
    if tools:
        return llm.bind_tools(tools).invoke(messages)
    return llm.invoke(messages)


# --- Retriever models ---
class RetreiveReranker:
    def __init__(
            self, 
            retrieve_n: int = 5, 
            rerank_n: int = 1, 
            rerank_model: str= RERANKER_MODEL,
            rernk_batch_size: int=5
    ):
        if rerank_n > retrieve_n:
            raise ValidationError("rerank_n shoud be less than retrieve_n")
        self.retrieve_n = retrieve_n
        self.rerank_n = rerank_n
        self.rerank_model = rerank_model
        self.rernk_batch_size = rernk_batch_size

    def retrieve(self, query: str) -> list[Payload]:
        client = QdrantClient(path=VECTOR_DATABASE_ADRESS)
        try:
            embedded_query = embeddings_model.embed_query(query)
            hits = client.query_points(
                collection_name=VECOTOR_DATABASE_COLLECTION,
                query=embedded_query,
                limit=self.retrieve_n
            )
            res = [point.payload for point in hits.points]
            client.close()
            return res
        finally:
            client.close()

    async def rerank(self, hits: list[str], query: str, batch_size=10) -> list[dict]:
        batches = list(batched(hits, n=batch_size))
        async def _rerank_batch(i: int, batch: tuple) -> list[dict]:
            response = await async_reranker_client.post(
                path="/score",
                cast_to=httpx.Response,
                body={
                    "model": self.rerank_model,
                    "encoding_format": "float",
                    "text_1": query,
                    "text_2": list(batch),
                }
            )
            data = response.json()["data"]
            for item in data:
                item["index"] += i * batch_size
            return data
        
        results = await atqdm.gather(
            *[_rerank_batch(i, batch) for i, batch in enumerate(batches)]
        )
        return [item for batch_result in results for item in batch_result]
    
    
    def rechank_docs(self, docs: list[str]) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=RERABKER_CONTEXT_LENGTH)
        return [chunk for doc in docs for chunk in splitter.split_text(doc)]



    def retrieve_and_rerank(self, query: str):
        retrieved_docs = self.retrieve(query)
        documents = [h["text"] for h in retrieved_docs]
        reranked_points = self.rerank(documents, query)
        for r in reranked_points:
            retrieved_docs[r["index"]]["score"] = r["score"]

        sorted_and_reranked = sorted(retrieved_docs, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_and_reranked[: self.rerank_n]


# --- Reader models ---
READER_URL = HttpUrl(url="http://localhost:3000")
BINARY_FILES = ['.jfif', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.pdf', '.docx', '.xlsx', '.png', '.jpg', '.zip', '.mp4']
BINARY_FILES += [ext.upper() for ext in BINARY_FILES]
URL_PATTERN = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'

class DaySchedule(BaseModel):
    open_time: time
    close_time: time
    last_entry_time: time | None = None
    is_closed: bool = False

class WebTools:
    """Broser logic - parse web pages and extract data from them"""

    def __init__(self, reader_url: HttpUrl):
        self.reader_url = reader_url

    async def parse_page_reader(self, session, page_url: HttpUrl) -> tuple[str, str]:
        request_url = f"{self.reader_url}{page_url}"
        async with session.get(request_url) as resp:
            resp = await (resp.text())
            return (str(page_url), resp)

    def insert_schedule_into_db(
            self,
            place_name: str,
            place_url: HttpUrl,
            monday: DaySchedule | dict,
            tuesday: DaySchedule | dict,
            wednesday: DaySchedule | dict,
            thursday: DaySchedule | dict,
            friday: DaySchedule | dict,
            saturday: DaySchedule | dict,
            sunday: DaySchedule | dict,
    ) -> tuple[bool, str, str | None]:
        engine = create_engine(DATABASE_ADRESS)
        try:
            with Session(engine) as session:
                museum = Museum(museum_name=place_name, url=str(place_url))
                session.add(museum)
                session.flush()

                raw_days = [monday, tuesday, wednesday,
                            thursday, friday, saturday, sunday]
                days = [DaySchedule.model_validate(d) if isinstance(
                    d, dict) else d for d in raw_days]

                session.add_all([
                    Schedule(
                        museum_id=museum.id,
                        day_of_week=day_index,
                        open_time=day.open_time,
                        close_time=day.close_time,
                        last_entry_time=day.last_entry_time,
                        is_closed=day.is_closed,
                    )
                    for day_index, day in enumerate(days)
                ])
                museum_id = museum.id
                session.commit()
            return (True, museum_id, None)
        except Exception as e:
            print(e)
            return (False, "-1", f"Error: {e}")

    def insert_info_to_vector_db(
            self,
            id: str,
            place_name: str,
            place_info: str
    ) -> tuple[bool, str | None]:
        try:
            text_to_embed = f"{place_name}. {place_info}"
            print(text_to_embed)
            embeddings = embeddings_model.embed_query(text_to_embed)
            client = QdrantClient(path=VECTOR_DATABASE_ADRESS)
            client.upsert(
                collection_name="museum_collection",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings,
                        payload={
                            "id": id,
                            "text": text_to_embed
                        }
                    )]
            )
            client.close()
            return (True, None)
        except Exception as e:
            print(e)
            return (False, f"Error: {e}")

    def extract_and_validate_urls(self, texts: list[str], expression: str, domain: str) -> set[str]:
        links = []
        for text in texts:
            links.extend(re.findall(expression, text))
        urls = set([f"{scheme}://{host}{path}" for scheme, host, path in links if host == domain])
        bin_files = BINARY_FILES
        pages = {url for url in urls if not url.split('?')[0].endswith(tuple(bin_files))}
        # for page in pages:
        #     print(page)
        return pages

    async def _fetch_with_semaphore(self, session, semaphore, url):
        async with semaphore:
            await asyncio.sleep(0.5)
            return await WEB_TOOLS.parse_page_reader(session, url)
    
    async def fetch_all_museum_data(self, url: HttpUrl, depth: int = 1, filter_languages: bool = True) -> list[dict]:
        expression = URL_PATTERN
        domain = re.match(expression, url)[2]
        semaphore = asyncio.Semaphore(5)

        pages_dict = dict()
        async with aiohttp.ClientSession() as session:
            async with semaphore:
                result = await WEB_TOOLS.parse_page_reader(session, url)
                pages_dict.update([result])

        pages = self.extract_and_validate_urls([pages_dict[f"{url}"]], expression, domain)
        print(f"Найдено страниц: {len(pages)}")

        async with aiohttp.ClientSession() as session:
            results = await atqdm.gather(
                *[self._fetch_with_semaphore(session, semaphore, p) for p in pages],
                desc="Парсинг"
            )
        pages_dict.update(results)
        return pages_dict
    
    async def rerank_and_filter(
        self,
        documents: list[str], 
        query: str, 
        reranker: RetreiveReranker, 
        threshold: float = 0.1, 
        batch_size: int=5        
    ) -> list[dict]:
        documetns_copy = copy.deepcopy(documents)
        reranked_points = await reranker.rerank(documetns_copy, query=query, batch_size=batch_size)
        for r in reranked_points:
            documetns_copy[r["index"]] = {
                "score": r["score"],
                "text" : documetns_copy[r["index"]]
            }
            
        return [doc['text'] for doc in documetns_copy if doc['score'] >= threshold]
    


