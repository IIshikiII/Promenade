"""Ready-to-use agent for parsing and processing information about museums."""
from promenade.models import *


class DaySchedule(BaseModel):
    open_time: time
    close_time: time
    last_entry_time: time | None = None
    is_closed: bool = False


class PlaceSchedule(BaseModel):
    place_name: str
    place_url: str
    monday: DaySchedule
    tuesday: DaySchedule
    wednesday: DaySchedule
    thursday: DaySchedule
    friday: DaySchedule
    saturday: DaySchedule 
    sunday: DaySchedule


class MainState(TypedDict):
    url: str
    subdocs: list[str]
    results: Annotated[list[dict], operator.add]


class DocState(TypedDict):
    doc: str
    schedule: PlaceSchedule | None
    museum_id: int | None
    saved_schedule: bool
    saved_vector: bool


def build_graph(WEB_TOOLS: WebTools, reranker: RetreiveReranker, llm: ChatOpenAI,
                DEV: bool = False, SAMPLES_PATH: Path | None = None) -> StateGraph:
    """
    Creates and compiles the agent graph for museum information processing.

    Args:
        WEB_TOOLS: WebTools instance for fetching and processing web content
        reranker: RetreiveReranker instance for document reranking
        llm: ChatOpenAI model for extraction and structuring
        DEV: Development mode (use samples instead of real parsing)
        SAMPLES_PATH: Path to samples for DEV mode

    Returns:
        Compiled StateGraph agent
    """
    # Initialize dependencies
    if DEV and SAMPLES_PATH is None:
        raise ValueError("SAMPLES_PATH is required in DEV mode")
    
    EXTRACTPR_SYSTEM = """
You are an AI assistant that extracts and structures visitor information from website content. Your output must be a clean, well-organized report in Russian, even if the input is messy or lacks JavaScript-generated content.

Follow these rules strictly:

1. **Identify all distinct physical locations** (museums, branches, exhibition halls) mentioned in the text.  
   - If there are multiple locations (e.g., main museum + house-museum), create a separate section for each.  
   - Separate sections with a line of exactly 4 equals signs: `====`  
   - Order sections by importance (main location first, then branches).

2. **For each location, extract the following categories** (if available):  
   - `Часы работы` (opening hours) – list days and times clearly.  
   - `Стоимость билетов` (ticket prices) – any numbers, or mention if free, or "не указано".  
   - `Актуальные выставки и события` (current exhibitions & events) – include dates and short descriptions.  
   - `Специальные предложения` (special offers) – e.g., free admission days, discounts, package tickets.  
   - `Адрес и контакты` (address & contacts) – any physical address, phone, email, website.  
   - `Дополнительная информация` (additional info) – anything else useful: accessibility, educational programs, virtual tours, etc.

3. **If a category has no information** in the provided text, write `— информация отсутствует —` (do not invent data).

4. **Preserve concrete facts** (dates, times, prices, names).  
   - Do not paraphrase numbers or dates incorrectly.  
   - If a date is relative (“ближайшее бесплатное посещение — 16 апреля”), keep it as is.

5. **Be thorough** – scan the entire text for any minor detail that could affect a visitor’s decision (e.g., “музей сегодня работает до 21:00”, “требуется предварительный билет”, “Пушкинская карта”).

6. **Output structure** (example for one location):

   ## [Название места]
   **Часы работы**  
   - понедельник: 10:00–18:00  
   - ...

   **Стоимость билетов**  
   - ...

   **Актуальные выставки и события**  
   - [Название] (даты): описание...

   **Специальные предложения**  
   - ...

   **Адрес и контакты**  
   - ...

   **Дополнительная информация**  
   - ...

   ==== (if another location follows)

7. **Language:** The entire report must be in Russian, except for the separators (`====`). Use proper Russian punctuation and formatting.

8. **If the input is very short or contains only “У вас отключен JavaScript”** – still extract any hours, addresses, or links that are visible. Do not say “no information” if something is present.

Now produce the report based on the user’s input.
"""

    def dev_parse_page_info(page_url: str) -> list[str]:
        """DEV mode: load from sample files."""
        if SAMPLES_PATH is None:
            raise ValueError("SAMPLES_PATH not specified")
        
        if page_url == "https://kosmo-museum.ru/":
            with open(SAMPLES_PATH / "cosmo_page.md", "r", encoding="utf8") as f:
                base_url_str = f"Base url: {page_url} \n\n"
                docs = [base_url_str + doc.strip() for doc in f.read().split("====")]
                return docs
        if page_url == "https://www.tretyakovgallery.ru/":
            with open(SAMPLES_PATH / "tretyakovka_page.md", "r", encoding="utf8") as f:
                base_url_str = f"Base url: {page_url} \n\n"
                docs = [base_url_str + doc.strip() for doc in f.read().split("====")]
                return docs
        return []
    
    async def parse_page_info(page_url: str) -> list[str]:
        """Convert webpage content to markdown format."""
        if DEV:
            return dev_parse_page_info(page_url)
        else:
            base_url = re.match(URL_PATTERN, page_url)[0]
            res = await WEB_TOOLS.fetch_all_museum_data(base_url)
            text_values = list(res.values())
            rechanked_values = reranker.rechank_docs(text_values)
            reranked_values =  await WEB_TOOLS.rerank_and_filter(rechanked_values, query=EXTRACTPR_SYSTEM, reranker = reranker)
            bin_masked_doc = WEB_TOOLS.mask_binary_urls(''.join(reranked_values))
            base_url_masked_doc = WEB_TOOLS.mask_binary_urls(bin_masked_doc)
            base_url_str = f"Base url: {base_url} \n\n"
            full_doc = base_url_str + base_url_masked_doc

            system_msg = SystemMessage(content=(EXTRACTPR_SYSTEM))
            messages = [system_msg, HumanMessage(content=full_doc)]

            response = llm_chat(messages=messages).content

            docs = [base_url_str + doc.strip() for doc in response.split("====")]
            return docs
    
    # === Subgraph for processing a single document ===
    structured_llm = llm.with_structured_output(PlaceSchedule)

    SHCEDULE_EXTRACTOR_SYSTEM = """You are a structured data extractor. Your only job is to fill the PlaceSchedule schema from a single location section.

## Input format
- The very first line starts with "Base url:" — that is the value for place_url.
- The section heading (## Name) is the value for place_name — use the Russian name as written.

## Filling each DaySchedule field
Think through all 7 days explicitly before producing output.

open_time / close_time:
- If a range covers multiple days ("Пн–Пт: 10:00–18:00"), apply those times to each day individually.
- If a day is not mentioned at all, apply the general/default schedule from the section.

last_entry_time:
- Fill only if explicitly stated ("вход до 20:00", "касса до 17:00", "last entry at ...").
- Leave null if not mentioned.

is_closed:
- Set True only if the text explicitly says the place is closed on that day ("выходной", "closed", "не работает").
- If is_closed is True, still set open_time and close_time to 00:00.

## Rules
- Never invent or guess times — only use what is written.
"""
    
    def llm_extract(state: DocState):
        schedule = structured_llm.invoke(
            [
                SystemMessage(SHCEDULE_EXTRACTOR_SYSTEM), 
                HumanMessage(state["doc"])
            ]
        )
        return {"schedule": schedule}

    def save_schedule(state: DocState):
        ok, museum_id, err = WEB_TOOLS.insert_schedule_into_db(**state["schedule"].model_dump())
        return {
            "saved_schedule": ok,
            "museum_id": museum_id
        }

    def save_vector(state: DocState):
        ok = WEB_TOOLS.insert_info_to_vector_db(
            id = state["museum_id"],
            place_name = state["schedule"].place_name,
            place_info = state["doc"]
        )
        return {"saved_vector": ok}

    def doc_result(state: DocState) -> dict:
        return {"results": [{
            "ok": state["saved_schedule"] and state["saved_vector"],
            "doc_preview": state["doc"][:50],
            "place_name": state["schedule"].place_name
        }]}
    
    doc_builder = StateGraph(DocState, output_schema=MainState)
    doc_builder.add_node("llm_extract", llm_extract)
    doc_builder.add_node("save_schedule", save_schedule)
    doc_builder.add_node("save_vector", save_vector)
    doc_builder.add_node("doc_result", doc_result)

    doc_builder.add_edge(START, "llm_extract")
    doc_builder.add_edge("llm_extract", "save_schedule")
    doc_builder.add_edge("save_schedule", "save_vector")
    doc_builder.add_edge("save_vector", "doc_result")
    doc_builder.add_edge("doc_result", END)

    doc_graph = doc_builder.compile()
    
    # === Main graph ===
    async def parse_node(state: MainState) -> dict:
        """Parse page and extract sub-documents."""
        subdocs = await parse_page_info(page_url = state["url"])
        return {"subdocs": subdocs}

    def dispatch(state: MainState) -> list[Send]:
        """Create dispatch tasks for each subdocument."""
        return [
            Send("process_doc", {
                "doc": doc, 
                "schedule": None,
                "saved_schedule": False, 
                "saved_vector": False,
                "museum_id": None
            })
            for doc in state["subdocs"]
        ]

    def aggregate_node(state: MainState) -> dict:
        """Aggregate results and print summary."""
        success = sum(1 for r in state["results"] if r["ok"])
        print(f"Processed: {success}/{len(state['results'])}")
        return {}

    main_builder = StateGraph(MainState)
    main_builder.add_node("parse_node", parse_node)
    main_builder.add_node("process_doc", doc_graph)       # подграф как нода
    main_builder.add_node("aggregate_node", aggregate_node)

    main_builder.add_edge(START, "parse_node")
    main_builder.add_conditional_edges("parse_node", dispatch, ["process_doc"])
    main_builder.add_edge("process_doc", "aggregate_node")
    main_builder.add_edge("aggregate_node", END)

    agent = main_builder.compile()

    return main_builder.compile()
