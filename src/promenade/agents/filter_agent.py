"""Ready-to-use agent for selecting information about places scheduel from SQL database."""
from promenade.models import *
import calendar
QDRANT_PATH = DATA_DIR / "qdrant"


TimeString = Annotated[str, "format: HH:MM:SS.ffffff"]

def validate_time_format(v: str) -> str:
    """Validate and normalize time format to HH:MM:SS.ffffff"""
    import re
    time = v
    
    # If just hour like "10" -> "10:00:00.000000"
    if re.match(r"^\d{1,2}$", v):
        time = f"{int(v):02d}:00:00.000000"
    
    # If HH:MM like "10:00" -> "10:00:00.000000"
    if re.match(r"^\d{1,2}:\d{2}$", v):
        parts = v.split(":")
        time = f"{int(parts[0]):02d}:{parts[1]}:00.000000"
    
    # Full format HH:MM:SS.ffffff - validate each part
    if re.match(r"^\d{1,2}:\d{2}:\d{2}\.\d+$", time):
        parts = v.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_part = parts[2].split(".")
        seconds = int(seconds_part[0])
        
        # Validate time ranges
        if hours > 23:
            raise ValueError(f"Invalid time: hours must be 0-23, got {hours}")
        if minutes > 59:
            raise ValueError(f"Invalid time: minutes must be 0-59, got {minutes}")
        if seconds > 59:
            raise ValueError(f"Invalid time: seconds must be 0-59, got {seconds}")
        
        return time
    
    # Invalid format - raise error
    raise ValueError(f"Invalid time format: {v}. Expected HH, HH:MM, or HH:MM:SS.ffffff")


class FilterState(TypedDict):
    input_query: str
    start_time: TimeString | None
    end_time: TimeString | None
    day_of_week: int | None
    filtred_places: list[tuple] | None
    res_status: str | None

class TimeSpace(BaseModel):
    start_time: str | None
    end_time: str | None
    day_of_week: int | None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def validate_time(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return validate_time_format(v)

    @field_validator("day_of_week", mode="before")
    @classmethod
    def validate_day(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 0 or v > 6:
            raise ValueError("day_of_week must be 0-6 (Monday-Sunday)")
        return v
    

def build_filtring_agent(
        llm,
) -> StateGraph:
    today = datetime.today().strftime('%Y-%m-%d')
    weekday_num = datetime.today().weekday()
    day_name_full = calendar.day_name[weekday_num]

    structured_llm = llm.with_structured_output(TimeSpace)
    QUERY_TRANSFORM_SYSTEM = f"""You are a structured data extractor. 
Your only job is to transform user query into TimeSpace schema.
today is {today}, {day_name_full}.

User will ask about when they can visit places. Extract the time window they are interested in.
The day_of_week should be extracted from phrases like "в следующий вторник", "в четверг", "в субботу" etc.
Days of week mapping: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6.

IMPORTANT: Time format MUST be HH:MM:SS.ffffff (e.g., 10:00:00.000000, 21:00:00.000000).
If user gives time as "10" or "10:00", convert it to "10:00:00.000000".
If user gives time as "21:00", convert it to "21:00:00.000000".

If the user provides insufficient information to define the time or the day of the week, set the corresponding field to None.

Return ONLY valid JSON matching TimeSpace schema with keys: start_time, end_time, day_of_week.
"""
    
    def tranform_user_query(state: FilterState):
        time_window = structured_llm.invoke(
            [
                SystemMessage(QUERY_TRANSFORM_SYSTEM), 
                HumanMessage(state["input_query"])
            ]
        )
        start_time = time_window.start_time
        end_time = time_window.end_time
        day_of_week =  time_window.day_of_week
        
        next_edge = "filter_database" if all([start_time, end_time, day_of_week]) else "aggregate_result"
        
        return {
            "start_time": start_time,
            "end_time":end_time,
            "day_of_week": day_of_week,
            "next": next_edge
        }
    
    def filter_database(state: FilterState):
        QUERY = f"""
SELECT 
    s.museum_id, m.museum_name, m.url
FROM schedule s
LEFT JOIN museum m
    on m.id = s.museum_id
WHERE 
    is_closed IS 0
    AND day_of_week IS {state["day_of_week"]}
    AND open_time <= "{state["start_time"]}"
    AND CASE
        WHEN last_entry_time IS NOT NULL 
        THEN last_entry_time >= "{state["end_time"]}"
        ELSE close_time >= "{state["end_time"]}"
    END

"""
        engine = create_engine(DATABASE_ADRESS)
        with engine.connect() as conn:
            res = conn.execute(text(QUERY))
            rows = res.fetchall()
        return {
            "filtred_places": rows
        }
    
    def aggregate_result(state: FilterState):
        start_time = state["start_time"]
        end_time = state["end_time"]
        day_of_week =  state["day_of_week"]
        if all([start_time, end_time, day_of_week]):
            places_count =  len(state["filtred_places"])
            if places_count == 0:
                return {
                    "res_status": "There are no open places in the database for this time."
                }
            else:
                return {
                    "res_status": f"During this time you can visit {places_count} places."
                }
        else:
            values = [
                (start_time, "start_time"), 
                (end_time, "end_time"), 
                (day_of_week, "day_of_week")
                ]
            
            missing_fieds = [elem[1] for elem in values if elem[0] is None]
            res_str = f"Missing following information:\n{", ".join(missing_fieds)}"
            return {
                "res_status": res_str
            }
        
    filter_builder = StateGraph(FilterState)
    filter_builder.add_node("tranform_user_query", tranform_user_query)
    filter_builder.add_node("filter_database", filter_database)
    filter_builder.add_node("aggregate_result", aggregate_result)


    filter_builder.add_edge(START, "tranform_user_query")
    filter_builder.add_conditional_edges(
        "tranform_user_query",
        lambda x: x["next"],
        ["filter_database", "aggregate_result"]
    )
    filter_builder.add_edge("filter_database", "aggregate_result")
    filter_builder.add_edge("aggregate_result", END)


    return filter_builder.compile()