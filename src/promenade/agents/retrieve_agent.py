"""Ready-to-use agent for retrieving information from vector store about museums."""
from promenade.models import *


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
retrieve_reranker = RetreiveReranker(retrieve_n=15, rerank_n=15, rerank_model=RERANKER_MODEL)


class RetrieveState(TypedDict):
    input_query: str
    reranked_documents: list[dict] | None
    result: str | None
    

def build_reranker_graph(
        llm: ChatOpenAI, 
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_threshold: float = 0.6
    ) -> StateGraph:
    retrieve_reranker = RetreiveReranker(retrieve_n=15, rerank_n=15, rerank_model=rerank_model)

    async def retrieve_and_rerank_node(state: RetrieveState) -> dict:
        res = await retrieve_reranker.retrieve_and_rerank(state["input_query"])
        return {
            "reranked_documents": res
        }
    
    def if_null(state: RetrieveState):
        if any(doc.get("score", 0) >= rerank_threshold for doc in state["reranked_documents"]):
            return {
                "next": "selection_node"
            }
        else:
            return   {
                "next": END,
                "result": "There is no relevant places"
            }
        
    def selection_node(state: RetrieveState):
        # for doc in state["reranked_documents"]:
        #     print(doc)
        docs = [doc["text"] for doc in state["reranked_documents"] if doc["score"] >= rerank_threshold]
        return{
            "result": "\n\n====\n".join(docs)
        }
    
    retrieve_builder = StateGraph(RetrieveState)
    retrieve_builder.add_node("retrieve_and_rerank_node", retrieve_and_rerank_node)
    retrieve_builder.add_node("if_null", if_null)
    retrieve_builder.add_node("selection_node", selection_node)

    retrieve_builder.add_edge(START, "retrieve_and_rerank_node")
    retrieve_builder.add_edge("retrieve_and_rerank_node", "if_null")
    retrieve_builder.add_conditional_edges(
        "if_null",
        lambda x: x["next"],
        ["selection_node", END]
    )
    retrieve_builder.add_edge("selection_node", END)

    return retrieve_builder.compile()