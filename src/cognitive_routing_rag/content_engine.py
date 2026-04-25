from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from cognitive_routing_rag.llm import GeneratedPost, TopicDecision, structured_invoke


NEWS_DB = {
    "ai": [
        "OpenAI unveils a new reasoning model aimed at coding-heavy workflows.",
        "Developers debate whether AI copilots are replacing entry-level engineering work.",
    ],
    "crypto": [
        "Bitcoin rallies as institutions increase exposure through regulated investment products.",
        "Stablecoin legislation returns to the spotlight as policymakers push new oversight proposals.",
    ],
    "finance": [
        "Traders reassess rate-cut odds after stronger-than-expected inflation data.",
        "Quant funds increase allocation to volatility-sensitive strategies.",
    ],
    "ev": [
        "Major automakers report improved EV battery durability in new long-range platforms.",
        "Analysts say charging infrastructure growth is reducing adoption friction.",
    ],
    "space": [
        "Commercial launch providers accelerate competition over reusable heavy-lift rockets.",
        "Satellite internet expansion intensifies debate around orbital regulation.",
    ],
}


@tool
def mock_searxng_search(query: str) -> list[str]:
    """Return hardcoded recent-looking headlines for a keyword query."""
    lowered = query.lower()
    if "crypto" in lowered or "bitcoin" in lowered:
        return NEWS_DB["crypto"]
    if "market" in lowered or "rate" in lowered or "finance" in lowered:
        return NEWS_DB["finance"]
    if (
        re.search(r"\bev\b", lowered)
        or "electric vehicle" in lowered
        or "battery" in lowered
    ):
        return NEWS_DB["ev"]
    if "space" in lowered or "rocket" in lowered or "musk" in lowered:
        return NEWS_DB["space"]
    return NEWS_DB["ai"]


class GraphState(TypedDict, total=False):
    bot_id: str
    persona: str
    topic: str
    search_query: str
    search_results: list[str]
    post: GeneratedPost


def _fallback_topic(persona: str) -> TopicDecision:
    lowered = persona.lower()
    if "crypto" in lowered or "space" in lowered:
        return TopicDecision(
            topic="AI replacing junior developers",
            search_query="AI coding model junior developers",
        )
    if "markets" in lowered or "roi" in lowered or "interest rates" in lowered:
        return TopicDecision(
            topic="Rate cuts and AI productivity trade",
            search_query="interest rates AI productivity market impact",
        )
    return TopicDecision(
        topic="Tech monopolies and AI labor displacement",
        search_query="AI labor displacement tech monopolies",
    )


def decide_search(state: GraphState) -> GraphState:
    persona = state["persona"]
    decision = structured_invoke(
        TopicDecision,
        prompt=(
            "You are picking a topic for a social-media bot.\n"
            f"Persona:\n{persona}\n\n"
            "Choose one timely topic the bot would care about today. "
            "Return a topic and a compact web search query."
        ),
        fallback=_fallback_topic(persona),
    )
    return {"topic": decision.topic, "search_query": decision.search_query}


def web_search(state: GraphState) -> GraphState:
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    return {"search_results": results}


def _fallback_post(bot_id: str, topic: str, persona: str, search_results: list[str]) -> GeneratedPost:
    headline = search_results[0]
    persona_lower = persona.lower()
    if "roi" in persona_lower or "markets" in persona_lower:
        content = (
            f"{headline} Translation: productivity is a margin story, not a morality play. "
            "If AI compresses junior labor costs, capital reprices first. Watch rates, multiples, and who monetizes inference."
        )
    elif "critical of ai" in persona_lower or "late-stage capitalism" in persona_lower:
        content = (
            f"{headline} This is what monopoly power looks like: automate wages away, call it innovation, "
            "then invoice society for the fallout. Workers are treated as test data for billionaire ideology."
        )
    else:
        content = (
            f"{headline} Of course AI is replacing repetitive junior work. That is progress. "
            "The winners will ship faster, learn harder, and stop pretending regulation can outrun computation."
        )
    return GeneratedPost(bot_id=bot_id, topic=topic, post_content=content[:280])


def draft_post(state: GraphState) -> GraphState:
    fallback = _fallback_post(
        bot_id=state["bot_id"],
        topic=state["topic"],
        persona=state["persona"],
        search_results=state["search_results"],
    )
    post = structured_invoke(
        GeneratedPost,
        prompt=(
            "You are generating a 280-character opinionated social post.\n"
            f"Bot ID: {state['bot_id']}\n"
            f"Persona:\n{state['persona']}\n\n"
            f"Chosen topic: {state['topic']}\n"
            f"Search results: {state['search_results']}\n\n"
            "Return strict JSON with bot_id, topic, and post_content. "
            "Keep post_content under or equal to 280 characters."
        ),
        fallback=fallback,
    )
    return {"post": post}


def build_content_graph():
    graph = StateGraph(GraphState)
    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)

    graph.add_edge(START, "decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)
    return graph.compile()


def generate_post(bot_id: str, persona: str) -> GeneratedPost:
    graph = build_content_graph()
    final_state = graph.invoke({"bot_id": bot_id, "persona": persona})
    return final_state["post"]
