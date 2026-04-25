from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from cognitive_routing_rag.config import settings


class TopicDecision(BaseModel):
    topic: str
    search_query: str


class GeneratedPost(BaseModel):
    bot_id: str
    topic: str
    post_content: str


def get_chat_model():
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=0.4)
    if provider == "groq":
        if not settings.groq_api_key:
            return None
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.llm_model, temperature=0.4)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.4,
        )
    return None


def structured_invoke(
    schema: type[BaseModel],
    prompt: str,
    fallback: BaseModel,
) -> BaseModel:
    llm = get_chat_model()
    if llm is None:
        return fallback

    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke(prompt)


def text_invoke(prompt: str, fallback_text: str) -> str:
    llm = get_chat_model()
    if llm is None:
        return fallback_text

    response: AIMessage = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return json.dumps(content)
    return str(content)
