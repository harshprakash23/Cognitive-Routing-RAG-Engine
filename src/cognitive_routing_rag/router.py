from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from cognitive_routing_rag.personas import PERSONAS


@dataclass
class RoutedBot:
    bot_id: str
    name: str
    similarity: float


class LocalEmbeddingModel:
    def __init__(self, n_features: int = 512) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class PersonaRouter:
    def __init__(self) -> None:
        self.embedding_model = LocalEmbeddingModel()
        self.personas = PERSONAS
        persona_vectors = self.embedding_model.embed(
            [self._expand_persona_text(persona) for persona in self.personas]
        )
        self.index = faiss.IndexFlatIP(persona_vectors.shape[1])
        self.index.add(persona_vectors)

    @staticmethod
    def _expand_persona_text(persona: dict[str, str]) -> str:
        tags_by_bot = {
            "bot_a": (
                "Topics: AI, OpenAI, LLMs, coding, developers, software engineers, automation, "
                "junior developers, technological progress, crypto, Elon Musk, rockets, space exploration."
            ),
            "bot_b": (
                "Topics: labor exploitation, job displacement, anti-billionaire politics, monopolies, "
                "surveillance, privacy, AI harms, social damage from automation, worker precarity."
            ),
            "bot_c": (
                "Topics: markets, trading, rates, monetization, margins, ROI, capital allocation, "
                "productivity gains, labor efficiency, software economics."
            ),
        }
        return f"{persona['persona']} {tags_by_bot.get(persona['bot_id'], '')}"

    def route_post_to_bots(
        self, post_content: str, threshold: float = 0.85
    ) -> list[RoutedBot]:
        query_vector = self.embedding_model.embed([post_content])
        similarities, indices = self.index.search(query_vector, len(self.personas))

        matches: list[RoutedBot] = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1 or similarity < threshold:
                continue
            bot = self.personas[idx]
            matches.append(
                RoutedBot(
                    bot_id=bot["bot_id"],
                    name=bot["name"],
                    similarity=float(similarity),
                )
            )
        return matches
