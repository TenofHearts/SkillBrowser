"""Hybrid sparse and dense retrieval for ranking skill specifications."""

from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from loader import load_skill_document
from schema import ScoreBreakdown, SkillCard, SkillSearchRequest, SkillSearchResponse, SkillSpec
from .embeddings import NullEmbedder, TextEmbedder, cosine as dense_cosine
from .sections import parse_markdown_sections
from .views import build_skill_search_text, build_skill_views


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "only",
    "or",
    "read",
    "the",
    "this",
    "to",
    "use",
    "user",
    "when",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOP_TOKENS]


def _skill_search_text(skill: SkillSpec) -> str:
    return build_skill_search_text(skill)


@dataclass(frozen=True)
class _RankedScore:
    skill_index: int
    score: float


@dataclass(frozen=True)
class _SearchScores:
    lexical: float
    sparse_view: float
    dense: float
    dense_description: float
    dense_capability: float
    dense_usage: float
    dense_schema: float
    dense_section: float
    rrf: float
    capability: float
    usage: float
    input_type: float
    output_type: float
    penalty: float
    negative_signal_penalty: float
    hard_mismatch_penalty: float

    def total(self, weights: "SearchWeights") -> float:
        return (
            self.lexical * weights.lexical
            + self.sparse_view * weights.sparse_view
            + self.dense * weights.dense
            + self.rrf * weights.rrf
            + self.capability * weights.capability
            + self.usage * weights.usage
            + self.input_type * weights.input_type
            + self.output_type * weights.output_type
            - self.penalty * weights.penalty
            - self.negative_signal_penalty * weights.penalty
            - self.hard_mismatch_penalty * weights.penalty
        )


@dataclass(frozen=True)
class SearchWeights:
    lexical: float = 0.2
    sparse_view: float = 0.1
    dense: float = 10.0
    rrf: float = 0.2
    capability: float = 0.0
    usage: float = 0.0
    input_type: float = 0.0
    output_type: float = 0.0
    penalty: float = 1.0


class SkillSearcher:
    def __init__(
        self,
        skills: list[SkillSpec],
        recall_k: int = 50,
        minimum_score_threshold: float = 0.0,
        *,
        embedder: TextEmbedder | None = None,
        dense_enabled: bool = False,
        bm25_enabled: bool = True,
        sparse_view_enabled: bool = True,
        dense_view_names: set[str] | None = None,
        weights: SearchWeights | None = None,
        dense_cache_dir: str | Path | None = None,
    ):
        self.skills = skills
        self.recall_k = max(recall_k, 1)
        self.minimum_score_threshold = minimum_score_threshold
        self.embedder = embedder or NullEmbedder()
        self.dense_enabled = dense_enabled and not isinstance(self.embedder, NullEmbedder)
        self.bm25_enabled = bm25_enabled
        self.sparse_view_enabled = sparse_view_enabled
        self.dense_view_names = dense_view_names
        self.weights = weights or SearchWeights()
        self.dense_cache_dir = Path(dense_cache_dir) if dense_cache_dir else None
        self.documents = [_skill_search_text(skill) for skill in skills]
        self.doc_tokens = [tokenize(doc) for doc in self.documents]
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.document_frequency = Counter()
        for tokens in self.doc_tokens:
            self.document_frequency.update(set(tokens))
        self.skill_views = [build_skill_views(skill) for skill in skills]
        self.view_tokens = [
            {view.view_name: tokenize(view.text) for view in views}
            for views in self.skill_views
        ]
        self.view_names = sorted({view.view_name for views in self.skill_views for view in views})
        self.view_document_frequency: dict[str, Counter[str]] = {}
        for view_name in self.view_names:
            frequency: Counter[str] = Counter()
            for skill_view_tokens in self.view_tokens:
                tokens = skill_view_tokens.get(view_name, [])
                if tokens:
                    frequency.update(set(tokens))
            self.view_document_frequency[view_name] = frequency
        self.dense_view_embeddings: dict[str, list[list[float]]] = {}
        if self.dense_enabled:
            self._build_dense_view_embeddings()

    def search(self, request: SkillSearchRequest, *, top_k: int = 5) -> SkillSearchResponse:
        top_k = min(max(top_k, 1), 50)
        abstention_reason = self._hard_abstention_reason(request)
        if abstention_reason:
            return SkillSearchResponse(
                query=request.query,
                results=[],
                abstained=True,
                abstention_reason=abstention_reason,
            )
        query_text = _request_query_text(request)
        query_tokens = tokenize(query_text)
        lexical_raw = (
            [self._bm25(query_tokens, tokens) for tokens in self.doc_tokens]
            if self.bm25_enabled
            else [0.0 for _ in self.doc_tokens]
        )
        lexical_rank = _rank_scores(lexical_raw)
        sparse_view_raw = (
            {
                view_name: [
                    self._token_vector_cosine(query_tokens, skill_view_tokens.get(view_name, []), view_name)
                    for skill_view_tokens in self.view_tokens
                ]
                for view_name in self.view_names
            }
            if self.sparse_view_enabled
            else {}
        )
        sparse_view_ranks = [_rank_scores(scores) for scores in sparse_view_raw.values()]
        dense_raw = self._dense_view_scores(request)
        dense_ranks = [_rank_scores(scores) for scores in dense_raw.values()]
        rrf_raw = self._rrf_fuse([lexical_rank, *sparse_view_ranks, *dense_ranks])
        rrf_norm = _normalize_by_max(rrf_raw)
        lexical_norm = _normalize_by_max(lexical_raw)
        sparse_view_norm = self._normalized_weighted_view_scores(sparse_view_raw)
        dense_components = self._dense_multiview_components(dense_raw)
        dense_norm = dense_components["dense"]

        rrf_ranked_indexes = sorted(rrf_raw, key=lambda index: rrf_raw[index], reverse=True)
        candidate_indexes = set(rrf_ranked_indexes[: self.recall_k])
        for index, skill in enumerate(self.skills):
            if self._request_capability_signal(request, skill) > 0:
                candidate_indexes.add(index)
            if self._type_overlap(request.input_types, skill.input_types) > 0:
                candidate_indexes.add(index)
            if self._type_overlap(request.output_types, skill.output_types) > 0:
                candidate_indexes.add(index)

        cards = []
        for index in candidate_indexes:
            skill = self.skills[index]
            capability = self._capability_signal(query_tokens, request, skill)
            usage = self._usage_overlap(query_tokens, skill)
            input_type = self._type_overlap(request.input_types, skill.input_types)
            output_type = self._type_overlap(request.output_types, skill.output_types)
            penalty = self._contraindication_penalty(query_tokens, skill)
            negative_signal_penalty = self._negative_signal_penalty(request, skill)
            hard_mismatches = self._hard_mismatches(
                request,
                skill,
                dense_capability=dense_components["capability"][index],
            )
            hard_mismatch_penalty = float(len(hard_mismatches)) * 2.0
            scores = _SearchScores(
                lexical=lexical_norm[index],
                sparse_view=sparse_view_norm[index],
                dense=dense_norm[index],
                dense_description=dense_components["description"][index],
                dense_capability=dense_components["capability"][index],
                dense_usage=dense_components["usage"][index],
                dense_schema=dense_components["schema"][index],
                dense_section=dense_components["section"][index],
                rrf=rrf_norm.get(index, 0.0),
                capability=capability,
                usage=usage,
                input_type=input_type,
                output_type=output_type,
                penalty=penalty,
                negative_signal_penalty=negative_signal_penalty,
                hard_mismatch_penalty=hard_mismatch_penalty,
            )
            total_score = scores.total(self.weights)
            if total_score <= self.minimum_score_threshold:
                continue
            cards.append(self._card(skill, request, scores, query_tokens, total_score, hard_mismatches))

        cards.sort(key=lambda card: card.score, reverse=True)
        results = cards[:top_k]
        if request.required_capabilities and results and all(
            "required_capability_miss" in card.hard_mismatches for card in results
        ):
            return SkillSearchResponse(
                query=request.query,
                results=[],
                abstained=True,
                abstention_reason="required_capability_miss",
            )
        if not results and request.required_capabilities and all(
            self._request_capability_signal(request, skill) <= 0 for skill in self.skills
        ):
            return SkillSearchResponse(
                query=request.query,
                results=[],
                abstained=True,
                abstention_reason="required_capability_miss",
            )
        return SkillSearchResponse(
            query=request.query,
            results=results,
            abstained=not results,
            abstention_reason=None if results else "no_candidate_above_threshold",
        )

    def _bm25(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in set(query_tokens):
            df = self.document_frequency.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self.skills) - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / max(self.avg_doc_len, 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _token_vector_cosine(self, query_tokens: list[str], doc_tokens: list[str], view_name: str) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        query_vector = self._tfidf_vector(query_tokens, self.view_document_frequency[view_name])
        doc_vector = self._tfidf_vector(doc_tokens, self.view_document_frequency[view_name])
        return _cosine(query_vector, doc_vector)

    def _tfidf_vector(self, tokens: list[str], document_frequency: Counter[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vector = {}
        for token, count in counts.items():
            df = document_frequency.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self.skills) + 1) / (df + 1))
            vector[token] = (1 + math.log(count)) * idf
        return vector

    def _normalized_weighted_view_scores(self, view_raw: dict[str, list[float]]) -> list[float]:
        totals = [0.0 for _ in self.skills]
        weights = [0.0 for _ in self.skills]
        for view_name, scores in view_raw.items():
            weight = _view_weight(view_name)
            normalized = _normalize_by_max(scores)
            for index, value in enumerate(normalized):
                if scores[index] <= 0:
                    continue
                totals[index] += value * weight
                weights[index] += weight
        return [totals[index] / weights[index] if weights[index] else 0.0 for index in range(len(self.skills))]

    def _build_dense_view_embeddings(self) -> None:
        for view_name in self.view_names:
            if self.dense_view_names is not None and view_name not in self.dense_view_names:
                continue
            texts = []
            skill_ids = []
            for skill_views in self.skill_views:
                text_by_view = {view.view_name: view.text for view in skill_views}
                texts.append(text_by_view.get(view_name, ""))
            for skill in self.skills:
                skill_ids.append(skill.id)
            self.dense_view_embeddings[view_name] = self._load_or_embed_view(view_name, skill_ids, texts)

    def _load_or_embed_view(self, view_name: str, skill_ids: list[str], texts: list[str]) -> list[list[float]]:
        cache_path = self._dense_cache_path(view_name, skill_ids, texts)
        if cache_path and cache_path.exists():
            cached = _read_dense_cache(cache_path)
            if cached is not None and len(cached) == len(texts):
                return cached
        embeddings = self.embedder.embed_texts(texts)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_dense_cache(cache_path, embeddings)
        return embeddings

    def _dense_cache_path(self, view_name: str, skill_ids: list[str], texts: list[str]) -> Path | None:
        if self.dense_cache_dir is None:
            return None
        signature = hashlib.sha256()
        signature.update(self.embedder.model_name.encode("utf-8"))
        signature.update(view_name.encode("utf-8"))
        for skill_id, text in zip(skill_ids, texts):
            signature.update(skill_id.encode("utf-8"))
            signature.update(hashlib.sha256(text.encode("utf-8")).digest())
        safe_view_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", view_name)
        return self.dense_cache_dir / f"{safe_view_name}-{signature.hexdigest()[:16]}.jsonl"

    def _dense_view_scores(self, request: SkillSearchRequest) -> dict[str, list[float]]:
        if not self.dense_enabled or not self.dense_view_embeddings:
            return {}
        query_by_group = _dense_query_groups(request)
        embeddings_by_group = {
            group: self.embedder.embed_texts([query_text])[0]
            for group, query_text in query_by_group.items()
            if query_text.strip()
        }
        scores: dict[str, list[float]] = {}
        for view_name, embeddings in self.dense_view_embeddings.items():
            group = _dense_group_for_view(view_name)
            query_embedding = embeddings_by_group.get(group)
            if query_embedding is None:
                scores[view_name] = [0.0 for _ in embeddings]
                continue
            scores[view_name] = [dense_cosine(query_embedding, embedding) for embedding in embeddings]
        return scores

    def _dense_multiview_components(self, dense_raw: dict[str, list[float]]) -> dict[str, list[float]]:
        components = {
            "description": [0.0 for _ in self.skills],
            "capability": [0.0 for _ in self.skills],
            "usage": [0.0 for _ in self.skills],
            "schema": [0.0 for _ in self.skills],
            "section": [0.0 for _ in self.skills],
        }
        for group in components:
            grouped_scores = {
                view_name: scores
                for view_name, scores in dense_raw.items()
                if _dense_group_for_view(view_name) == group
            }
            components[group] = self._normalized_weighted_view_scores(grouped_scores)

        components["dense"] = [
            (
                0.25 * components["description"][index]
                + 0.35 * components["capability"][index]
                + 0.20 * components["usage"][index]
                + 0.10 * components["schema"][index]
                + 0.10 * components["section"][index]
            )
            for index in range(len(self.skills))
        ]
        return components

    def _rrf_fuse(self, rank_lists: list[list[_RankedScore]], k: int = 60) -> dict[int, float]:
        fused: defaultdict[int, float] = defaultdict(float)
        for rank_list in rank_lists:
            for rank, ranked_score in enumerate(rank_list, start=1):
                fused[ranked_score.skill_index] += 1 / (k + rank)
        return dict(fused)

    def _capability_signal(self, query_tokens: list[str], request: SkillSearchRequest, skill: SkillSpec) -> float:
        overlap = self._capability_overlap(query_tokens, skill)
        requested = self._request_capability_signal(request, skill)
        return min(overlap + requested, 1.0)

    def _capability_overlap(self, query_tokens: list[str], skill: SkillSpec) -> float:
        query = set(query_tokens)
        matched = 0
        for capability in skill.capabilities:
            if query & set(tokenize(f"{capability.id} {capability.description}")):
                matched += 1
        return min(matched / max(len(skill.capabilities), 1), 1.0)

    def _request_capability_signal(self, request: SkillSearchRequest, skill: SkillSpec) -> float:
        if not request.required_capabilities:
            return 0.0
        requested = set(tokenize(" ".join(request.required_capabilities)))
        if not requested:
            return 0.0
        matched = 0
        for capability in skill.capabilities:
            capability_tokens = set(tokenize(f"{capability.id} {capability.description}"))
            if requested & capability_tokens:
                matched += 1
        return min(matched / len(request.required_capabilities), 1.0)

    def _usage_overlap(self, query_tokens: list[str], skill: SkillSpec) -> float:
        query = set(query_tokens)
        usage_text = " ".join(skill.when_to_use + skill.examples.json().split())
        return min(len(query & set(tokenize(usage_text))) * 0.12, 1.0)

    def _type_overlap(self, requested_types: list[str], skill_types: list[str]) -> float:
        if not requested_types:
            return 0.0
        requested = set(tokenize(" ".join(requested_types)))
        available = set(tokenize(" ".join(skill_types)))
        if not requested or not available:
            return 0.0
        return len(requested & available) / len(requested)

    def _contraindication_penalty(self, query_tokens: list[str], skill: SkillSpec) -> float:
        query = set(query_tokens)
        contraindication_text = " ".join(skill.when_not_to_use)
        try:
            contraindication_text += "\n" + "\n".join(
                section.content
                for section in parse_markdown_sections(load_skill_document(skill))
                if section.key in {"failure_modes", "when_not_to_use", "contraindications"}
            )
        except Exception:
            pass
        positive_text = _skill_positive_signal_text(skill)
        negative_only_tokens = set(tokenize(contraindication_text)) - set(tokenize(positive_text))
        return min(len(query & negative_only_tokens) * 3.0, 12.0)

    def _negative_signal_penalty(self, request: SkillSearchRequest, skill: SkillSpec) -> float:
        if not request.negative_signals:
            return 0.0
        negative_tokens = set(tokenize(" ".join(request.negative_signals)))
        if not negative_tokens:
            return 0.0
        skill_negative_text = _skill_negative_signal_text(skill)
        skill_positive_text = _skill_positive_signal_text(skill)
        negative_only_tokens = set(tokenize(skill_negative_text)) - set(tokenize(skill_positive_text))
        return min(len(negative_tokens & negative_only_tokens) * 4.0, 16.0)

    def _hard_mismatches(
        self,
        request: SkillSearchRequest,
        skill: SkillSpec,
        *,
        dense_capability: float = 0.0,
    ) -> list[str]:
        mismatches: list[str] = []
        if (
            request.required_capabilities
            and self._request_capability_signal(request, skill) <= 0
            and dense_capability < 0.15
        ):
            mismatches.append("required_capability_miss")
        if request.input_types and self._type_overlap(request.input_types, skill.input_types) <= 0:
            mismatches.append("input_type_miss")
        if request.constraints.get("readable_required") is True and not skill.interaction.readable:
            mismatches.append("readable_required_miss")
        if request.constraints.get("execution_allowed") is False and skill.execution_available:
            mismatches.append("execution_not_allowed")
        if request.constraints.get("execution_required") is True and not skill.execution_available:
            mismatches.append("execution_required_miss")
        return mismatches

    def _hard_abstention_reason(self, request: SkillSearchRequest) -> str | None:
        if not self.skills:
            return "empty_skill_library"
        if request.input_types and all(self._type_overlap(request.input_types, skill.input_types) <= 0 for skill in self.skills):
            return "input_type_miss"
        if request.constraints.get("readable_required") is True and all(
            not skill.interaction.readable for skill in self.skills
        ):
            return "readable_required_miss"
        if request.constraints.get("execution_required") is True and all(
            not skill.execution_available for skill in self.skills
        ):
            return "execution_required_miss"
        return None

    def _card(
        self,
        skill: SkillSpec,
        request: SkillSearchRequest,
        scores: _SearchScores,
        query_tokens: list[str],
        total_score: float,
        hard_mismatches: list[str],
    ) -> SkillCard:
        matched_capabilities = [
            capability.id
            for capability in skill.capabilities
            if set(query_tokens) & set(tokenize(f"{capability.id} {capability.description}"))
        ]
        sections = _available_sections(skill)
        read_section = skill.interaction.default_read_level
        if read_section not in sections and sections:
            read_section = sections[0]
        return SkillCard(
            id=skill.id,
            name=skill.name,
            score=round(total_score, 4),
            skill_type=skill.skill_type,
            interaction_mode=skill.interaction.mode,
            execution_available=skill.execution_available,
            description=skill.description.short,
            matched_capabilities=matched_capabilities,
            available_sections=sections,
            read_recommendation=read_section,
            score_breakdown=ScoreBreakdown(
                lexical=round(scores.lexical, 4),
                capability=round(scores.capability, 4),
                usage=round(scores.usage, 4),
                sparse_view=round(scores.sparse_view, 4),
                dense=round(scores.dense, 4),
                dense_description=round(scores.dense_description, 4),
                dense_capability=round(scores.dense_capability, 4),
                dense_usage=round(scores.dense_usage, 4),
                dense_schema=round(scores.dense_schema, 4),
                dense_section=round(scores.dense_section, 4),
                vector=round(scores.dense, 4),
                rrf=round(scores.rrf, 4),
                input_type=round(scores.input_type, 4),
                output_type=round(scores.output_type, 4),
                contraindication_penalty=round(scores.penalty, 4),
                negative_signal_penalty=round(scores.negative_signal_penalty, 4),
                hard_mismatch_penalty=round(scores.hard_mismatch_penalty, 4),
            ),
            usage_constraints=skill.when_not_to_use,
            missing_capabilities=self._missing_required_capabilities(request, skill),
            negative_matches=self._negative_matches(request, skill),
            hard_mismatches=hard_mismatches,
            input_schema=skill.input_schema,
            output_schema=skill.output_schema,
        )

    def _missing_required_capabilities(self, request: SkillSearchRequest, skill: SkillSpec) -> list[str]:
        if not request.required_capabilities:
            return []
        skill_tokens = set()
        for capability in skill.capabilities:
            skill_tokens.update(tokenize(f"{capability.id} {capability.description}"))
        return [
            capability
            for capability in request.required_capabilities
            if not set(tokenize(capability)) & skill_tokens
        ]

    def _negative_matches(self, request: SkillSearchRequest, skill: SkillSpec) -> list[str]:
        if not request.negative_signals:
            return []
        skill_negative_tokens = set(tokenize(_skill_negative_signal_text(skill)))
        matches = []
        for signal in request.negative_signals:
            if set(tokenize(signal)) & skill_negative_tokens:
                matches.append(signal)
        return matches


def _available_sections(skill: SkillSpec) -> list[str]:
    try:
        parsed = parse_markdown_sections(load_skill_document(skill))
    except Exception:
        return skill.content.sections
    parsed_keys = [section.key for section in parsed]
    return parsed_keys or skill.content.sections


def _request_query_text(request: SkillSearchRequest) -> str:
    return "\n".join(
        [
            request.query,
            request.task_context or "",
            " ".join(request.required_capabilities),
            " ".join(request.desired_capabilities),
            " ".join(request.input_types),
            " ".join(request.output_types),
            " ".join(request.positive_signals),
            " ".join(request.negative_signals),
            _constraints_text(request.constraints),
        ]
    )


def _dense_query_groups(request: SkillSearchRequest) -> dict[str, str]:
    return {
        "description": "\n".join([request.query, request.task_context or ""]),
        "capability": "\n".join(
            [
                " ".join(request.required_capabilities),
                " ".join(request.desired_capabilities),
                request.query,
            ]
        ),
        "usage": "\n".join([" ".join(request.positive_signals), request.query]),
        "schema": "\n".join(
            [
                " ".join(request.input_types),
                " ".join(request.output_types),
                _constraints_text(request.constraints),
            ]
        ),
        "section": "\n".join([request.task_context or "", request.query]),
    }


def _dense_group_for_view(view_name: str) -> str:
    if view_name == "capability":
        return "capability"
    if view_name in {"usage", "examples"}:
        return "usage"
    if view_name == "schema":
        return "schema"
    if view_name.startswith("content_section:"):
        return "section"
    return "description"


def _skill_positive_signal_text(skill: SkillSpec) -> str:
    return "\n".join(
        [
            skill.name,
            skill.description.short,
            skill.description.long or "",
            " ".join(f"{capability.id} {capability.description}" for capability in skill.capabilities),
            " ".join(skill.when_to_use),
            " ".join(example.user_query for example in skill.examples.positive),
            " ".join(skill.input_types),
            " ".join(skill.output_types),
            str(skill.input_schema or ""),
            str(skill.output_schema or ""),
            " ".join(skill.tags),
        ]
    )


def _skill_negative_signal_text(skill: SkillSpec) -> str:
    negative_examples = " ".join(example.user_query for example in skill.examples.negative)
    failure_sections = ""
    try:
        failure_sections = "\n".join(
            section.content
            for section in parse_markdown_sections(load_skill_document(skill))
            if section.key in {"failure_modes", "when_not_to_use", "contraindications"}
        )
    except Exception:
        pass
    return "\n".join([*skill.when_not_to_use, negative_examples, failure_sections])


def _constraints_text(constraints: dict[str, object]) -> str:
    parts = []
    for key, value in sorted(constraints.items()):
        if value is None or value == "":
            continue
        parts.append(f"{key}: {value}")
    return " ".join(parts)


def rrf_fusion(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, skill_id in enumerate(rank_list, start=1):
            scores[skill_id] += 1.0 / (k + rank)
    return dict(scores)


def _rank_scores(scores: list[float]) -> list[_RankedScore]:
    ranked = [_RankedScore(index, score) for index, score in enumerate(scores) if score > 0]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _normalize_by_max(scores: list[float] | dict[int, float]) -> list[float] | dict[int, float]:
    values = list(scores.values()) if isinstance(scores, dict) else scores
    max_score = max(values, default=0.0)
    if max_score <= 0:
        if isinstance(scores, dict):
            return {key: 0.0 for key in scores}
        return [0.0 for _ in scores]
    if isinstance(scores, dict):
        return {key: value / max_score for key, value in scores.items()}
    return [value / max_score for value in scores]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _read_dense_cache(path: Path) -> list[list[float]] | None:
    try:
        return [json.loads(line)["vector"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_dense_cache(path: Path, embeddings: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for position, vector in enumerate(embeddings):
            handle.write(json.dumps({"position": position, "vector": vector}) + "\n")


def _view_weight(view_name: str) -> float:
    if view_name == "capability":
        return 1.4
    if view_name in {"usage", "examples"}:
        return 1.2
    if view_name == "schema":
        return 1.1
    if view_name.startswith("content_section:"):
        return 0.7
    return 1.0
