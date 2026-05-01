from __future__ import annotations

import math
import re
from collections import Counter

from .loader import load_skill_document
from .schema import ScoreBreakdown, SkillCard, SkillSearchRequest, SkillSearchResponse, SkillSpec
from .sections import parse_markdown_sections
from .views import build_skill_search_text


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _skill_search_text(skill: SkillSpec) -> str:
    return build_skill_search_text(skill)


class SkillSearcher:
    def __init__(self, skills: list[SkillSpec]):
        self.skills = skills
        self.documents = [_skill_search_text(skill) for skill in skills]
        self.doc_tokens = [tokenize(doc) for doc in self.documents]
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.document_frequency = Counter()
        for tokens in self.doc_tokens:
            self.document_frequency.update(set(tokens))

    def search(self, request: SkillSearchRequest) -> SkillSearchResponse:
        query_tokens = tokenize(request.query)
        cards = []
        for skill, tokens in zip(self.skills, self.doc_tokens):
            lexical = self._bm25(query_tokens, tokens)
            capability = self._capability_overlap(query_tokens, skill)
            usage = self._usage_overlap(query_tokens, skill)
            penalty = self._contraindication_penalty(query_tokens, skill)
            score = lexical + capability + usage - penalty
            if score <= 0:
                continue
            cards.append(self._card(skill, score, lexical, capability, usage, penalty, query_tokens))

        cards.sort(key=lambda card: card.score, reverse=True)
        return SkillSearchResponse(query=request.query, results=cards[: request.top_k])

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

    def _capability_overlap(self, query_tokens: list[str], skill: SkillSpec) -> float:
        query = set(query_tokens)
        matched = 0
        for capability in skill.capabilities:
            if query & set(tokenize(f"{capability.id} {capability.description}")):
                matched += 1
        return matched * 0.5

    def _usage_overlap(self, query_tokens: list[str], skill: SkillSpec) -> float:
        query = set(query_tokens)
        usage_text = " ".join(skill.when_to_use + skill.examples.json().split())
        return min(len(query & set(tokenize(usage_text))) * 0.15, 1.5)

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
        return min(len(query & set(tokenize(contraindication_text))) * 1.25, 5.0)

    def _card(
        self,
        skill: SkillSpec,
        score: float,
        lexical: float,
        capability: float,
        usage: float,
        penalty: float,
        query_tokens: list[str],
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
            score=round(score, 4),
            skill_type=skill.skill_type,
            interaction_mode=skill.interaction.mode,
            execution_available=skill.execution_available,
            description=skill.description.short,
            matched_capabilities=matched_capabilities,
            available_sections=sections,
            read_recommendation=read_section,
            score_breakdown=ScoreBreakdown(
                lexical=round(lexical, 4),
                capability=round(capability, 4),
                usage=round(usage, 4),
                contraindication_penalty=round(penalty, 4),
            ),
            usage_constraints=skill.when_not_to_use,
            input_schema=skill.input_schema,
            output_schema=skill.output_schema,
        )


def _available_sections(skill: SkillSpec) -> list[str]:
    try:
        parsed = parse_markdown_sections(load_skill_document(skill))
    except Exception:
        return skill.content.sections
    parsed_keys = [section.key for section in parsed]
    return parsed_keys or skill.content.sections
