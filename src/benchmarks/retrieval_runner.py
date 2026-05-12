"""Command runner for local skill retrieval benchmark evaluation."""

from __future__ import annotations

import argparse

from core.search import SkillSearcher
from schema import SkillSpec

from .retrieval import evaluate_retrieval, load_retrieval_dataset


def run_eval_retrieval(args: argparse.Namespace, skills: list[SkillSpec]):
    examples = load_retrieval_dataset(args.dataset)
    return evaluate_retrieval(SkillSearcher(skills), examples, args.top_k)
