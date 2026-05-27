from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional


@dataclass(frozen=True)
class QueryPlan:
    query: str
    filters: dict[str, Any]
    retrieval_strategy: str
    reason: Optional[str] = None


class QueryPlanner:
    """Lightweight deterministic planner for online query routing."""

    COMPANIES = {
        "apple": "Apple",
        "amazon": "Amazon",
        "microsoft": "Microsoft",
    }

    SECTION_HINTS = {
        "risk": "Risk Factors",
        "risks": "Risk Factors",
        "business": "Business",
        "supply chain": "Risk Factors",
    }

    def plan(self, query: str, filters: Optional[dict[str, Any]] = None) -> QueryPlan:
        normalized_query = " ".join(query.split())
        if len(normalized_query) < 4:
            return QueryPlan(
                query=normalized_query,
                filters=filters or {},
                retrieval_strategy="insufficient_query",
                reason="Query is too short for reliable retrieval.",
            )

        inferred_filters = self._infer_filters(normalized_query)
        applied_filters = {**inferred_filters, **self._non_empty(filters or {})}
        strategy = "metadata_then_vector" if applied_filters else "vector_only"
        return QueryPlan(
            query=normalized_query,
            filters=applied_filters,
            retrieval_strategy=strategy,
        )

    def _infer_filters(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        filters: dict[str, Any] = {}

        for key, company in self.COMPANIES.items():
            if key in lowered:
                filters["company"] = company
                filters.setdefault("source_file", f"{company}_10k.pdf")
                break

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            filters["year"] = int(year_match.group(1))

        for hint, section_title in self.SECTION_HINTS.items():
            if hint in lowered:
                filters.setdefault("section_title", section_title)
                break

        return filters

    def _non_empty(self, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}
