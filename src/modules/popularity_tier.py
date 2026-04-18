from __future__ import annotations

import re
from datetime import date
from typing import Any

BOX_OFFICE_RELATION = "http://www.wikidata.org/prop/direct/P2142"
RELEASE_DATE_RELATION = "http://www.wikidata.org/prop/direct/P577"
POPULARITY_TIER_RELATION = "http://ddis.ch/atai/popularity_tier"


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d{4})", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _base_tier_from_box_office(box_office: float | None) -> int:
    if box_office is None:
        return 1
    if box_office > 200_000_000:
        return 4
    if box_office >= 80_000_000:
        return 3
    if box_office >= 10_000_000:
        return 2
    return 1


def popularity_tier_for_values(
    box_office_value: Any,
    release_date_value: Any,
    current_year: int | None = None,
) -> int:
    """
    Compute popularity tier from box office and release year.

    Rules:
    - > 200M => 4
    - 80M to 200M => 3
    - 10M to 80M => 2
    - < 10M or missing => 1
    - every full decade before the current year deducts 1 point
    - minimum tier is 1
    """
    if current_year is None:
        current_year = date.today().year

    base = _base_tier_from_box_office(_parse_float(box_office_value))
    release_year = _parse_year(release_date_value)
    if release_year is None:
        return max(1, base)

    years_old = max(0, int(current_year) - int(release_year))
    decade_penalty = years_old // 10
    return max(1, base - decade_penalty)


def popularity_value_label(tier: int) -> str:
    tier_int = max(1, min(4, int(tier)))
    return f"tier {tier_int}"


def popularity_weight_from_value(value: Any) -> float:
    if value is None:
        return 1.0
    m = re.search(r"([1-4])", str(value))
    if not m:
        return 1.0
    return float(int(m.group(1)))


def augment_movie_attrs_with_popularity(movie_attrs: dict[str, dict[str, set[str]]]) -> None:
    for _movie_id, attrs in movie_attrs.items():
        box_values = list(attrs.get(BOX_OFFICE_RELATION, set()) or [])
        release_values = list(attrs.get(RELEASE_DATE_RELATION, set()) or [])
        box_value = box_values[0] if box_values else None
        release_value = release_values[0] if release_values else None
        tier = popularity_tier_for_values(box_value, release_value)
        attrs.setdefault(POPULARITY_TIER_RELATION, set()).add(popularity_value_label(tier))


def augment_attr_index_with_popularity(
    attr_index: dict[str, dict[str, set[str]]],
    movie_attrs: dict[str, dict[str, set[str]]],
) -> None:
    value_to_movies = attr_index.setdefault(POPULARITY_TIER_RELATION, {})
    for movie_id, attrs in movie_attrs.items():
        for value in attrs.get(POPULARITY_TIER_RELATION, set()) or []:
            value_to_movies.setdefault(value, set()).add(movie_id)
