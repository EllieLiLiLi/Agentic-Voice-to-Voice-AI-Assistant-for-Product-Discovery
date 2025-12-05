"""Pydantic models and typed data containers for the project."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Product(BaseModel):
    """Normalized product representation used across components."""

    id: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    categories: List[str] = []
    url: Optional[str] = None
    rating: Optional[float] = None


class SearchResult(BaseModel):
    """Generic search result with provenance information."""

    product: Product
    score: float
    source: str  # e.g., "local" or "web"
