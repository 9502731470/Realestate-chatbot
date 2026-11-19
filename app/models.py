"""Domain models shared by the scrapers, the index and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["darglobal", "wasalt"]
Kind = Literal["listing", "project", "article", "page", "faq", "stat", "guide"]

SOURCE_LABELS: dict[str, str] = {"darglobal": "DarGlobal", "wasalt": "Wasalt.sa"}

# Approximate FX → USD, used ONLY to compare prices across currencies when
# filtering (Gulf currencies are USD-pegged; EUR/GBP are approximations).
FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "SAR": 1 / 3.75,
    "AED": 1 / 3.6725,
    "QAR": 1 / 3.64,
    "OMR": 1 / 0.3845,
    "BHD": 1 / 0.376,
    "KWD": 3.26,
    "EUR": 1.08,
    "GBP": 1.27,
}


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_usd(amount: float | None, currency: str | None) -> float | None:
    if amount is None:
        return None
    rate = FX_TO_USD.get((currency or "SAR").upper())
    return round(amount * rate, 2) if rate else None


RENT_PERIODS_PER_YEAR = {"year": 1, "month": 12, "week": 52, "day": 365}


def comparable_usd(amount: float | None, currency: str | None, purpose: str | None, rent_period: str | None) -> float | None:
    """Price in USD on a common basis: rents are annualised so a monthly
    listing at 5,000/month compares correctly against a 48,000/year budget."""
    if amount is None:
        return None
    rate = FX_TO_USD.get((currency or "SAR").upper())
    if not rate:
        return None
    usd = amount * rate
    if purpose == "rent" and rent_period:
        usd *= RENT_PERIODS_PER_YEAR.get(rent_period, 1)
    return round(usd, 2)


class Record(BaseModel):
    """One unit of knowledge: a listing, a project, an article, an FAQ, ..."""

    id: str
    source: Source
    kind: Kind
    url: str
    title: str
    language: str = "en"
    summary: str | None = None
    text: str = ""

    # Structured attributes (mostly listings / projects)
    purpose: Literal["sale", "rent"] | None = None
    property_type: str | None = None
    property_types: list[str] = Field(default_factory=list)
    category: Literal["residential", "commercial"] | None = None
    price: float | None = None
    price_max: float | None = None
    currency: str | None = None
    rent_period: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_sqm: float | None = None
    area_max_sqm: float | None = None
    city: str | None = None
    district: str | None = None
    country: str | None = None
    developer: str | None = None
    project: str | None = None
    status: str | None = None
    completion: str | None = None
    furnishing: str | None = None
    amenities: list[str] = Field(default_factory=list)
    image: str | None = None

    published_at: datetime | None = None
    updated_at: datetime | None = None
    scraped_at: datetime = Field(default_factory=utcnow)
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def price_usd(self) -> float | None:
        return to_usd(self.price, self.currency)

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    def location_label(self) -> str:
        parts = [p for p in (self.district, self.city, self.country) if p]
        # Avoid "Riyadh, Riyadh" style repetition.
        seen: list[str] = []
        for part in parts:
            if part.lower() not in (s.lower() for s in seen):
                seen.append(part)
        return ", ".join(seen)

    def price_label(self) -> str | None:
        if self.price is None:
            return None
        cur = self.currency or ""

        def fmt(value: float) -> str:
            if value >= 1_000_000:
                return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
            return f"{value:,.0f}"

        label = f"{fmt(self.price)} {cur}".strip()
        if self.price_max and self.price_max > self.price:
            label = f"{fmt(self.price)}–{fmt(self.price_max)} {cur}".strip()
            if self.kind == "project":
                label = f"From {label}"
        if self.purpose == "rent" and self.rent_period:
            label += f" / {self.rent_period}"
        return label


class CrawlReport(BaseModel):
    source: str
    ok: bool = True
    records: int = 0
    requests: int = 0
    errors: int = 0
    blocked_by_robots: int = 0
    notes: list[str] = Field(default_factory=list)
    status_codes: dict[str, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
