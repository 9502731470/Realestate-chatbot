"""Deterministic bilingual (EN/AR) query understanding.

Turns "3-bedroom flats for rent in Riyadh under 80k" or
"شقق للإيجار في الرياض أقل من 80 ألف" into structured filters, an intent and
follow-up information - without spending an LLM call. Free OpenRouter models
are limited to 50-1000 requests/day, so each chat turn uses exactly one LLM
request (generation); understanding is done here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from rapidfuzz import fuzz

from app.textutil import is_arabic, normalize_arabic, normalize_for_search, place_key, to_ascii_digits
from app.vocab import (
    COUNTRY_CURRENCY,
    CURRENCY_ALIASES,
    MULTIPLIERS,
    PROPERTY_TYPES,
    TYPE_LABELS,
    canonical_city,
    city_alias_keys,
    city_country,
    detect_currency,
    find_places,
    parse_number,
)


@dataclass
class QueryFilters:
    purpose: str | None = None
    property_types: list[str] = field(default_factory=list)
    category: str | None = None
    bedrooms: int | None = None
    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_price: float | None = None
    max_price: float | None = None
    currency: str | None = None
    min_area: float | None = None
    max_area: float | None = None
    cities: list[str] = field(default_factory=list)
    districts: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    furnishing: str | None = None
    project_ids: list[str] = field(default_factory=list)
    sort: str | None = None
    limit: int | None = None

    STRUCTURAL = (
        "purpose", "property_types", "category", "bedrooms", "min_bedrooms", "max_bedrooms",
        "min_price", "max_price", "min_area", "max_area", "cities", "districts", "countries",
        "furnishing", "project_ids",
    )

    def is_structured(self) -> bool:
        return any(getattr(self, name) not in (None, [], "") for name in self.STRUCTURAL)

    def merged_with(self, newer: QueryFilters) -> QueryFilters:
        merged = QueryFilters(**asdict(self))
        for f in fields(QueryFilters):
            value = getattr(newer, f.name)
            if value not in (None, [], ""):
                setattr(merged, f.name, value)
        if newer.cities and not newer.districts:
            merged.districts = []  # a new city invalidates an old district
        if newer.bedrooms is not None:
            merged.min_bedrooms = newer.min_bedrooms
            merged.max_bedrooms = newer.max_bedrooms
        if newer.project_ids:
            merged.project_ids = newer.project_ids
        return merged

    def describe(self) -> str:
        bits: list[str] = []
        if self.purpose:
            bits.append(f"for {self.purpose}")
        if self.property_types:
            bits.append("type: " + "/".join(TYPE_LABELS.get(t, t) for t in self.property_types))
        if self.category:
            bits.append(self.category)
        if self.bedrooms is not None:
            bits.append(f"{self.bedrooms} bedrooms")
        if self.min_bedrooms is not None:
            bits.append(f"≥{self.min_bedrooms} bedrooms")
        if self.max_bedrooms is not None:
            bits.append(f"≤{self.max_bedrooms} bedrooms")
        cur = self.currency or ""
        if self.min_price is not None:
            bits.append(f"price ≥ {self.min_price:,.0f} {cur}".strip())
        if self.max_price is not None:
            bits.append(f"price ≤ {self.max_price:,.0f} {cur}".strip())
        if self.min_area is not None:
            bits.append(f"area ≥ {self.min_area:,.0f} sqm")
        if self.max_area is not None:
            bits.append(f"area ≤ {self.max_area:,.0f} sqm")
        if self.districts:
            bits.append("district: " + ", ".join(self.districts))
        if self.cities:
            bits.append("city: " + ", ".join(self.cities))
        if self.countries:
            bits.append("country: " + ", ".join(self.countries))
        if self.furnishing:
            bits.append(self.furnishing)
        if self.sources:
            bits.append("source: " + ", ".join(self.sources))
        if self.sort:
            bits.append(f"sorted: {self.sort}")
        return "; ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


@dataclass
class Vocabulary:
    """Index-derived names the parser should recognise."""

    districts: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # key -> (name, city)
    projects: list[tuple[str, str, str]] = field(default_factory=list)  # (alias, record_id, title)
    cities: set[str] = field(default_factory=set)


@dataclass
class ParsedQuery:
    text: str
    language: str
    intent: str
    filters: QueryFilters
    is_follow_up: bool = False
    references_previous: bool = False
    ordinal: int | None = None
    search_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "filters": self.filters.to_dict(),
            "is_follow_up": self.is_follow_up,
        }


# --------------------------------------------------------------------- patterns
_MULT = r"(k|thousand|m|mn|mil|million|millions|b|bn|billion|الف|الاف|آلاف|مليون|ملايين|مليار)"
_NB = r"(?![a-z؀-ۿ])"  # not followed by a letter ("3 baths" is not 3 billion)
_NUM = rf"(\d+(?:[.,]\d+)*)\s*(?:{_MULT}{_NB})?"
_NUM_MULT = rf"(\d+(?:[.,]\d+)*)\s*{_MULT}{_NB}"  # multiplier required, e.g. "500k"
_CUR_WORDS = sorted({a for aliases in CURRENCY_ALIASES.values() for a in aliases if len(a) > 1} | {"sar", "aed"}, key=len, reverse=True)
_CUR = (
    r"(?:(?<![a-z؀-ۿ])(?:"
    + "|".join(re.escape(normalize_arabic(c)) for c in _CUR_WORDS)
    + r")(?![a-z؀-ۿ])|\$|£|€)"
)
_GENERAL_Q = re.compile(
    r"\b(?:how|why|what is|what are|explain|process|guide|law|laws|rules|regulations?|tax|taxes|visa|mortgage|"
    r"finance|financing|loan|fees|documents?|steps)\b|كيف|لماذا|ما هو|ما هي|اشرح|قانون|ضريبه|تمويل|رهن"
)
_BED = r"(?:bed(?:room)?s?|br|bhk|b/r|bd|rooms?)"
_AREA_UNIT = (
    r"(?:sq\.?\s*m(?:eters?|etres?)?|sqm|m2|m²|square\s*met(?:er|re)s?|sq\.?\s*ft|sqft|square\s*feet|ft2|"
    r"متر(?:\s*مربع)?|م2|م²)"
)
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_AR_WORD_NUMS = {"واحده": 1, "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4, "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6, "سبع": 7, "سبعه": 7}

_MAX_WORDS = (
    r"under|below|less than|lower than|cheaper than|max(?:imum)?|up to|upto|within|budget(?:\s+(?:of|is))?|"
    r"no more than|not more than|at most|<=?|اقل من|تحت|بحدود|لا يتجاوز|حد اقصى|ميزانيه|ميزانيتي"
)
_MIN_WORDS = r"over|above|more than|at least|min(?:imum)?|starting (?:at|from)|>=?|اكثر من|فوق|لا يقل عن|من"
_AROUND_WORDS = r"around|about|approximately|approx\.?|~|roughly|حوالي|تقريبا"

_RENT = re.compile(r"\b(?:rent|rental|rentals|renting|lease|leasing|to let)\b|ايجار|للايجار|استئجار|استاجر|مستاجر|اجار")
_SALE = re.compile(r"\b(?:buy|buying|purchase|purchasing|for sale|on sale|sale|sell|selling|ownership)\b|للبيع|بيع|شراء|اشتري|تمليك|تملك")
_UNFURNISHED = re.compile(r"\bun-?furnished\b|\bnot furnished\b|غير مفروش")
_SEMI = re.compile(r"\bsemi[- ]?furnished\b|نصف مفروش|شبه مفروش")
_FURNISHED = re.compile(r"\bfurnished\b|مفروش")
_COMMERCIAL = re.compile(r"\bcommercial\b|تجاري|تجاريه")
_RESIDENTIAL = re.compile(r"\bresidential\b|سكني|سكنيه")
_DARGLOBAL = re.compile(r"\bdar\s*-?\s*global\b|darglobal|دار جلوبال|دار غلوبال|دارجلوبال|دار قلوبال")
_WASALT = re.compile(r"\bwasalt\b|وصلت")
_PROJECT_WORDS = re.compile(
    r"\b(?:projects?|developments?|off[- ]?plan|branded residences?|launch(?:es)?|masterplans?|compounds?|handover)\b|"
    r"مشروع|مشاريع|تحت الانشاء|على الخارطه"
)
_SORT_PATTERNS = [
    ("price_asc", re.compile(r"\b(?:cheapest|lowest price|least expensive|most affordable|cheaper|budget[- ]friendly)\b|ارخص|الارخص")),
    ("price_desc", re.compile(r"\b(?:most expensive|priciest|highest price|top end|pricier)\b|اغلى|الاغلى")),
    ("area_desc", re.compile(r"\b(?:largest|biggest|most spacious|bigger|larger)\b|اكبر|الاكبر|اوسع")),
    ("newest", re.compile(r"\b(?:newest|latest|most recent|recently listed|new listings)\b|احدث|الاحدث")),
]
_LIMIT = re.compile(
    r"\b(?:top|first|show me|give me|list)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:options|listings|properties|apartments|villas|results|projects|homes|units|choices)\b"
)
_GREETING = re.compile(
    r"\s*(?:hi|hello|hey|hola|salam|salaam|good (?:morning|evening|afternoon)|السلام عليكم|سلام|مرحبا|اهلا|هلا|صباح الخير|مساء الخير)[\s!.,؟?]*(?:there)?[\s!.,؟?]*"
)
_THANKS = re.compile(r"\s*(?:thanks|thank you|thx|great|perfect|ok|okay|شكرا|جزاك الله خير|ممتاز)[\s!.,؟?]*")
_HELP = re.compile(r"\b(?:what can you do|who are you|what are you|help me|how do you work|what do you know)\b|ماذا تستطيع|من انت|ساعدني")
_STATS = re.compile(r"\bhow many\b|\bnumber of\b|\bcount of\b|كم عدد|كم يوجد")
_SEARCH_VERBS = re.compile(
    r"\b(?:find|search|show|looking for|look for|recommend|suggest|options|available|want|need|any|list|get me)\b|"
    r"ابحث|اريد|ابغى|ابي|اعرض|متوفر|ابحث عن|دور لي|عندكم"
)
_FOLLOW_START = re.compile(
    r"^(?:and|what about|how about|also|same|then|cheaper|bigger|smaller|larger|more|other|another|any other|only|but|"
    r"in|with|without|for|و|طيب|وماذا|ماذا عن|وش عن|ايش عن|بس|فقط)\b"
)
_PRONOUNS = re.compile(
    r"\b(?:it|its|this|that|these|those|them|they|there|this one|that one|the ones?|first one|second one|third one|last one|above)\b|"
    r"\bهذا\b|\bهذه\b|\bذلك\b|\bتلك\b|الاول|الثاني|الثالث"
)
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "الاول": 1, "الثاني": 2, "الثالث": 3}


def _prep(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return normalize_arabic(to_ascii_digits(text)).lower()


def _amount(num: str, mult: str | None) -> float | None:
    value = parse_number(num)
    if value is None:
        return None
    return value * MULTIPLIERS.get((mult or "").lower(), 1.0)


class QueryParser:
    def __init__(self, vocab: Vocabulary | None = None) -> None:
        self.vocab = vocab or Vocabulary()

    # --------------------------------------------------------------- pieces
    @staticmethod
    def _bedrooms(s: str, f: QueryFilters) -> str:
        def sub(pattern: str, fn) -> None:
            nonlocal s
            match = re.search(pattern, s)
            if match:
                fn(match)
                s = s[: match.start()] + " " + s[match.end():]

        if re.search(rf"(\d+)\s*(?:\+|or more|and above|plus)\s*-?\s*{_BED}\b", s):
            sub(rf"(\d+)\s*(?:\+|or more|and above|plus)\s*-?\s*{_BED}\b", lambda m: setattr(f, "min_bedrooms", int(m.group(1))))
        elif re.search(rf"(?:at least|minimum|min\.?)\s*(\d+)\s*-?\s*{_BED}\b", s):
            sub(rf"(?:at least|minimum|min\.?)\s*(\d+)\s*-?\s*{_BED}\b", lambda m: setattr(f, "min_bedrooms", int(m.group(1))))
        elif re.search(rf"(?:more than|over)\s*(\d+)\s*-?\s*{_BED}\b", s):
            sub(rf"(?:more than|over)\s*(\d+)\s*-?\s*{_BED}\b", lambda m: setattr(f, "min_bedrooms", int(m.group(1)) + 1))
        elif re.search(rf"(?:up to|max(?:imum)?|at most)\s*(\d+)\s*-?\s*{_BED}\b", s):
            sub(rf"(?:up to|max(?:imum)?|at most)\s*(\d+)\s*-?\s*{_BED}\b", lambda m: setattr(f, "max_bedrooms", int(m.group(1))))
        elif re.search(rf"(\d+)\s*(?:-|to|or)\s*(\d+)\s*-?\s*{_BED}\b", s):
            def rng(m: re.Match[str]) -> None:
                lo, hi = sorted((int(m.group(1)), int(m.group(2))))
                f.min_bedrooms, f.max_bedrooms = lo, hi
            sub(rf"(\d+)\s*(?:-|to|or)\s*(\d+)\s*-?\s*{_BED}\b", rng)
        elif re.search(rf"\b(\d{{1,2}})\s*-?\s*{_BED}\b", s):
            sub(rf"\b(\d{{1,2}})\s*-?\s*{_BED}\b", lambda m: setattr(f, "bedrooms", int(m.group(1))))
        elif re.search(rf"\b({'|'.join(_WORD_NUMS)})\s*-?\s*{_BED}\b", s):
            sub(rf"\b({'|'.join(_WORD_NUMS)})\s*-?\s*{_BED}\b", lambda m: setattr(f, "bedrooms", _WORD_NUMS[m.group(1)]))
        # Arabic
        elif re.search(r"(\d{1,2})\s*(?:غرف|غرفه)", s):
            sub(r"(\d{1,2})\s*(?:غرف|غرفه)(?:\s*نوم)?", lambda m: setattr(f, "bedrooms", int(m.group(1))))
        elif re.search(r"غرفتين|غرفتان", s):
            sub(r"غرفتين|غرفتان", lambda m: setattr(f, "bedrooms", 2))
        elif re.search(rf"({'|'.join(_AR_WORD_NUMS)})\s*(?:غرف|غرفه)", s):
            sub(rf"({'|'.join(_AR_WORD_NUMS)})\s*(?:غرف|غرفه)(?:\s*نوم)?", lambda m: setattr(f, "bedrooms", _AR_WORD_NUMS[m.group(1)]))
        elif re.search(r"غرفه\s*واحده", s):
            sub(r"غرفه\s*واحده", lambda m: setattr(f, "bedrooms", 1))
        return s

    @staticmethod
    def _area(s: str, f: QueryFilters) -> str:
        def to_sqm(value: float | None, unit: str) -> float | None:
            if value is None:
                return None
            return round(value * 0.09290304, 1) if re.search(r"ft|feet", unit) else value

        patterns = [
            (rf"(?:over|above|more than|at least|min(?:imum)?|bigger than|larger than|اكثر من|اكبر من)\s*(\d+(?:[.,]\d+)?)\s*({_AREA_UNIT})", "min"),
            (rf"(?:under|below|less than|up to|max(?:imum)?|smaller than|اقل من|اصغر من)\s*(\d+(?:[.,]\d+)?)\s*({_AREA_UNIT})", "max"),
            (rf"(\d+(?:[.,]\d+)?)\s*(?:-|to)\s*(\d+(?:[.,]\d+)?)\s*({_AREA_UNIT})", "range"),
            (rf"(\d+(?:[.,]\d+)?)\s*({_AREA_UNIT})", "approx"),
        ]
        for pattern, mode in patterns:
            match = re.search(pattern, s)
            if not match:
                continue
            if mode == "range":
                unit = match.group(3)
                lo, hi = to_sqm(parse_number(match.group(1)), unit), to_sqm(parse_number(match.group(2)), unit)
                f.min_area, f.max_area = (min(lo, hi), max(lo, hi)) if lo and hi else (lo, hi)
            else:
                value = to_sqm(parse_number(match.group(1)), match.group(2))
                if mode == "min":
                    f.min_area = value
                elif mode == "max":
                    f.max_area = value
                elif value:
                    f.min_area, f.max_area = round(value * 0.8), round(value * 1.2)
            s = s[: match.start()] + " " + s[match.end():]
            break
        return s

    @staticmethod
    def _price(s: str, f: QueryFilters) -> str:
        cur = rf"\s*(?:{_CUR})?\s*"
        found = False

        def apply(match: re.Match[str], mode: str) -> None:
            nonlocal found
            groups = [g for g in match.groups()]
            if mode == "between":
                a, b = _amount(groups[0], groups[1]), _amount(groups[2], groups[3])
                if a and b:
                    f.min_price, f.max_price = min(a, b), max(a, b)
                    found = True
                return
            value = _amount(groups[0], groups[1])
            if not value or value < 500:
                return
            if not groups[1] and 1900 <= value <= 2100:
                return  # a year, not a price
            found = True
            if mode == "max":
                f.max_price = value
            elif mode == "min":
                f.min_price = value
            elif mode == "around":
                f.min_price, f.max_price = value * 0.85, value * 1.15
            elif mode == "bare":
                f.max_price = value * 1.1

        patterns = [
            (rf"(?:between|from|بين|من){cur}{_NUM}{cur}(?:and|to|-|و|الى){cur}{_NUM}", "between"),
            (rf"(?:{_MAX_WORDS})\s*(?:of\s*)?{cur}{_NUM}", "max"),
            (rf"(?:{_AROUND_WORDS}){cur}{_NUM}", "around"),
            (rf"(?:{_MIN_WORDS}){cur}{_NUM}", "min"),
            (rf"{_CUR}\s*{_NUM}", "bare"),
            (rf"\b{_NUM_MULT}", "bare"),  # "500k", "2m sar", "2 مليون"
            (rf"\b(\d{{4,}}(?:[.,]\d+)*)()\s*{_CUR}", "bare"),
        ]
        for pattern, mode in patterns:
            match = re.search(pattern, s)
            if match:
                apply(match, mode)
                if found:
                    s = s[: match.start()] + " " + s[match.end():]
                    break
        if found:
            f.currency = detect_currency(s) or f.currency
        if re.search(r"\b(?:per month|a month|monthly|/\s*month|/mo)\b|شهري|بالشهر|في الشهر", s):
            if f.max_price:
                f.max_price *= 12
            if f.min_price:
                f.min_price *= 12
        return s

    def _projects(self, norm: str, f: QueryFilters) -> None:
        padded = f" {norm} "
        for alias, record_id, _title in self.vocab.projects:
            if record_id in f.project_ids:
                continue
            if f" {alias} " in padded or (len(alias) >= 9 and alias.count(" ") >= 1 and fuzz.partial_ratio(alias, norm) >= 92):
                f.project_ids.append(record_id)

    def _districts(self, norm: str, f: QueryFilters) -> None:
        if not self.vocab.districts:
            return
        # Some districts carry a city's name: "Al-Riyadh" is a district of Jeddah,
        # "Al Khobar" one of Madinah, "Al Jubail" one of Jubail. When the words were
        # already read as a city, they mean that city, unless the district's own
        # city is named as well ("Al-Riyadh district, Jeddah").
        city_words = {key: city for city in f.cities for key in city_alias_keys(city)}
        tokens = norm.split()
        found: list[str] = []
        for size in (4, 3, 2, 1):
            for i in range(len(tokens) - size + 1):
                key = place_key(" ".join(tokens[i : i + size]))
                if len(key) < 4:
                    continue
                hit = self.vocab.districts.get(key)
                if not hit or hit[0] in found:
                    continue
                name, district_city = hit[0], canonical_city(hit[1])
                named_city = city_words.get(key)
                if named_city:
                    others = [c for c in f.cities if c != named_city]
                    if not district_city or district_city not in others:
                        continue  # "Riyadh" means the city of Riyadh
                    f.cities.remove(named_city)  # the words named the district, not a second city
                found.append(name)
                if district_city and district_city not in f.cities:
                    f.cities.append(district_city)
        f.districts = found[:3]

    # ------------------------------------------------------------------ main
    def parse(self, text: str, history: list[str] | None = None) -> ParsedQuery:
        raw = (text or "").strip()
        s = _prep(raw)
        f = QueryFilters()
        language = "ar" if is_arabic(raw) else "en"

        s = self._bedrooms(s, f)
        s = self._area(s, f)
        s = self._price(s, f)

        limit = _LIMIT.search(s)
        if limit:
            value = int(limit.group(1) or limit.group(2))
            if 1 <= value <= 20:
                f.limit = value

        rent, sale = bool(_RENT.search(s)), bool(_SALE.search(s))
        if rent != sale:
            f.purpose = "rent" if rent else "sale"

        norm = normalize_for_search(s)
        padded = f" {norm} "
        for canonical, aliases in PROPERTY_TYPES:
            for alias in aliases:
                a = normalize_for_search(alias)
                if not re.search(rf"(?<=\s)(?:[وبلف]?ال)?{re.escape(a)}(?=\s)", padded):
                    continue
                if canonical == "floor" and re.search(rf"(?:\d+(?:st|nd|rd|th)?|ground|top|first|second|third|upper|lower)\s+{a}\b", norm):
                    continue
                if canonical not in f.property_types:
                    f.property_types.append(canonical)
                break
        if "studio" in f.property_types and "apartment" not in f.property_types:
            f.property_types.append("apartment")

        if _COMMERCIAL.search(s):
            f.category = "commercial"
        elif _RESIDENTIAL.search(s):
            f.category = "residential"

        if _UNFURNISHED.search(s):
            f.furnishing = "unfurnished"
        elif _SEMI.search(s):
            f.furnishing = "semi-furnished"
        elif _FURNISHED.search(s):
            f.furnishing = "furnished"

        for name, pattern in _SORT_PATTERNS:
            if pattern.search(s):
                f.sort = name
                break

        if _DARGLOBAL.search(s):
            f.sources.append("darglobal")
        if _WASALT.search(s):
            f.sources.append("wasalt")
        if _PROJECT_WORDS.search(s) and "darglobal" not in f.sources:
            f.kinds = ["project"]

        cities, countries = find_places(raw)
        f.cities = [c for c in cities]
        f.countries = [c for c in countries if not any(city_country(x) == c for x in cities)]
        self._districts(norm, f)
        self._projects(normalize_for_search(raw), f)

        if (f.min_price or f.max_price) and not f.currency:
            country = (f.countries or [city_country(f.cities[0]) if f.cities else None])[0]
            f.currency = COUNTRY_CURRENCY.get(country or "", "SAR")

        intent = self._intent(s, f)
        follow_up, references = False, False
        ordinal = None
        for word, n in _ORDINALS.items():
            if re.search(rf"(?:^|\s){word}(?:\s|$)", s):
                ordinal = n
                break
        if history:
            words = norm.split()
            references = bool(_PRONOUNS.search(s)) or ordinal is not None
            short_bare = len(words) <= 4 and not f.is_structured() and not _GENERAL_Q.search(s)
            follow_up = len(words) <= 9 and (bool(_FOLLOW_START.search(s.strip())) or references or short_bare)
            if intent in ("greeting", "thanks", "help"):
                follow_up = False

        parsed = ParsedQuery(
            text=raw,
            language=language,
            intent=intent,
            filters=f,
            is_follow_up=follow_up,
            references_previous=references,
            ordinal=ordinal,
            search_text=raw,
        )
        if follow_up and history:
            parsed = self._merge_history(parsed, history)
        return parsed

    def _intent(self, s: str, f: QueryFilters) -> str:
        stripped = s.strip()
        if _GREETING.fullmatch(stripped):
            return "greeting"
        if _THANKS.fullmatch(stripped):
            return "thanks"
        if _HELP.search(stripped):
            return "help"
        if _STATS.search(stripped):
            return "stats"
        if f.project_ids and not (f.property_types or f.purpose or f.bedrooms or f.max_price or f.min_price):
            return "project_detail"
        strong = bool(
            f.property_types or f.bedrooms is not None or f.min_bedrooms or f.max_bedrooms
            or f.max_price or f.min_price or f.min_area or f.max_area or f.furnishing or f.districts
        )
        place_signals = bool(f.cities or f.districts or f.countries)
        if (f.kinds or f.sources) and place_signals:
            return "property_search"
        if f.sources and _SEARCH_VERBS.search(s):
            return "property_search"
        if _GENERAL_Q.search(stripped) and not strong:
            return "general"
        if strong or f.purpose or (place_signals and _SEARCH_VERBS.search(s)):
            return "property_search"
        if f.kinds and (f.sources or _SEARCH_VERBS.search(s)):
            return "property_search"
        return "general"

    def _merge_history(self, parsed: ParsedQuery, history: list[str]) -> ParsedQuery:
        """Carry structured context from earlier user turns into a follow-up."""
        base = QueryFilters()
        previous_text, previous_intent = "", "general"
        for prev in history[-3:]:
            prev_parsed = QueryParser(self.vocab).parse(prev)
            if prev_parsed.intent in ("greeting", "thanks", "help"):
                continue
            base = base.merged_with(prev_parsed.filters)
            previous_text, previous_intent = prev, prev_parsed.intent
        parsed.filters = base.merged_with(parsed.filters)
        if parsed.intent == "general" and previous_intent in ("property_search", "project_detail", "stats"):
            parsed.intent = previous_intent
        parsed.search_text = f"{previous_text} {parsed.text}".strip()
        return parsed


def default_currency_for(filters: QueryFilters) -> str:
    if filters.currency:
        return filters.currency
    country = (filters.countries or [city_country(filters.cities[0]) if filters.cities else None])[0]
    return COUNTRY_CURRENCY.get(country or "", "SAR")
