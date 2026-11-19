"""Domain vocabulary: property types, places and money parsing (EN + AR).

Shared by the scrapers (to normalise source data) and by the query parser
(to understand user questions), so both sides speak the same canonical terms.
"""

from __future__ import annotations

import re

from app.textutil import normalize_for_search, place_key, to_ascii_digits

# --------------------------------------------------------------------------- types
# Order matters: more specific types first.
PROPERTY_TYPES: list[tuple[str, list[str]]] = [
    ("penthouse", ["penthouse", "penthouses", "بنتهاوس", "بنت هاوس"]),
    ("townhouse", ["townhouse", "townhouses", "town house", "town houses", "تاون هاوس"]),
    ("duplex", ["duplex", "duplexes", "دوبلكس", "دبلكس"]),
    ("studio", ["studio", "studios", "استوديو", "ستوديو"]),
    ("hotel_room", ["hotel room", "hotel rooms", "hotel apartment", "hotel apartments", "serviced apartment", "serviced apartments", "غرفه فندقيه"]),
    ("apartment", ["apartment", "apartments", "flat", "flats", "apt", "condo", "condos", "شقه", "شقق"]),
    ("mansion", ["mansion", "mansions", "قصر", "قصور"]),
    ("villa", ["villa", "villas", "فيلا", "فلل", "فله", "فيلات"]),
    ("rest_house", ["rest house", "rest houses", "resthouse", "rest-house", "istiraha", "estraha", "استراحه", "استراحات"]),
    ("chalet", ["chalet", "chalets", "شاليه", "شاليهات"]),
    ("farm", ["farm", "farms", "مزرعه", "مزارع"]),
    ("land", ["land", "lands", "plot", "plots", "ارض", "اراضي", "قطعه ارض"]),
    ("building", ["building", "buildings", "عماره", "عمائر", "مبنى", "مباني", "بنايه"]),
    ("floor", ["floor", "floors", "دور", "ادوار", "طابق"]),
    ("office", ["office", "offices", "مكتب", "مكاتب"]),
    ("showroom", ["showroom", "showrooms", "معرض", "معارض"]),
    ("shop", ["shop", "shops", "retail", "store", "stores", "محل", "محلات"]),
    ("warehouse", ["warehouse", "warehouses", "storage", "مستودع", "مستودعات"]),
]

TYPE_LABELS = {
    "penthouse": "Penthouse", "townhouse": "Townhouse", "duplex": "Duplex", "studio": "Studio",
    "hotel_room": "Hotel room", "apartment": "Apartment", "mansion": "Mansion", "villa": "Villa",
    "rest_house": "Rest house", "chalet": "Chalet", "farm": "Farm", "land": "Land",
    "building": "Building", "floor": "Floor", "office": "Office", "showroom": "Showroom",
    "shop": "Shop", "warehouse": "Warehouse",
}

COMMERCIAL_TYPES = {"office", "showroom", "shop", "warehouse"}


def canonical_property_type(raw: str | None) -> str | None:
    """Map a source label/slug ("residential-land", "Villas", "شقة") to a canonical type."""
    if not raw:
        return None
    text = " " + normalize_for_search(str(raw).replace("-", " ").replace("_", " ")) + " "
    for canonical, aliases in PROPERTY_TYPES:
        for alias in aliases:
            if f" {normalize_for_search(alias)} " in text:
                return canonical
    return None


def property_types_in(text: str | None) -> list[str]:
    """All canonical types mentioned in a free-text label, e.g. 'Apartments & Penthouses'."""
    if not text:
        return []
    norm = " " + normalize_for_search(text.replace("-", " ")) + " "
    found: list[str] = []
    for canonical, aliases in PROPERTY_TYPES:
        if any(f" {normalize_for_search(a)} " in norm for a in aliases) and canonical not in found:
            found.append(canonical)
    return found


# -------------------------------------------------------------------------- places
COUNTRIES: dict[str, list[str]] = {
    "Saudi Arabia": ["saudi arabia", "saudi", "ksa", "kingdom of saudi arabia", "السعوديه", "المملكه العربيه السعوديه"],
    "United Arab Emirates": ["united arab emirates", "uae", "emirates", "الامارات", "دوله الامارات"],
    "Oman": ["oman", "sultanate of oman", "سلطنه عمان"],
    "Qatar": ["qatar", "قطر"],
    "United Kingdom": ["united kingdom", "uk", "england", "britain", "great britain", "بريطانيا", "المملكه المتحده"],
    "Spain": ["spain", "اسبانيا"],
    "Maldives": ["maldives", "المالديف", "جزر المالديف"],
    "Greece": ["greece", "اليونان"],
    "Bosnia and Herzegovina": ["bosnia", "bosnia and herzegovina", "البوسنه"],
    "Bahrain": ["bahrain", "البحرين"],
    "United States": ["united states", "usa", "america", "امريكا"],
}

COUNTRY_CURRENCY = {
    "Saudi Arabia": "SAR", "United Arab Emirates": "AED", "Oman": "OMR", "Qatar": "QAR",
    "United Kingdom": "GBP", "Spain": "EUR", "Greece": "EUR", "Maldives": "USD",
    "Bahrain": "BHD", "United States": "USD",
}

# canonical city -> (country, aliases)
CITIES: dict[str, tuple[str, list[str]]] = {
    "Riyadh": ("Saudi Arabia", ["riyadh", "ar riyadh", "al riyadh", "riyad", "ryadh", "الرياض", "رياض"]),
    "Jeddah": ("Saudi Arabia", ["jeddah", "jedda", "jiddah", "jidda", "jedah", "جده"]),
    "Makkah": ("Saudi Arabia", ["makkah", "mecca", "makka", "makkah al mukarramah", "مكه", "مكه المكرمه"]),
    "Madinah": ("Saudi Arabia", ["madinah", "medina", "madina", "al madinah", "al madinah al munawwarah", "المدينه", "المدينه المنوره"]),
    # Wasalt spells many city names its own way ("Aldammam", "Alttayif", "Bariduh",
    # "Hayil", "Tbwk", "Earear", "Eanizah", "Alzahran"); those spellings are aliases
    # here, so stored listings and users' questions meet on the same canonical name.
    "Dammam": ("Saudi Arabia", ["dammam", "ad dammam", "al dammam", "aldammam", "الدمام"]),
    "Al Khobar": ("Saudi Arabia", ["al khobar", "khobar", "al-khobar", "alkhobar", "الخبر"]),
    "Dhahran": ("Saudi Arabia", ["dhahran", "alzahran", "al zahran", "aldhahran", "الظهران"]),
    "Taif": ("Saudi Arabia", ["taif", "at taif", "al taif", "alttayif", "altaif", "tayif", "الطائف"]),
    "Abha": ("Saudi Arabia", ["abha", "ابها"]),
    "Khamis Mushait": ("Saudi Arabia", ["khamis mushait", "khamis mushayt", "خميس مشيط"]),
    "Buraidah": ("Saudi Arabia", ["buraidah", "buraydah", "buraida", "burayda", "bariduh", "بريده"]),
    "Tabuk": ("Saudi Arabia", ["tabuk", "tbwk", "تبوك"]),
    "Hail": ("Saudi Arabia", ["hail", "ha il", "hayil", "hayel", "حائل"]),
    "Jazan": ("Saudi Arabia", ["jazan", "jizan", "gizan", "جازان", "جيزان"]),
    "Najran": ("Saudi Arabia", ["najran", "نجران"]),
    "Al Ahsa": ("Saudi Arabia", ["al ahsa", "al hasa", "ahsa", "hofuf", "al hofuf", "الاحساء", "الهفوف"]),
    "Jubail": ("Saudi Arabia", ["jubail", "al jubail", "الجبيل"]),
    "Yanbu": ("Saudi Arabia", ["yanbu", "ينبع"]),
    "Al Kharj": ("Saudi Arabia", ["al kharj", "kharj", "الخرج"]),
    "Qatif": ("Saudi Arabia", ["qatif", "al qatif", "القطيف"]),
    "Unaizah": ("Saudi Arabia", ["unaizah", "onaizah", "unayzah", "eanizah", "عنيزه"]),
    "Arar": ("Saudi Arabia", ["arar", "earear", "عرعر"]),
    "Abqaiq": ("Saudi Arabia", ["abqaiq", "buqayq", "بقيق", "ابقيق"]),
    "Hafar Al Batin": ("Saudi Arabia", ["hafar al batin", "hafr al batin", "hafr albatin", "حفر الباطن"]),
    "Al Khafji": ("Saudi Arabia", ["al khafji", "khafji", "alkhafji", "الخفجي"]),
    "Rabigh": ("Saudi Arabia", ["rabigh", "رابغ"]),
    "Al Zulfi": ("Saudi Arabia", ["al zulfi", "zulfi", "alzulfi", "الزلفي"]),
    "Ar Rass": ("Saudi Arabia", ["ar rass", "al rass", "alrass", "الرس"]),
    "Al Muzahimiyah": ("Saudi Arabia", ["al muzahimiyah", "muzahimiyah", "almuzahimih", "المزاحميه"]),
    "Turabah": ("Saudi Arabia", ["turabah", "turba", "تربه"]),
    "Hotat Bani Tamim": ("Saudi Arabia", ["hotat bani tamim", "hawtat bani tamim", "حوطه بني تميم"]),
    "King Abdullah Economic City": ("Saudi Arabia", ["king abdullah economic city", "kaec"]),
    "AlUla": ("Saudi Arabia", ["alula", "al ula", "العلا"]),
    "Diriyah": ("Saudi Arabia", ["diriyah", "ad diriyah", "al diriyah", "الدرعيه"]),
    "Wadi Safar": ("Saudi Arabia", ["wadi safar", "وادي صفار"]),
    "Dubai": ("United Arab Emirates", ["dubai", "دبي"]),
    "Abu Dhabi": ("United Arab Emirates", ["abu dhabi", "ابوظبي", "ابو ظبي"]),
    "Ras Al Khaimah": ("United Arab Emirates", ["ras al khaimah", "rak", "راس الخيمه"]),
    "Muscat": ("Oman", ["muscat", "مسقط"]),
    "Sohar": ("Oman", ["sohar", "صحار"]),
    "Doha": ("Qatar", ["doha", "الدوحه"]),
    "Lusail": ("Qatar", ["lusail", "لوسيل"]),
    "Simaisma": ("Qatar", ["simaisma", "سميسمه"]),
    "London": ("United Kingdom", ["london", "لندن"]),
    "Marbella": ("Spain", ["marbella", "ماربيا"]),
    "Estepona": ("Spain", ["estepona"]),
    "Malaga": ("Spain", ["malaga", "málaga", "مالقا"]),
}


def _alias_index() -> list[tuple[str, str, str]]:
    """(normalised alias, canonical, kind) sorted longest-first for greedy matching."""
    rows: list[tuple[str, str, str]] = []
    for city, (_country, aliases) in CITIES.items():
        for alias in {city, *aliases}:
            rows.append((normalize_for_search(alias), city, "city"))
    for country, aliases in COUNTRIES.items():
        for alias in {country, *aliases}:
            rows.append((normalize_for_search(alias), country, "country"))
    rows.sort(key=lambda r: len(r[0]), reverse=True)
    return rows


_ALIASES = _alias_index()


def find_places(text: str | None) -> tuple[list[str], list[str]]:
    """Return (cities, countries) mentioned in text, in order of appearance."""
    if not text:
        return [], []
    norm = " " + normalize_for_search(text) + " "
    hits: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for alias, canonical, kind in _ALIASES:
        if not alias:
            continue
        for match in re.finditer(rf"(?<=[\s])(?:[وبلف]?)(?:{re.escape(alias)})(?=[\s])", norm):
            start, end = match.span()
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            hits.append((start, canonical, kind))
    hits.sort()
    cities: list[str] = []
    countries: list[str] = []
    for _, canonical, kind in hits:
        bucket = cities if kind == "city" else countries
        if canonical not in bucket:
            bucket.append(canonical)
    return cities, countries


_CITY_BY_ALIAS: dict[str, str] = {alias: canonical for alias, canonical, kind in _ALIASES if kind == "city"}


def canonical_city(name: str | None) -> str | None:
    """Standard spelling of a city name ("Aldammam" -> "Dammam", "Khobar" -> "Al Khobar").

    Only a whole-name match counts, so a longer place name that merely contains a
    city ("King Fahd Road, Riyadh") is never collapsed; unknown names come back unchanged.
    """
    if not name:
        return name
    return _CITY_BY_ALIAS.get(normalize_for_search(name.replace("-", " ")), name)


def city_alias_keys(city: str) -> set[str]:
    """Every ``place_key`` a canonical city answers to, e.g. {"riyadh", "riyad", "رياض", ...}."""
    _country, aliases = CITIES.get(city, ("", []))
    return {key for key in (place_key(a) for a in (city, *aliases)) if key}


def parse_location(text: str | None) -> tuple[str | None, str | None]:
    cities, countries = find_places(text)
    city = cities[0] if cities else None
    country = countries[0] if countries else (CITIES[city][0] if city else None)
    return city, country


def city_country(city: str | None) -> str | None:
    if not city:
        return None
    entry = CITIES.get(city)
    return entry[0] if entry else None


# --------------------------------------------------------------------------- money
CURRENCY_ALIASES: dict[str, list[str]] = {
    "SAR": ["sar", "sr", "riyal", "riyals", "rial", "ريال", "ر س"],
    "AED": ["aed", "dirham", "dirhams", "درهم"],
    "QAR": ["qar", "qr", "qatari riyal", "ريال قطري"],
    "OMR": ["omr", "omani rial", "ريال عماني"],
    "USD": ["usd", "$", "dollar", "dollars", "us dollar", "دولار"],
    "GBP": ["gbp", "£", "pound", "pounds", "sterling", "جنيه"],
    "EUR": ["eur", "€", "euro", "euros", "يورو"],
}
_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}

MULTIPLIERS: dict[str, float] = {
    "k": 1e3, "thousand": 1e3, "الف": 1e3, "الاف": 1e3, "آلاف": 1e3,
    "m": 1e6, "mn": 1e6, "mil": 1e6, "million": 1e6, "millions": 1e6, "مليون": 1e6, "ملايين": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "مليار": 1e9,
}


def detect_currency(text: str | None) -> str | None:
    if not text:
        return None
    for symbol, code in _SYMBOLS.items():
        if symbol in text:
            return code
    norm = " " + normalize_for_search(text) + " "
    for code, aliases in CURRENCY_ALIASES.items():
        for alias in aliases:
            a = normalize_for_search(alias)
            if a and f" {a} " in norm:
                return code
    return None


_MONEY = re.compile(
    r"(\d+(?:[.,]\d+)*)\s*(k|thousand|m|mn|mil|million|millions|b|bn|billion|الف|ألف|آلاف|الاف|مليون|ملايين|مليار)?(?![a-z])",
    re.I,
)


def parse_number(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    # "1,200,000" / "1.200.000" -> thousands separators; "2.5" -> decimal
    if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+", raw):
        return float(re.sub(r"[,.]", "", raw))
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def parse_money(text: str | None) -> tuple[float | None, str | None]:
    """'Starting from AED 2.5M' -> (2_500_000.0, 'AED')."""
    if not text:
        return None, None
    t = to_ascii_digits(str(text))
    currency = detect_currency(t)
    for match in _MONEY.finditer(t):
        value = parse_number(match.group(1))
        if value is None:
            continue
        suffix = (match.group(2) or "").lower()
        if not suffix and 1900 <= value <= 2100 and float(value).is_integer():
            continue  # a year ("Q4 2029"), not a price
        amount = value * MULTIPLIERS.get(suffix, 1.0)
        if amount >= 1000:  # ignore stray small numbers (floors, bedrooms...)
            return amount, currency
    return None, currency


_AREA = re.compile(r"(\d+(?:[.,]\d+)?)(?:\s*(?:to|-|–|and)\s*(\d+(?:[.,]\d+)?))?\s*(sq\.?\s*ft|sqft|square feet|ft2)?", re.I)
SQFT_TO_SQM = 0.09290304


def parse_area_range(text: str | None) -> tuple[float | None, float | None]:
    """'70 to 365' -> (70, 365); '721 sq.ft' -> (67.0, None)."""
    if not text:
        return None, None
    match = _AREA.search(to_ascii_digits(str(text)))
    if not match:
        return None, None
    lo = parse_number(match.group(1))
    hi = parse_number(match.group(2)) if match.group(2) else None
    if match.group(3):
        lo = round(lo * SQFT_TO_SQM, 1) if lo else lo
        hi = round(hi * SQFT_TO_SQM, 1) if hi else hi
    return lo, hi


BRANDS = [
    "Trump", "Missoni", "Pagani", "Elie Saab", "W Residences", "Marriott", "Mouawad", "Fendi",
    "Aston Martin", "Bentley", "Mulliner", "Lamborghini", "Nickelodeon", "Dolce&Gabbana",
    "Versace", "Armani", "Padel",
]


def detect_brands(text: str | None) -> list[str]:
    if not text:
        return []
    low = text.lower()
    return [b for b in BRANDS if b.lower() in low]


def place_variants(name: str) -> set[str]:
    return {place_key(name), normalize_for_search(name)}
