"""Wasalt (wasalt.sa) adapter.

Compliance (robots.txt + https://wasalt.sa/llms.txt):
* ``/search`` is disallowed, so we never call it. Listings are discovered via
  the public category landing pages listed in the official category sitemap;
  each page server-renders 32 structured listings in ``__NEXT_DATA__``.
* "Moderate" crawl rate: one request at a time with a >=1 s interval.
* No contact details: owner/agent/lead/contact structures are never read,
  and free text is PII-scrubbed.
* Attribution: every Wasalt record links back to its live page and the bot
  cites "Source: Wasalt.sa".
"""

from __future__ import annotations

import hashlib
import html as htmllib
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from app.config import Settings
from app.models import CrawlReport, Record, utcnow
from app.scraping.http import (
    FetchError,
    HostBlocked,
    NotFound,
    PoliteClient,
    RobotsDisallowed,
    gather_isolated,
)
from app.scraping.nextdata import extract_next_data, html_to_text, page_props
from app.scraping.pii import scrub_pii
from app.textutil import clean_ws, truncate
from app.vocab import COMMERCIAL_TYPES, TYPE_LABELS, canonical_city, canonical_property_type, find_places

log = logging.getLogger(__name__)

BASE = "https://wasalt.sa"
CATEGORY_SITEMAP = "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz"
BLOG_API = "https://blog.wasalt.sa/en/wp-json/wp/v2"

CATEGORY_RE = re.compile(r"^/en/(?P<type>[a-z0-9-]+?)-for-(?P<purpose>sale|rent)-in-(?P<loc>[a-z0-9-]+)$")
TYPE_SLUGS = [
    "properties", "apartments", "villas", "lands", "floors", "buildings", "duplexes",
    "commercial-properties", "furnished-apartments", "showrooms", "rest-houses", "chalets",
    "farms", "commercial-lands", "1-bedroom-apartments", "2-bedroom-apartments",
    "3-bedroom-apartments", "4-bedroom-apartments", "5-bedroom-apartments",
]
PRIORITY_CITIES = [
    "riyadh", "jeddah", "dammam", "al-khobar", "makkah", "madinah", "taif", "abha",
    "khamis-mushait", "dhahran", "buraidah", "tabuk", "hail", "jazan", "najran", "al-ahsa",
    "jubail", "yanbu", "al-kharj", "qatif", "unaizah",
]
TOP_CITIES = 6
COUNTRY_TYPES = ["properties", "apartments", "villas", "lands", "commercial-properties"]
TIER1_TYPES = ["properties", "apartments", "villas", "lands", "floors", "commercial-properties"]
TIER2_TYPES = ["properties", "apartments", "villas"]
EXTRA_TYPES = ["buildings", "duplexes", "furnished-apartments", "showrooms", "rest-houses"]

# Labels from listing detail pages that may identify people/licences: never keep.
_SENSITIVE_LABEL = re.compile(
    r"licen|rega|advertis|responsible|owner|agent|broker|number|phone|mobile|contact|whats|"
    r"\bid\b|deed|instrument|plan no|parcel|source|authori",
    re.I,
)


def category_url(type_slug: str, purpose: str, loc: str) -> str:
    return f"{BASE}/en/{type_slug}-for-{purpose}-in-{loc}"


def plan_categories(paths: list[str], budget: int) -> list[str]:
    """Choose a diverse, prioritised set of category pages within ``budget``.

    Cities are interleaved inside each type so every city gets its most
    important categories before any city gets long-tail ones. Order:
    country-level core types -> core types for the six largest cities ->
    main types for other cities -> extra types for top cities ->
    bedroom-specific rentals -> district pages of the top cities.
    """
    parsed: dict[tuple[str, str, str], str] = {}
    for path in paths:
        match = CATEGORY_RE.match(path)
        if match and match.group("type") in TYPE_SLUGS:
            parsed[(match.group("type"), match.group("purpose"), match.group("loc"))] = BASE + path
    if not parsed:
        return []
    locs = {loc for (_, _, loc) in parsed}
    city_locs = [c for c in PRIORITY_CITIES if c in locs]
    # Any other location that prefixes district pages is also a city.
    others = sorted(
        {loc for loc in locs if loc not in city_locs and loc != "saudi-arabia" and any(m.startswith(loc + "-") for m in locs)},
        key=lambda loc: -sum(1 for m in locs if m.startswith(loc + "-")),
    )
    city_locs += others

    plan: list[str] = []

    def take(type_slug: str, purpose: str, loc: str) -> None:
        url = parsed.get((type_slug, purpose, loc))
        if url and url not in plan and len(plan) < budget:
            plan.append(url)

    top, rest = city_locs[:TOP_CITIES], city_locs[TOP_CITIES:]
    for type_slug in COUNTRY_TYPES:
        for purpose in ("sale", "rent"):
            take(type_slug, purpose, "saudi-arabia")
    for types, locs in ((TIER1_TYPES, top), (TIER2_TYPES, rest), (EXTRA_TYPES, top)):
        for type_slug in types:
            for purpose in ("sale", "rent"):
                for loc in locs:
                    take(type_slug, purpose, loc)
    for beds in range(1, 6):
        for loc in top:
            take(f"{beds}-bedroom-apartments", "rent", loc)
    # District pages of the top cities, round-robin for geographic spread.
    district_lists = [
        sorted(loc for loc in locs if loc.startswith(city + "-")) for city in city_locs[:TOP_CITIES]
    ]
    while len(plan) < budget and any(district_lists):
        for districts in district_lists:
            if districts:
                loc = districts.pop(0)
                take("properties", "sale", loc)
                take("properties", "rent", loc)
    return plan


FALLBACK_PLAN = [
    category_url(t, p, loc)
    for loc in ("saudi-arabia", "riyadh", "jeddah", "dammam", "al-khobar", "makkah", "madinah")
    for t in ("properties", "apartments", "villas", "lands")
    for p in ("sale", "rent")
]


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _rent(info: dict[str, Any]) -> tuple[float | None, str | None]:
    freq = info.get("rentFreq")
    amount, period = None, None
    if isinstance(freq, dict) and freq:
        chosen = next((v for v in freq.values() if isinstance(v, dict) and v.get("default_freq")), None)
        key = next((k for k, v in freq.items() if v is chosen), None) if chosen else next(iter(freq))
        chosen = chosen or freq.get(key)
        if isinstance(chosen, dict):
            amount = _to_float(chosen.get("amount"))
            period = {"yearly": "year", "monthly": "month", "daily": "day", "weekly": "week"}.get(
                str(chosen.get("freq") or key), str(chosen.get("freq") or key)
            )
    expected = _to_float(info.get("expectedRent"))
    return expected or amount, period or ("year" if expected else None)


def _title(subtype: str | None, purpose: str, district: str | None, city: str | None, raw: str | None) -> str:
    place = ", ".join(x for x in (district, city) if x)
    base = subtype or "Property"
    raw = clean_ws(raw or "")
    if raw and len(raw.split()) > 2 and not re.search(r"[؀-ۿ]", raw):
        return f"{raw} — {place}" if place else raw
    return f"{base} for {'sale' if purpose == 'sale' else 'rent'} in {place}" if place else f"{base} for {purpose}"


def parse_listing(item: dict[str, Any]) -> Record | None:
    """Normalise one listing from a category page. Never touches contact/owner data."""
    info = item.get("propertyInfo") or {}
    pid, slug, purpose = item.get("id"), info.get("slug"), info.get("propertyFor")
    if not pid or not slug or purpose not in ("sale", "rent"):
        return None
    attrs = {
        a.get("key"): a.get("value")
        for a in item.get("attributes") or []
        if isinstance(a, dict) and a.get("key")
    }
    if purpose == "sale":
        price, period = _to_float(info.get("salePrice")) or _to_float(info.get("conversionPrice")), None
    else:
        price, period = _rent(info)
    subtype = info.get("propertySubType")
    # The human label decides: Wasalt files villas under the slug "villa-townhouse",
    # so reading the slug first would turn every villa into a townhouse.
    ptype = canonical_property_type(subtype) or canonical_property_type(info.get("propertySubTypeSlug"))
    main = str(info.get("propertyMainType") or "").lower()
    category = main if main in ("residential", "commercial") else ("commercial" if ptype in COMMERCIAL_TYPES else None)
    city, district = canonical_city(info.get("city")), info.get("district")
    area = _to_float(attrs.get("builtUpArea")) or _to_float(item.get("floorSize")) or _to_float(
        attrs.get("landArea") or attrs.get("plotArea") or attrs.get("totalArea")
    )
    amenities = [
        clean_ws(a.get("name")) for a in item.get("featureAmenities") or []
        if isinstance(a, dict) and isinstance(a.get("name"), str) and a.get("name").strip()
    ]
    beds, baths = _to_int(attrs.get("noOfBedrooms")), _to_int(attrs.get("noOfBathrooms"))
    furnishing = info.get("furnishingType") or None
    title = _title(subtype, purpose, district, city, info.get("title"))
    extra = {
        k: str(v)
        for k, v in {
            "territory": info.get("territory"),
            "built": attrs.get("completionYear"),
            "living_rooms": attrs.get("noOfLivingrooms"),
            "verified": "yes" if item.get("isVerified") else None,
            "rega_listed": "yes" if item.get("isRegaProp") else None,
            "listing_age": item.get("ageOfListing"),
            "wasalt_id": pid,
            "subtype": subtype,
        }.items()
        if v not in (None, "", [])
    }
    rec = Record(
        id=f"wasalt:listing:{pid}",
        source="wasalt",
        kind="listing",
        url=f"{BASE}/en/property/{purpose}/{slug}",
        title=title,
        purpose=purpose,
        property_type=ptype,
        property_types=[ptype] if ptype else [],
        category=category,  # type: ignore[arg-type]
        price=price,
        currency=info.get("currencyType") or "SAR",
        rent_period=period,
        bedrooms=beds,
        bathrooms=baths,
        area_sqm=area,
        city=city,
        district=district,
        country=info.get("country") or "Saudi Arabia",
        furnishing=furnishing,
        amenities=amenities,
        published_at=_parse_dt(item.get("publishedAt") or item.get("createdAt")),
        updated_at=_parse_dt(item.get("updatedAt")),
        extra=extra,
    )
    rec.text = listing_text(rec)
    rec.summary = truncate(rec.text, 240)
    return rec


def listing_text(rec: Record, description: str | None = None) -> str:
    label = TYPE_LABELS.get(rec.property_type or "", rec.extra.get("subtype") or "Property")
    parts = [
        f"{label} for {rec.purpose} in {rec.location_label() or 'Saudi Arabia'} (Wasalt listing).",
    ]
    if rec.price is not None:
        parts.append(f"Price: {rec.price_label()}.")
    specs = []
    if rec.bedrooms is not None:
        specs.append(f"{rec.bedrooms} bedrooms")
    if rec.bathrooms is not None:
        specs.append(f"{rec.bathrooms} bathrooms")
    if rec.area_sqm:
        specs.append(f"{rec.area_sqm:,.0f} sqm")
    if specs:
        parts.append(", ".join(specs).capitalize() + ".")
    if rec.furnishing:
        parts.append(f"Furnishing: {rec.furnishing}.")
    if rec.extra.get("territory"):
        parts.append(f"Area: {rec.extra['territory']}.")
    if rec.extra.get("built"):
        parts.append(f"Age/built: {rec.extra['built']}.")
    if rec.amenities:
        parts.append("Amenities: " + ", ".join(rec.amenities[:20]) + ".")
    flags = [x for x in ("verified" if rec.extra.get("verified") else "", "REGA-listed" if rec.extra.get("rega_listed") else "") if x]
    if flags:
        parts.append("Listing status: " + ", ".join(flags) + ".")
    if rec.updated_at:
        parts.append(f"Last updated {rec.updated_at.date().isoformat()}.")
    if description:
        parts.append("Description: " + truncate(description, 900))
    return scrub_pii(" ".join(parts))


def parse_project_item(item: dict[str, Any]) -> Record | None:
    info = item.get("projectInfo") or {}
    pid, slug = item.get("id"), info.get("slug")
    if not pid or not slug:
        return None
    raw_city = info.get("city")  # Wasalt's own spelling: part of the project's URL
    city = canonical_city(raw_city)
    types = [
        t
        for t in (
            canonical_property_type(u.get("name")) or canonical_property_type(u.get("slug"))
            for u in info.get("unitTypes") or []
            if isinstance(u, dict)
        )
        if t
    ]
    units = item.get("unitsAvailable") or {}
    rec = Record(
        id=f"wasalt:project:{pid}",
        source="wasalt",
        kind="project",
        url=f"{BASE}/en/project/{(raw_city or 'Saudi-Arabia').replace(' ', '-')}/{slug}",
        title=clean_ws(info.get("title") or slug.replace("-", " ").title()),
        purpose=info.get("propertyFor") if info.get("propertyFor") in ("sale", "rent") else "sale",
        property_type=types[0] if types else None,
        property_types=types,
        price=_to_float(info.get("salePriceMin")),
        price_max=_to_float(info.get("salePriceMax")),
        currency=info.get("currencyType") or "SAR",
        city=city,
        district=info.get("district"),
        country="Saudi Arabia",
        published_at=_parse_dt(item.get("publishedAt")),
        updated_at=_parse_dt(item.get("updatedAt")),
        extra={
            k: str(v)
            for k, v in {
                "units_available": units.get("available"),
                "units_total": units.get("total"),
                "zone": info.get("zone"),
                "wasalt_id": pid,
            }.items()
            if v not in (None, "")
        },
    )
    rec.text = project_text(rec)
    rec.summary = truncate(rec.text, 240)
    return rec


def project_text(rec: Record, description: str | None = None, developer_about: str | None = None) -> str:
    lines = [f"# {rec.title} — real-estate project listed on Wasalt"]
    facts = [
        ("Location", rec.location_label()),
        ("Developer", rec.developer),
        ("Unit types", ", ".join(TYPE_LABELS.get(t, t) for t in rec.property_types)),
        ("Prices", rec.price_label()),
        ("Units available", rec.extra.get("units_available")),
        ("Completion", rec.completion),
    ]
    lines.append("\n".join(f"{k}: {v}" for k, v in facts if v))
    if description:
        lines.append("## About\n" + description)
    if developer_about:
        lines.append("## About the developer\n" + developer_about)
    return scrub_pii("\n\n".join(lines))


def parse_category_page(url: str, next_data: dict[str, Any]) -> dict[str, Any]:
    """Extract listings, projects, a market stat, FAQs and the SEO guide."""
    pp = page_props(next_data)
    result = pp.get("searchResult") or {}
    listings: list[Record] = []
    projects: list[Record] = []
    for item in result.get("properties") or []:
        if not isinstance(item, dict):
            continue
        if item.get("propertyInfo"):
            rec = parse_listing(item)
            if rec:
                listings.append(rec)
        elif item.get("projectInfo"):
            rec = parse_project_item(item)
            if rec:
                projects.append(rec)

    meta = pp.get("metaData") or {}
    h1 = clean_ws(meta.get("h1tag") or meta.get("title_tag") or "")
    path = urlsplit(url).path
    match = CATEGORY_RE.match(path)
    stat: Record | None = None
    count = result.get("count")
    if h1 and isinstance(count, int):
        type_slug = match.group("type") if match else "properties"
        purpose = match.group("purpose") if match else None
        loc_text = match.group("loc").replace("-", " ") if match else ""
        cities, _ = find_places(loc_text)
        ptype = canonical_property_type(type_slug.replace("-", " "))
        stat = Record(
            id=f"wasalt:stat:{path.rsplit('/', 1)[-1]}",
            source="wasalt",
            kind="stat",
            url=url,
            title=f"{h1} — Wasalt category",
            text=(
                f"Wasalt.sa showed {count:,} results for the category “{h1}” "
                f"(snapshot {utcnow().date().isoformat()}). Counts change constantly; open the live "
                f"category page for current availability."
            ),
            purpose=purpose,  # type: ignore[arg-type]
            property_type=ptype if type_slug != "properties" else None,
            city=cities[0] if cities else None,
            country="Saudi Arabia",
            extra={"count": str(count), "category": h1},
        )

    faqs: list[Record] = []
    for entity in ((pp.get("FAQSchema") or {}).get("mainEntity") or []):
        question = clean_ws((entity or {}).get("name") or "")
        answer = clean_ws(html_to_text(((entity or {}).get("acceptedAnswer") or {}).get("text") or ""))
        if question and answer:
            digest = hashlib.sha1(question.lower().encode()).hexdigest()[:12]
            faqs.append(
                Record(
                    id=f"wasalt:faq:{digest}",
                    source="wasalt",
                    kind="faq",
                    url=url,
                    title=question,
                    text=scrub_pii(f"Q: {question}\nA: {answer}"),
                    summary=truncate(answer, 240),
                )
            )

    guide: Record | None = None
    para = pp.get("para")
    if isinstance(para, str) and len(para) > 300:
        body = scrub_pii(html_to_text(para))
        if len(body) > 300:
            digest = hashlib.sha1(body.encode()).hexdigest()[:12]
            guide = Record(
                id=f"wasalt:guide:{digest}",
                source="wasalt",
                kind="guide",
                url=url,
                title=f"{h1} — guide" if h1 else "Wasalt guide",
                text=f"# {h1}\n\n{body}" if h1 else body,
                summary=truncate(body, 240),
            )
    return {"listings": listings, "projects": projects, "stat": stat, "faqs": faqs, "guide": guide}


def enrich_listing(rec: Record, next_data: dict[str, Any]) -> Record:
    """Merge description and safe attributes from a listing detail page."""
    v3 = page_props(next_data).get("propertyDetailsV3") or {}
    info = v3.get("propertyInfo") or {}
    description = scrub_pii(html_to_text(info.get("description") or ""))
    for attr in v3.get("additionalAttributes") or []:
        if not isinstance(attr, dict):
            continue
        label, value = clean_ws(str(attr.get("label") or "")), attr.get("value")
        if not label or value in (None, "", "-") or _SENSITIVE_LABEL.search(label) or _SENSITIVE_LABEL.search(str(attr.get("key") or "")):
            continue
        rec.extra.setdefault(label.lower().replace(" ", "_")[:40], clean_ws(str(value))[:120])
    if info.get("facingType"):
        rec.extra.setdefault("facing", str(info["facingType"]))
    rec.text = listing_text(rec, description or None)
    rec.summary = truncate(rec.text, 240)
    return rec


def enrich_project(rec: Record, next_data: dict[str, Any]) -> Record:
    data = (page_props(next_data).get("compoundDetails") or {}).get("data") or {}
    info = data.get("projectInfo") or {}
    description = scrub_pii(html_to_text(info.get("description") or ""))
    developer = data.get("developerInfo") or {}
    if isinstance(developer.get("name"), str) and developer["name"].strip():
        rec.developer = clean_ws(developer["name"])
    if info.get("completionDate"):
        rec.completion = str(info["completionDate"])[:10]
    about = scrub_pii(html_to_text(developer.get("about") or "")) if developer.get("about") else None
    rec.text = project_text(rec, description or None, about)
    rec.summary = truncate(description or rec.text, 240)
    return rec


def parse_blog_post(post: dict[str, Any], category_names: dict[int, str]) -> Record | None:
    pid = post.get("id")
    link = post.get("link")
    title = htmllib.unescape(clean_ws(((post.get("title") or {}).get("rendered")) or ""))
    body = scrub_pii(html_to_text(((post.get("content") or {}).get("rendered")) or ""))
    if not pid or not link or not title or len(body) < 200:
        return None
    cats = [category_names.get(c, "") for c in post.get("categories") or []]
    excerpt = scrub_pii(html_to_text(((post.get("excerpt") or {}).get("rendered")) or ""))
    return Record(
        id=f"wasalt:article:{pid}",
        source="wasalt",
        kind="article",
        url=link,
        title=title,
        summary=truncate(excerpt or body, 280),
        text=f"# {title}\n\n{body}",
        published_at=_parse_dt(post.get("date")),
        updated_at=_parse_dt(post.get("modified")),
        extra={"categories": ", ".join(c for c in cats if c), "section": "blog"},
    )


class WasaltScraper:
    source = "wasalt"

    def __init__(self, client: PoliteClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.report = CrawlReport(source=self.source)
        self.skipped: Counter[str] = Counter()

    async def _category_paths(self) -> list[str]:
        try:
            xml = await self.client.get_text(CATEGORY_SITEMAP)
        except FetchError as exc:
            self.report.notes.append(f"category sitemap unavailable ({exc.reason}); using fallback plan")
            return []
        return [urlsplit(loc).path for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)]

    async def _next_data(self, url: str) -> dict[str, Any] | None:
        try:
            html = await self.client.get_text(url)
        except (NotFound, RobotsDisallowed) as exc:
            self.skipped[exc.reason] += 1
            return None
        except HostBlocked:
            raise
        except FetchError as exc:
            self.report.errors += 1
            log.warning("wasalt fetch failed", extra={"url": url, "reason": exc.reason, "status": exc.status})
            return None
        return extract_next_data(html)

    async def _crawl_blog(self, records: dict[str, Record]) -> None:
        cap = self.settings.wasalt_max_blog_posts
        if cap <= 0:
            return
        exclude = {c.lower() for c in self.settings.wasalt_blog_exclude_categories}
        try:
            cats = await self.client.get_json(f"{BLOG_API}/categories?per_page=100&_fields=id,name")
            names = {c["id"]: htmllib.unescape(c.get("name", "")) for c in cats if isinstance(c, dict) and "id" in c}
        except (FetchError, TypeError, KeyError) as exc:
            names = {}
            self.report.notes.append(f"blog categories unavailable: {exc}")
        page, kept, skipped = 1, 0, 0
        fields = "id,date,modified,link,title,content,excerpt,categories"
        while kept < cap:
            try:
                posts, headers = await self.client.get_json_with_headers(
                    f"{BLOG_API}/posts?per_page=100&page={page}&_fields={fields}"
                )
            except FetchError as exc:
                if page == 1:
                    self.report.notes.append(f"blog API unavailable: {exc.reason}")
                break
            if not isinstance(posts, list) or not posts:
                break
            for post in posts:
                cat_names = {names.get(c, "").lower() for c in post.get("categories") or []} - {""}
                if cat_names and cat_names <= exclude:
                    skipped += 1
                    continue
                rec = parse_blog_post(post, names)
                if rec and kept < cap:
                    records[rec.id] = rec
                    kept += 1
            total_pages = int(headers.get("x-wp-totalpages") or page)
            if page >= total_pages:
                break
            page += 1
        self.report.notes.append(f"blog: kept {kept} posts, skipped {skipped} lifestyle-only posts")

    async def run(self) -> tuple[list[Record], CrawlReport]:
        records: dict[str, Record] = {}
        types: Counter[str] = Counter()
        try:
            paths = await self._category_paths()
        except HostBlocked as exc:
            self.report.notes.append(f"sitemap host blocked: {exc.reason}")
            paths = []
        plan = plan_categories(paths, self.settings.wasalt_max_category_pages) or FALLBACK_PLAN[
            : self.settings.wasalt_max_category_pages
        ]
        self.report.notes.append(f"category plan: {len(plan)} pages from {len(paths)} sitemap entries")

        projects: dict[str, Record] = {}
        seen_faq: set[str] = set()

        async def handle_category(url: str) -> None:
            nd = await self._next_data(url)
            if not nd:
                return
            parsed = parse_category_page(url, nd)
            for rec in parsed["listings"]:
                if rec.id not in records:
                    records[rec.id] = rec
                    types[rec.property_type or "other"] += 1
            for rec in parsed["projects"]:
                projects.setdefault(rec.id, rec)
            if parsed["stat"]:
                records[parsed["stat"].id] = parsed["stat"]
            for faq in parsed["faqs"]:
                if faq.id not in seen_faq:
                    seen_faq.add(faq.id)
                    records[faq.id] = faq
            if parsed["guide"]:
                records.setdefault(parsed["guide"].id, parsed["guide"])

        blocked = await gather_isolated([handle_category(u) for u in plan], self.report, "categories")

        if not blocked:
            # Enrich a diverse sample of listings with their detail pages.
            listings = [r for r in records.values() if r.kind == "listing"]
            sample: list[Record] = []
            buckets: dict[tuple[str | None, str | None, str | None], list[Record]] = {}
            for rec in listings:
                buckets.setdefault((rec.city, rec.property_type, rec.purpose), []).append(rec)
            while len(sample) < self.settings.wasalt_max_detail_pages and any(buckets.values()):
                for bucket in buckets.values():
                    if bucket and len(sample) < self.settings.wasalt_max_detail_pages:
                        sample.append(bucket.pop(0))

            async def handle_detail(rec: Record) -> None:
                nd = await self._next_data(rec.url)
                if nd:
                    enrich_listing(rec, nd)

            blocked = await gather_isolated([handle_detail(r) for r in sample], self.report, "details")

        project_list = list(projects.values())
        if not blocked:

            async def handle_project(rec: Record) -> None:
                nd = await self._next_data(rec.url)
                if nd:
                    enrich_project(rec, nd)

            await gather_isolated(
                [handle_project(r) for r in project_list[: self.settings.wasalt_max_project_pages]],
                self.report,
                "projects",
            )
        for rec in project_list:
            records[rec.id] = rec

        # The blog lives on a different host, so it is crawled even if wasalt.sa blocked us.
        try:
            await self._crawl_blog(records)
        except HostBlocked as exc:
            self.report.notes.append(f"blog stopped: {exc.reason}")

        self.report.records = len(records)
        self.report.notes.append("listing types: " + ", ".join(f"{k}={v}" for k, v in types.most_common()))
        if self.skipped:
            self.report.notes.append("skipped: " + ", ".join(f"{k}={v}" for k, v in self.skipped.items()))
        self.report.finished_at = utcnow()
        if not any(r.kind == "listing" for r in records.values()):
            self.report.ok = False
        return list(records.values()), self.report
