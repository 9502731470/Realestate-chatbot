"""DarGlobal (darglobal.co.uk) adapter.

Discovery: robots-permitted sitemap.xml + the /projects index (project cards).
Extraction: the Strapi-backed ``__NEXT_DATA__`` payload of each page, which
contains the structured project data (location, unit types, sizes, status,
completion, amenities, landmarks, FAQs...). Articles (blog / press /
insights) are captured for general and market questions.
"""

from __future__ import annotations

import hashlib
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
from app.scraping.nextdata import (
    DEFAULT_SKIP_KEYS,
    extract_next_data,
    find_first,
    harvest_text,
    html_meta,
    html_to_text,
    page_props,
    rich_to_text,
)
from app.scraping.pii import scrub_pii
from app.textutil import clean_ws, normalize_for_search, truncate
from app.vocab import (
    canonical_property_type,
    detect_brands,
    parse_area_range,
    parse_location,
    parse_money,
    property_types_in,
)

log = logging.getLogger(__name__)

BASE = "https://darglobal.co.uk"
SITEMAP = f"{BASE}/sitemap.xml"
ARTICLE_SECTIONS = ("insights", "press", "blog")
SUBPAGE_SECTIONS = ("amaya", "rayana", "trump-plaza", "one-of-one", "partners")

# Transactional / legal / campaign pages that carry no useful knowledge.
SKIP_SLUGS = {
    "pay-online", "thank-you", "success", "careers", "internship-application", "internships",
    "privacy-policy", "terms-conditions", "terms-of-uses", "landing-page", "landingpage",
    "new-home-page", "campaigns", "bookyourunit", "win-a-trip-to-dubai", "become-a-broker",
    "become-an-agent", "broker-contest-terms-conditions", "get-in-touch", "exclusive-membership",
    "dg-circle", "aston-martin-media", "darglobal-insignia-card", "insignia", "projects", "blog",
    "press", "insights", "status", "newstaticlandingpagehtml",
}
ARTICLE_SKIP_KEYS = DEFAULT_SKIP_KEYS | {
    "relatedBlogs", "otherBlogs", "recentBlogs", "latestBlogs", "similarBlogs", "moreBlogs",
    "relatedPress", "otherPress", "recentPress", "pressListSSR", "pressYears", "initialBlogs",
    "initialMeta", "meta", "pageData", "relatedInsights", "otherInsights",
}
_TITLE_SUFFIX = re.compile(r"\s*[|\-–]\s*(?:dar\s*global|darglobal).*$", re.I)


def _slug(url: str) -> str:
    return urlsplit(url).path.strip("/")


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _attrs(entity: Any) -> dict[str, Any]:
    if isinstance(entity, dict):
        if isinstance(entity.get("attributes"), dict):
            return entity["attributes"]
        if isinstance(entity.get("data"), dict):
            return _attrs(entity["data"])
        return entity
    if isinstance(entity, list) and entity:
        return _attrs(entity[0])
    return {}


def _clean_title(text: str | None) -> str:
    return _TITLE_SUFFIX.sub("", clean_ws(text or "")).strip()


_ANCHOR_HREF = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"'#?]+)", re.I)
_CANONICAL_TAG = re.compile(r"<link\b[^>]*\brel\s*=\s*[\"']canonical[\"'][^>]*>", re.I)
_HREF_VALUE = re.compile(r"\bhref\s*=\s*[\"']([^\"']+)[\"']", re.I)
PROJECT_SUBSECTIONS = ("one-of-one",)


def _site_path(href: str) -> str | None:
    """'/dg1', 'https://www.darglobal.co.uk/dg1/' -> '/dg1'; other sites -> None."""
    parts = urlsplit(href.strip())
    if parts.netloc and parts.netloc.lower().removeprefix("www.") != "darglobal.co.uk":
        return None
    path = "/" + parts.path.strip("/")
    return path if path != "/" else None


def project_links_from_html(html: str) -> list[str]:
    """Project-like paths linked from the /projects page markup, in page order.

    The fallback when the page ships no project cards in ``__NEXT_DATA__``: the
    cards' links are still in the HTML, so the projects are fetched first.
    """
    paths: list[str] = []
    for href in _ANCHOR_HREF.findall(html or ""):
        path = _site_path(href)
        if not path:
            continue
        segments = path.strip("/").split("/")
        top = segments[0].lower()
        if len(segments) == 1:
            if top in SKIP_SLUGS or top in ARTICLE_SECTIONS:
                continue
        elif top not in PROJECT_SUBSECTIONS:
            continue
        if path not in paths:
            paths.append(path)
    return paths


def canonical_page_url(html: str, fallback: str) -> str:
    """The page's ``<link rel="canonical">`` on darglobal.co.uk, else ``fallback``.

    Redirecting aliases (/Trump-International-Hotel-Tower-Dubai) serve the same
    project as the canonical page; keying records on the canonical URL merges them.
    """
    tag = _CANONICAL_TAG.search((html or "")[:200_000])
    href = _HREF_VALUE.search(tag.group(0)) if tag else None
    path = _site_path(href.group(1)) if href else None
    if not path or path.strip("/").split("/")[0].lower() in SKIP_SLUGS:
        return fallback
    return BASE + path


# ---------------------------------------------------------------------- parsing
def parse_project_cards(next_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map '/slug' -> card metadata from the /projects page tabs."""
    pp = page_props(next_data)
    cards: dict[str, dict[str, Any]] = {}
    tabs = _attrs(pp.get("pageData")).get("Tabs") or []
    for tab in tabs:
        heading = (tab or {}).get("TabHeading")
        for item in (tab or {}).get("TabData") or []:
            link = (item or {}).get("Link")
            if not isinstance(link, str) or not link.startswith("/"):
                continue
            path = "/" + link.strip("/").split("?")[0]
            card = cards.setdefault(path, {"countries": []})
            card.update(
                {
                    "title": item.get("ProjectTitle"),
                    "location": item.get("Location"),
                    "type": item.get("ProjectType"),
                    "completion": item.get("completionDate"),
                    "bedrooms": item.get("Bedrooms"),
                    "description": item.get("ProjectDescription"),
                }
            )
            if heading and heading not in card["countries"]:
                card["countries"].append(heading)
    return cards


def parse_project_page(
    url: str, next_data: dict[str, Any], html: str = "", card: dict[str, Any] | None = None
) -> Record | None:
    pp = page_props(next_data)
    pdd = pp.get("projectDetailsData")
    if not pdd:
        return None
    attrs = _attrs(pdd)
    details_root = attrs.get("ProjectDetails") or {}
    card = card or {}
    meta = html_meta(html) if html else {}

    banner = details_root.get("ProjectBanner") or {}
    about = details_root.get("AboutProject") or {}
    facts: dict[str, str] = {}
    for item in about.get("AboutDetailsType") or []:
        name, value = (item or {}).get("name"), (item or {}).get("value")
        if isinstance(name, str) and value not in (None, ""):
            facts[clean_ws(name).rstrip(":")] = clean_ws(str(value))

    def fact(*names: str) -> str | None:
        lowered = {k.lower(): v for k, v in facts.items()}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    title = _clean_title(attrs.get("title") or banner.get("Heading") or card.get("title") or meta.get("h1"))
    if not title:
        return None

    location_text = " ".join(
        str(x)
        for x in (
            banner.get("Location"),
            fact("Location"),
            card.get("location"),
            ((details_root.get("loaction") or {}).get("locationDetailsWithImages") or {}).get("location"),
            " ".join(card.get("countries") or []),
            title,
        )
        if x
    )
    city, country = parse_location(location_text)
    if not (city or country):
        # Campaign pages reuse the project template ("Buy Property with Crypto");
        # every real project has a place. Such pages are kept as ordinary pages.
        return None
    display_location = clean_ws(str(banner.get("Location") or card.get("location") or fact("Location") or ""))

    type_text = " ".join(
        str(x) for x in (card.get("type"), fact("Property Type"), fact("Unit Type", "Unit type", "Unit types")) if x
    )
    types = property_types_in(type_text)
    if not types:
        guessed = canonical_property_type(card.get("type"))
        types = [guessed] if guessed else []

    status = fact("Status", "Project Status")
    completion = card.get("completion") or fact("Completion", "Completion Date", "Handover", "Delivery")
    unit_types = fact("Unit Type", "Unit type", "Unit types", "Bedrooms")
    area_lo, area_hi = parse_area_range(fact("Area (sqm)", "Area", "Size", "Sizes", "Unit Size"))
    price, currency = parse_money(
        " ".join(str(x) for x in (banner.get("StartingPrice"), fact("Starting Price", "Price")) if x)
    )

    amenities_section = details_root.get("Amenities") or {}
    amenities = [
        clean_ws(a.get("name"))
        for a in amenities_section.get("amenitiesList") or []
        if isinstance(a, dict) and isinstance(a.get("name"), str) and a.get("name").strip()
    ]
    key_features = details_root.get("KeyFeatures") or {}
    landmarks = [
        f"{clean_ws(k.get('Title'))} ({clean_ws(k.get('Duration') or '')})".replace(" ()", "")
        for k in key_features.get("keyfeaturelist") or []
        if isinstance(k, dict) and k.get("Title")
    ]
    why = details_root.get("whyInvest") or {}
    why_points = [
        f"{clean_ws(i.get('name'))}: {clean_ws(str(i.get('value') or ''))}".rstrip(": ")
        for i in why.get("investlist") or []
        if isinstance(i, dict) and i.get("name")
    ]
    location_section = details_root.get("loaction") or details_root.get("location") or {}

    faqs: list[str] = []
    for group in attrs.get("faqContent") or []:
        for faq in (group or {}).get("faqs") or []:
            q = clean_ws((faq or {}).get("question") or "")
            a = rich_to_text((faq or {}).get("answer"))
            if q and a:
                faqs.append(f"Q: {q}\nA: {a}")

    seo = details_root.get("SEO") or attrs.get("SEO") or {}
    seo_desc = clean_ws(seo.get("metaDescription") or meta.get("description") or "")

    known = {
        "ProjectBanner", "AboutProject", "Amenities", "KeyFeatures", "whyInvest", "loaction",
        "location", "SEO",
    }
    extra_text = harvest_text({k: v for k, v in details_root.items() if k not in known})

    sections: list[str] = [f"# {title} — DarGlobal project"]
    header = [
        ("Developer", "DarGlobal"),
        ("Location", display_location or ", ".join(x for x in (city, country) if x)),
        ("Property type", fact("Property Type") or card.get("type")),
        ("Unit types", unit_types),
        ("Status", status),
        ("Completion / handover", completion),
        ("Unit sizes (sqm)", fact("Area (sqm)", "Area", "Size")),
        ("Number of floors", fact("Number of Floors", "Floors")),
        ("Starting price", banner.get("StartingPrice") or fact("Starting Price")),
    ]
    for k, v in facts.items():
        if k.lower() not in {h[0].lower() for h in header} and k.lower() not in {
            "location", "property type", "unit type", "unit types", "status", "area (sqm)", "number of floors",
        }:
            header.append((k, v))
    sections.append("\n".join(f"{k}: {v}" for k, v in header if v))

    about_text = "\n".join(
        x for x in (clean_ws(about.get("title") or ""), rich_to_text(about.get("richDescription") or about.get("description"))) if x
    )
    if about_text:
        sections.append("## About\n" + about_text)
    if amenities:
        intro = rich_to_text(amenities_section.get("description"))
        sections.append("## Amenities\n" + (intro + "\n" if intro else "") + "\n".join(f"- {a}" for a in amenities))
    loc_desc = rich_to_text(location_section.get("description"))
    kf_desc = rich_to_text(key_features.get("description"))
    if loc_desc or kf_desc or landmarks:
        body = "\n".join(x for x in (loc_desc, kf_desc) if x)
        if landmarks:
            body += "\nNearby (travel time): " + "; ".join(landmarks)
        sections.append("## Location & connectivity\n" + body.strip())
    why_desc = rich_to_text(why.get("description"))
    if why_desc or why_points:
        sections.append("## Why invest\n" + "\n".join([why_desc, *[f"- {p}" for p in why_points]]).strip())
    if faqs:
        sections.append("## FAQs\n" + "\n\n".join(faqs))
    if extra_text:
        sections.append("## More details\n" + "\n".join(extra_text))
    disclaimer = clean_ws(attrs.get("disclaimer") or "")
    if disclaimer:
        sections.append("## Disclaimer\n" + disclaimer)

    text = scrub_pii("\n\n".join(sections))
    summary = seo_desc or truncate(rich_to_text(about.get("description")), 280) or None
    brands = detect_brands(" ".join([title, text[:4000]]))

    extra = {
        k: v
        for k, v in {
            "unit_types": unit_types,
            "brands": ", ".join(brands),
            "floors": fact("Number of Floors", "Floors"),
            "location_label": display_location,
            "card_location": card.get("location"),
        }.items()
        if v
    }

    return Record(
        id=f"darglobal:project:{_slug(url).lower()}",
        source="darglobal",
        kind="project",
        url=url,
        title=title,
        summary=scrub_pii(summary) if summary else None,
        text=text,
        purpose="sale",
        property_type=types[0] if types else None,
        property_types=types,
        category="residential" if not set(types) & {"office", "shop", "showroom"} else "commercial",
        price=price,
        currency=currency,
        area_sqm=area_lo,
        area_max_sqm=area_hi,
        city=city,
        country=country,
        developer="DarGlobal",
        project=title,
        status=status,
        completion=str(completion) if completion else None,
        amenities=amenities,
        updated_at=_parse_dt(attrs.get("updatedAt")),
        published_at=_parse_dt(attrs.get("publishedAt")),
        extra=extra,
    )


def parse_article_page(url: str, section: str, next_data: dict[str, Any] | None, html: str) -> Record | None:
    pp = page_props(next_data)
    meta = html_meta(html)
    title = _clean_title(meta.get("h1") or meta.get("og_title") or meta.get("title"))

    # Prefer a full-article HTML string if the page ships one (e.g. blogDetailsHTML).
    candidates: list[str] = []
    for _key, value in pp.items():
        if isinstance(value, str) and len(value) > 400 and "<" in value:
            candidates.append(html_to_text(value))
    harvested = "\n\n".join(harvest_text({k: v for k, v in pp.items() if k not in ARTICLE_SKIP_KEYS}))
    candidates.append(harvested)
    body = max(candidates, key=len) if candidates else ""
    body = scrub_pii(body)
    if not title:
        title = truncate(body.split("\n", 1)[0], 120)
    if len(body) < 200 or not title:
        return None

    date = find_first(
        {k: v for k, v in pp.items() if k not in {"globalData", "pageData"}},
        ("publishedDate", "PublishedDate", "publishDate", "date", "Date", "publishedAt"),
    )
    summary = meta.get("description") or truncate(body, 280)
    return Record(
        id=f"darglobal:article:{_slug(url).lower()}",
        source="darglobal",
        kind="article",
        url=url,
        title=title,
        summary=scrub_pii(summary),
        text=f"# {title}\n\n{body}",
        published_at=_parse_dt(date),
        extra={"section": section},
    )


def parse_generic_page(url: str, next_data: dict[str, Any] | None, html: str) -> list[Record]:
    pp = page_props(next_data)
    meta = html_meta(html)
    texts = harvest_text({k: v for k, v in pp.items() if k not in ARTICLE_SKIP_KEYS - {"pageData"}})
    if not texts and html:
        texts = [html_to_text(html)]
    slug = _slug(url).lower() or "home"
    title = _clean_title(meta.get("h1") or meta.get("og_title") or meta.get("title")) or slug.replace("-", " ").title()

    records: list[Record] = []
    prose: list[str] = []
    for chunk in texts:
        if chunk.startswith("Q: ") and "\nA: " in chunk:
            question = chunk[3:].split("\nA: ", 1)[0].strip()
            digest = hashlib.sha1(question.lower().encode()).hexdigest()[:12]
            records.append(
                Record(
                    id=f"darglobal:faq:{digest}",
                    source="darglobal",
                    kind="faq",
                    url=url,
                    title=question,
                    text=chunk,
                    summary=truncate(chunk.split("\nA: ", 1)[1], 240),
                )
            )
        else:
            prose.append(chunk)
    body = scrub_pii("\n\n".join(prose))
    if len(body) >= 200:
        records.append(
            Record(
                id=f"darglobal:page:{slug}",
                source="darglobal",
                kind="page",
                url=url,
                title=title,
                summary=scrub_pii(meta.get("description") or truncate(body, 280)),
                text=f"# {title}\n\n{body}",
            )
        )
    return records


def dedupe_projects(records: dict[str, Record]) -> int:
    """Merge project records that describe the same project under different URLs.

    Same title in the same place means the same project; the copy at a lowercase
    (canonical-looking) URL is kept, else the richer one. Returns how many were dropped.
    """
    kept: dict[tuple[str, str], Record] = {}
    dropped = 0
    for rec in sorted((r for r in records.values() if r.kind == "project"), key=lambda r: r.id):
        key = (normalize_for_search(rec.title), normalize_for_search(rec.city or rec.country or ""))
        other = kept.get(key)
        if other is None:
            kept[key] = rec
            continue

        def rank(r: Record) -> tuple[bool, int]:
            return (_slug(r.url) != _slug(r.url).lower(), -len(r.text))

        winner, loser = (other, rec) if rank(other) <= rank(rec) else (rec, other)
        kept[key] = winner
        records.pop(loser.id, None)
        dropped += 1
    return dropped


# ---------------------------------------------------------------------- crawler
class DarGlobalScraper:
    source = "darglobal"

    def __init__(self, client: PoliteClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.report = CrawlReport(source=self.source)
        self.skipped: Counter[str] = Counter()

    async def _sitemap_paths(self) -> list[str]:
        try:
            xml = await self.client.get_text(SITEMAP)
        except FetchError as exc:
            self.report.notes.append(f"sitemap unavailable: {exc.reason}")
            return []
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
        paths: list[str] = []
        for loc in locs:
            parts = urlsplit(loc)
            if parts.netloc and "darglobal.co.uk" not in parts.netloc:
                continue
            path = "/" + parts.path.strip("/")
            if path not in paths:
                paths.append(path)
        return paths

    async def _fetch(self, path: str) -> tuple[str, dict[str, Any] | None] | None:
        url = BASE + path
        try:
            html = await self.client.get_text(url)
        except (NotFound, RobotsDisallowed) as exc:
            self.skipped[exc.reason] += 1
            return None
        except HostBlocked:
            raise
        except FetchError as exc:
            self.report.errors += 1
            log.warning("darglobal fetch failed", extra={"url": url, "reason": exc.reason, "status": exc.status})
            return None
        return html, extract_next_data(html)

    async def run(self) -> tuple[list[Record], CrawlReport]:
        records: dict[str, Record] = {}
        try:
            paths = await self._sitemap_paths()
            cards: dict[str, dict[str, Any]] = {}
            fetched = await self._fetch("/projects")
            if fetched and fetched[1]:
                cards = parse_project_cards(fetched[1])
                if not cards:
                    self.report.notes.append("/projects: no project cards in __NEXT_DATA__; using sitemap discovery")
            elif fetched:
                self.report.notes.append("/projects: no __NEXT_DATA__; using sitemap discovery")
            else:
                self.report.notes.append("/projects: fetch unavailable; using sitemap discovery")
        except HostBlocked as exc:
            self.report.ok = False
            self.report.notes.append(f"stopped: {exc.reason}")
            log.warning("darglobal crawl stopped", extra={"reason": exc.reason})
            return self._finish(records)
        if not cards and fetched:
            links = project_links_from_html(fetched[0])
            if links:
                cards = {path: {} for path in links}
                self.report.notes.append(f"/projects: {len(links)} project links read from the page HTML")
        self.report.notes.append(f"{len(cards)} project cards, {len(paths)} sitemap URLs")

        root_paths = [p for p in paths if p.count("/") == 1 and p.strip("/").lower() not in SKIP_SLUGS and p != "/"]
        sub_paths = [p for p in paths if p.count("/") == 2 and p.split("/")[1] in SUBPAGE_SECTIONS]
        project_first = list(dict.fromkeys([*cards.keys(), *root_paths]))
        page_paths = project_first[: len(cards) + self.settings.dg_max_pages] + sub_paths[:20]
        page_paths = list(dict.fromkeys(["/", "/about", "/faq", "/why-invest", *page_paths]))

        async def handle_page(path: str) -> None:
            fetched = await self._fetch(path)
            if not fetched:
                return
            html, nd = fetched
            url = BASE + (path if path != "/" else "")
            project = (
                parse_project_page(canonical_page_url(html, url), nd or {}, html, cards.get(path)) if nd else None
            )
            if project:
                records[project.id] = project  # aliases of one page share its canonical id
                return
            for rec in parse_generic_page(url, nd, html):
                records.setdefault(rec.id, rec)

        if await gather_isolated([handle_page(p) for p in page_paths], self.report, "pages"):
            return self._finish(records)

        # Articles: round-robin across sections so each is represented.
        by_section = {s: [p for p in paths if p.startswith(f"/{s}/")] for s in ARTICLE_SECTIONS}
        article_paths: list[tuple[str, str]] = []
        while len(article_paths) < self.settings.dg_max_articles and any(by_section.values()):
            for section in ARTICLE_SECTIONS:
                if by_section[section] and len(article_paths) < self.settings.dg_max_articles:
                    article_paths.append((section, by_section[section].pop(0)))

        async def handle_article(section: str, path: str) -> None:
            fetched = await self._fetch(path)
            if not fetched:
                return
            html, nd = fetched
            rec = parse_article_page(BASE + path, section, nd, html)
            if rec:
                records[rec.id] = rec

        await gather_isolated([handle_article(s, p) for s, p in article_paths], self.report, "articles")
        return self._finish(records)

    def _finish(self, records: dict[str, Record]) -> tuple[list[Record], CrawlReport]:
        merged = dedupe_projects(records)
        if merged:
            self.report.notes.append(f"merged {merged} duplicate project pages")
        kinds = Counter(r.kind for r in records.values())
        self.report.notes.append("records: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
        if self.skipped:
            self.report.notes.append("skipped: " + ", ".join(f"{k}={v}" for k, v in self.skipped.items()))
        self.report.records = len(records)
        self.report.finished_at = utcnow()
        if not any(r.kind == "project" for r in records.values()):
            self.report.ok = False
        return list(records.values()), self.report
