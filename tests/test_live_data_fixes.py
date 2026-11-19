"""Regressions found by testing the live deployment against real DarGlobal / Wasalt data.

1. "Riyadh" also matched a *district* called Al-Riyadh (in Jeddah), so Riyadh
   questions returned a Jeddah flat; "Khobar" matched a district of Madinah.
2. Wasalt spells cities its own way (Aldammam, Alttayif, Bariduh, Hayil, Tbwk...),
   so "Dammam", "Taif", "Buraydah" or "Hail" found nothing.
3. Wasalt files villas under the slug "villa-townhouse": every villa became a townhouse.
4. DarGlobal: with /projects unreadable, only the first 40 sitemap paths were
   fetched (projects sit at the end of the alphabetical sitemap); a redirecting
   alias produced a duplicate project; a campaign page was counted as a project.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings
from app.models import Record
from app.rag.embeddings import HashingEmbedder
from app.rag.index import SearchIndex, build_index
from app.rag.query import QueryParser, Vocabulary
from app.rag.retriever import Retriever
from app.scraping.darglobal import (
    BASE,
    canonical_page_url,
    dedupe_projects,
    parse_project_page,
    project_links_from_html,
)
from app.scraping.nextdata import extract_next_data
from app.scraping.pipeline import crawl_all
from app.scraping.wasalt import parse_listing, parse_project_item
from app.textutil import place_key
from app.vocab import canonical_city
from tests.fixtures.sites import (
    MockSites,
    dg_project_page,
    dg_projects_page_without_data,
    wasalt_listing,
    wasalt_project_item,
)

# Districts that share a city's name, as found in the live Wasalt data.
LIVE_DISTRICTS = [
    ("Al-Riyadh", "Jeddah"), ("Riyadh", "Diriyah"), ("Al Khobar", "Madinah"), ("Al Jubail", "Jubail"),
    ("Al Dhahran", "Dhahran"), ("An-Narjis", "Riyadh"), ("Al Malqa", "Riyadh"),
]


def _vocab() -> Vocabulary:
    districts: dict[str, tuple[str, str | None]] = {}
    for name, city in LIVE_DISTRICTS:  # first one wins, as in SearchIndex._build_vocab
        districts.setdefault(place_key(name), (name, city))
    return Vocabulary(districts=districts)


# ------------------------------------------------------------- 1. city vs district
@pytest.mark.parametrize(
    "question,cities,districts",
    [
        ("Apartments for rent in Riyadh under 80,000 SAR", ["Riyadh"], []),  # homepage example
        ("3-bedroom apartments for rent in Riyadh under 90,000 SAR", ["Riyadh"], []),  # homepage example
        ("شقق للإيجار في الرياض", ["Riyadh"], []),
        ("apartments for rent in Khobar", ["Al Khobar"], []),
        ("apartments for rent in Al Khobar", ["Al Khobar"], []),
        ("land for sale in Jubail", ["Jubail"], []),  # same-city collision: not narrowed to one district
        ("villas in Dhahran", ["Dhahran"], []),
        # Naming the district's own city still selects the district.
        ("apartment in Al-Riyadh district, Jeddah", ["Jeddah"], ["Al-Riyadh"]),
        ("apartments in Al Khobar district in Madinah", ["Madinah"], ["Al Khobar"]),
        # Ordinary districts are unaffected.
        ("villa in An-Narjis", ["Riyadh"], ["An-Narjis"]),
        ("cheapest land in Al Malqa", ["Riyadh"], ["Al Malqa"]),
    ],
)
def test_city_names_are_not_mistaken_for_districts(question, cities, districts):
    f = QueryParser(_vocab()).parse(question).filters
    assert (f.cities, f.districts) == (cities, districts)


# ------------------------------------------------------------- 2. city spellings
@pytest.mark.parametrize(
    "stored,canonical",
    [
        ("Aldammam", "Dammam"), ("Alttayif", "Taif"), ("Bariduh", "Buraidah"), ("Hayil", "Hail"),
        ("Tbwk", "Tabuk"), ("Earear", "Arar"), ("Eanizah", "Unaizah"), ("Alzahran", "Dhahran"),
        ("Khobar", "Al Khobar"), ("Makkah Al Mukarramah", "Makkah"), ("Hafar Al-Batin", "Hafar Al Batin"),
        ("Al-Zulfi", "Al Zulfi"), ("Riyadh", "Riyadh"),
        ("Mukhtat Rest Houses Al Omariyah", "Mukhtat Rest Houses Al Omariyah"),  # unknown: unchanged
        ("King Fahd Road, Riyadh", "King Fahd Road, Riyadh"),  # only whole names are standardised
    ],
)
def test_wasalt_city_spellings_are_standardised(stored, canonical):
    assert canonical_city(stored) == canonical


def test_listing_city_is_standardised_but_project_url_keeps_wasalt_spelling():
    rec = parse_listing(wasalt_listing(1, "rent", "Apartment", "Aldammam", "Al Faisaliyah", 40000, 2, 2, 120))
    assert rec.city == "Dammam" and "Dammam" in rec.title
    item = wasalt_project_item(7)
    item["projectInfo"]["city"] = "Aldammam"
    project = parse_project_item(item)
    assert project.city == "Dammam"
    assert "/en/project/Aldammam/" in project.url  # Wasalt's own URL must not change


# ------------------------------------------------------------- 3. villa vs townhouse
@pytest.mark.parametrize(
    "label,slug,expected",
    [
        ("Villa", "villa-townhouse", "villa"),  # what Wasalt actually sends for villas
        ("Townhouse", "villa-townhouse", "townhouse"),
        (None, "apartment", "apartment"),  # no label: the slug is still used
        ("Apartment", "apartment", "apartment"),
    ],
)
def test_wasalt_label_decides_the_property_type(label, slug, expected):
    item = wasalt_listing(2, "sale", "Villa", "Jeddah", "Al Marjan", 3_500_000, 7, 5, 305)
    item["propertyInfo"]["propertySubType"] = label
    item["propertyInfo"]["propertySubTypeSlug"] = slug
    assert parse_listing(item).property_type == expected


def test_project_unit_types_prefer_the_label():
    item = wasalt_project_item(8)
    item["projectInfo"]["unitTypes"] = [{"name": "Villa", "slug": "villa-townhouse"}, {"name": "Land", "slug": "residential-land"}]
    assert parse_project_item(item).property_types == ["villa", "land"]


# ----------------------------------------------- 1-3 end to end: question -> index -> results
def _listing(pid: int, city: str, district: str, ptype: str, purpose: str, price: float, beds: int) -> Record:
    return Record(
        id=f"wasalt:listing:{pid}", source="wasalt", kind="listing", url=f"https://wasalt.sa/en/property/{purpose}/x-{pid}",
        title=f"{ptype.title()} for {purpose} in {district}, {city}", purpose=purpose, property_type=ptype,
        property_types=[ptype], price=price, currency="SAR", rent_period="year" if purpose == "rent" else None,
        bedrooms=beds, city=city, district=district, country="Saudi Arabia",
        text=f"{ptype.title()} for {purpose} in {district}, {city}. {beds} bedrooms.",
    )


@pytest.fixture
def live_like_index(tmp_path: Path) -> Iterator[SearchIndex]:
    records = [
        _listing(1, "Riyadh", "Al-Quds", "apartment", "rent", 70_000, 3),
        _listing(2, "Riyadh", "Al-Hamra", "apartment", "rent", 85_000, 3),
        _listing(3, "Jeddah", "Al-Riyadh", "apartment", "rent", 20_000, 3),  # the impostor
        _listing(4, "Diriyah", "Riyadh", "apartment", "rent", 60_000, 3),
        _listing(5, "Madinah", "Al Khobar", "apartment", "rent", 30_000, 2),
        _listing(6, "Al Khobar", "Al Aqrabiyah", "apartment", "rent", 50_000, 2),
        _listing(7, "Aldammam", "Al Faisaliyah", "apartment", "rent", 40_000, 2),  # spelling from an old snapshot
        _listing(8, "Dammam", "Al Shati", "apartment", "rent", 45_000, 2),
        _listing(9, "Jeddah", "Al Marjan", "villa", "sale", 3_500_000, 7),
    ]
    build_index(records, tmp_path / "build", HashingEmbedder(), backend="numpy")
    index = SearchIndex(tmp_path / "build", HashingEmbedder())
    yield index
    index.close()


@pytest.mark.parametrize(
    "question,expected_ids",
    [
        ("3-bedroom apartments for rent in Riyadh under 90,000 SAR", {1, 2}),
        ("apartments for rent in Khobar", {6}),
        ("apartments for rent in Dammam", {7, 8}),
        ("villas for sale in Jeddah", {9}),
    ],
)
def test_questions_reach_the_right_listings(live_like_index: SearchIndex, question, expected_ids):
    retriever = Retriever(live_like_index)
    f = retriever.parser.parse(question).filters
    res = live_like_index.search_records(f, kinds=["listing"], limit=20)
    assert {int(r.id.rsplit(":", 1)[1]) for r in res.records} == expected_ids, f.to_dict()


# ------------------------------------------------------------- 4. DarGlobal coverage
def test_project_links_are_read_from_projects_page_markup():
    html = (
        '<a href="/dg1">DG1</a><a href="https://darglobal.co.uk/urban-oasis-by-missoni/">Missoni</a>'
        '<a href="/one-of-one/project-pagani">Pagani</a><a href="/blog/news">Blog</a><a href="/careers">Jobs</a>'
        '<a href="https://example.com/x">Elsewhere</a><a href="/projects">All</a><a href="/dg1#top">DG1 again</a>'
    )
    assert project_links_from_html(html) == ["/dg1", "/urban-oasis-by-missoni", "/one-of-one/project-pagani"]


@pytest.mark.parametrize(
    "canonical_href,expected",
    [
        ("https://darglobal.co.uk/trump-international-hotel-and-tower-dubai", f"{BASE}/trump-international-hotel-and-tower-dubai"),
        ("/dg1", f"{BASE}/dg1"),
        ("https://other-site.com/dg1", f"{BASE}/Alias-Path"),  # another site: ignored
        ("https://darglobal.co.uk/", f"{BASE}/Alias-Path"),  # homepage: ignored
        ("https://darglobal.co.uk/projects", f"{BASE}/Alias-Path"),  # listing page: ignored
    ],
)
def test_canonical_page_url(canonical_href, expected):
    html = f'<html><head><link rel="canonical" href="{canonical_href}"/></head></html>'
    assert canonical_page_url(html, f"{BASE}/Alias-Path") == expected
    assert canonical_page_url("<html></html>", f"{BASE}/Alias-Path") == f"{BASE}/Alias-Path"


def test_page_without_a_place_is_not_a_project():
    next_data = {"props": {"pageProps": {"projectDetailsData": {"attributes": {
        "title": "Buy Property with Crypto",
        "ProjectDetails": {"ProjectBanner": {"Heading": "Buy Property with Crypto", "Location": None}},
    }}}}}
    assert parse_project_page(f"{BASE}/Invest-crypto", next_data) is None


def test_duplicate_projects_are_merged():
    page = dg_project_page("ttj")
    a = parse_project_page(f"{BASE}/trump-tower-jeddah", extract_next_data(page), page)
    b = parse_project_page(f"{BASE}/Trump-Tower-Jeddah-Alias", extract_next_data(page), page)
    records = {a.id: a, b.id: b}
    assert dedupe_projects(records) == 1
    assert [r.url for r in records.values()] == [f"{BASE}/trump-tower-jeddah"]


def _sitemap(paths: list[str]) -> bytes:
    urls = "".join(f"<url><loc>{BASE}{p}</loc></url>" for p in paths)
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'.encode()


FILLER = [f"/a-page-{i:02d}" for i in range(60)]  # alphabetically before the projects, like the live sitemap


def _project_ids(crawl: dict) -> set[str]:
    return {r.id for r in crawl["records"] if r.kind == "project"}


def test_projects_at_the_end_of_a_long_sitemap_are_still_fetched(settings: Settings, sites: MockSites):
    # Live situation: /projects ships no data and the projects come after ~60 other paths.
    sites.routes[f"{BASE}/sitemap.xml"] = (200, {"content-type": "application/xml"}, _sitemap([*FILLER, "/dg1", "/trump-tower-jeddah"]))
    sites.routes[f"{BASE}/projects"] = (200, {}, dg_projects_page_without_data().encode())
    crawl = asyncio.run(crawl_all(settings, sources=["darglobal"], transport=sites.transport))
    assert _project_ids(crawl) == {"darglobal:project:dg1", "darglobal:project:trump-tower-jeddah"}


def test_projects_page_links_are_fetched_first(settings: Settings, sites: MockSites):
    settings.dg_max_pages = 5  # far fewer than the sitemap holds: only the page links guarantee coverage
    sites.routes[f"{BASE}/sitemap.xml"] = (200, {"content-type": "application/xml"}, _sitemap([*FILLER, "/dg1", "/trump-tower-jeddah"]))
    markup = '<html><body><a href="/dg1">DG1</a> <a href="/trump-tower-jeddah">Trump Tower Jeddah</a></body></html>'
    sites.routes[f"{BASE}/projects"] = (200, {}, markup.encode())
    crawl = asyncio.run(crawl_all(settings, sources=["darglobal"], transport=sites.transport))
    report = crawl["manifest"]["reports"][0]
    assert "/projects: 2 project links read from the page HTML" in report["notes"]
    assert _project_ids(crawl) == {"darglobal:project:dg1", "darglobal:project:trump-tower-jeddah"}


def test_redirecting_alias_does_not_duplicate_a_project(settings: Settings, sites: MockSites):
    canonical = f'<link rel="canonical" href="{BASE}/trump-tower-jeddah"/>'
    ttj = dg_project_page("ttj").replace("</head>", canonical + "</head>").encode()
    sites.routes[f"{BASE}/trump-tower-jeddah"] = (200, {}, ttj)
    sites.routes[f"{BASE}/Trump-Tower-Jeddah-Old"] = (200, {}, ttj)  # what the redirect serves
    sites.routes[f"{BASE}/sitemap.xml"] = (200, {"content-type": "application/xml"},
                                           _sitemap(["/dg1", "/Trump-Tower-Jeddah-Old", "/trump-tower-jeddah"]))
    crawl = asyncio.run(crawl_all(settings, sources=["darglobal"], transport=sites.transport))
    ttj_records = [r for r in crawl["records"] if r.kind == "project" and "Trump" in r.title]
    assert [r.url for r in ttj_records] == [f"{BASE}/trump-tower-jeddah"]
