import asyncio

import pytest

from app.config import Settings
from app.rag.embeddings import HashingEmbedder
from app.scraping.darglobal import parse_project_cards, parse_project_page
from app.scraping.nextdata import extract_next_data
from app.scraping.pipeline import crawl_all, ingest, load_snapshot, write_snapshot
from app.scraping.wasalt import parse_category_page, plan_categories
from tests.fixtures.sites import (
    MockSites,
    dg_project_page,
    dg_projects_page,
    dg_projects_page_without_data,
    wasalt_category,
    wasalt_listing,
)


def test_darglobal_project_parsing():
    cards = parse_project_cards(extract_next_data(dg_projects_page()))
    assert set(cards) == {"/dg1", "/trump-tower-jeddah"}
    rec = parse_project_page("https://darglobal.co.uk/dg1", extract_next_data(dg_project_page("dg1")), dg_project_page("dg1"), cards["/dg1"])
    assert rec.title == "DG1" and rec.kind == "project" and rec.developer == "DarGlobal"
    assert (rec.city, rec.country) == ("Dubai", "United Arab Emirates")
    assert rec.price == 2_200_000 and rec.currency == "AED"
    assert (rec.area_sqm, rec.area_max_sqm) == (45, 210)
    assert rec.status == "Under Development" and rec.completion == "December 2027"
    assert "Infinity Swimming Pool" in rec.amenities
    assert "The Dubai Mall (6 Min)" in rec.text and "Golden Visa" in rec.text
    assert "60/40 payment plan" in rec.text  # unknown sections are harvested too
    assert "cdn.darglobal" not in rec.text


def test_darglobal_faq_and_disclaimer_and_pii():
    html = dg_project_page("ttj")
    rec = parse_project_page("https://darglobal.co.uk/trump-tower-jeddah", extract_next_data(html), html, None)
    assert rec.city == "Jeddah" and set(rec.property_types) == {"penthouse", "apartment"}
    assert "Q: When will Trump Tower Jeddah be completed" in rec.text
    assert "not owned, developed or sold by The Trump Organization" in rec.text
    assert "123 4567" not in rec.text  # phone inside an FAQ answer was scrubbed


def test_wasalt_category_parsing_strips_personal_data():
    items = [wasalt_listing(1, "rent", "Apartment", "Riyadh", "An-Narjis", 75000, 3, 2, 150)]
    parsed = parse_category_page("https://wasalt.sa/en/apartments-for-rent-in-riyadh", extract_next_data(wasalt_category("x", "Apartments for Rent in Riyadh", items, 5120)))
    (rec,) = parsed["listings"]
    assert rec.url == "https://wasalt.sa/en/property/rent/apartment-1"
    assert (rec.purpose, rec.property_type, rec.bedrooms, rec.bathrooms, rec.area_sqm) == ("rent", "apartment", 3, 2, 150)
    assert rec.price == 75000 and rec.rent_period == "year" and rec.price_label() == "75,000 SAR / year"
    blob = rec.model_dump_json()
    for secret in ("Ahmad", "Private Owner", "987 6543", "7100348130", "wa.me"):
        assert secret not in blob
    assert parsed["stat"].extra["count"] == "5120" and parsed["stat"].city == "Riyadh"
    assert parsed["faqs"] and "[contact removed]" in parsed["faqs"][0].text
    assert parsed["guide"] is not None


def test_category_plan_respects_budget_and_priorities():
    paths = [f"/en/{t}-for-{p}-in-{loc}" for t in ("properties", "apartments", "villas") for p in ("sale", "rent")
             for loc in ("saudi-arabia", "riyadh", "riyadh-an-narjis", "riyadh-al-malqa", "jeddah", "jeddah-al-naeem", "tabuk")]
    plan = plan_categories([*paths, "/en/search?x=1", "/en/weird-thing"], budget=10)
    assert len(plan) == 10
    assert plan[0].endswith("/en/properties-for-sale-in-saudi-arabia")
    assert not any("search" in u for u in plan)
    assert any(u.endswith("-in-riyadh") for u in plan) and any(u.endswith("-in-jeddah") for u in plan)


def test_full_crawl_is_compliant(settings: Settings, sites: MockSites):
    crawl = asyncio.run(crawl_all(settings, transport=sites.transport))
    records = crawl["records"]
    kinds = {(r.source, r.kind) for r in records}
    assert ("darglobal", "project") in kinds and ("wasalt", "listing") in kinds and ("wasalt", "article") in kinds
    assert not [u for u in sites.requests if "/search" in u]
    assert not [u for u in sites.requests if u.endswith(("/careers", "/pay-online"))]
    assert all("robots.txt" in u or u.startswith("https://") for u in sites.requests)
    # lifestyle-only blog posts are skipped
    assert not any("restaurants" in r.url for r in records)
    reports = {r["source"]: r for r in crawl["manifest"]["reports"]}
    assert reports["darglobal"]["ok"] and reports["wasalt"]["ok"]
    assert crawl["manifest"]["stale_sources"] == []


def test_darglobal_missing_project_data_uses_sitemap(settings: Settings, sites: MockSites):
    sites.routes["https://darglobal.co.uk/projects"] = (200, {}, dg_projects_page_without_data().encode())
    crawl = asyncio.run(crawl_all(settings, sources=["darglobal"], transport=sites.transport))
    report = crawl["manifest"]["reports"][0]
    assert report["ok"]
    assert "/projects: no __NEXT_DATA__; using sitemap discovery" in report["notes"]
    projects = {record.id for record in crawl["records"] if record.kind == "project"}
    assert projects == {"darglobal:project:dg1", "darglobal:project:trump-tower-jeddah"}


@pytest.mark.parametrize("other_source_snapshot", [False, True])
def test_failed_source_without_previous_records_is_stale(settings: Settings, other_source_snapshot: bool):
    if other_source_snapshot:
        project = parse_project_page("https://darglobal.co.uk/dg1", extract_next_data(dg_project_page("dg1")))
        write_snapshot(settings, [project], {})
    settings.wasalt_max_blog_posts = 0
    crawl = asyncio.run(crawl_all(settings, transport=MockSites(wasalt_blocked=True).transport))
    assert crawl["manifest"]["stale_sources"] == ["wasalt"]
    assert any(record.source == "darglobal" for record in crawl["records"])
    wasalt = next(report for report in crawl["manifest"]["reports"] if report["source"] == "wasalt")
    assert not wasalt["ok"] and wasalt["records"] == 0
    assert not any("carried forward" in note for note in wasalt["notes"])


def test_blocked_source_carries_forward_previous_snapshot(settings: Settings):
    asyncio.run(ingest(settings, HashingEmbedder(), transport=MockSites().transport, min_records=5))
    before, _ = load_snapshot(settings)
    n_listings = sum(1 for r in before if r.kind == "listing")
    blocked = MockSites(wasalt_blocked=True)
    summary = asyncio.run(ingest(settings, HashingEmbedder(), transport=blocked.transport, min_records=5))
    after, manifest = load_snapshot(settings)
    assert "wasalt" in summary["stale_sources"]
    assert sum(1 for r in after if r.kind == "listing") == n_listings  # nothing lost
    wasalt_report = next(r for r in manifest["reports"] if r["source"] == "wasalt")
    assert not wasalt_report["ok"] and any("challenge" in n for n in wasalt_report["notes"])
    # the blog lives on another host and is still crawled
    assert any(u.startswith("https://blog.wasalt.sa/en/wp-json") for u in blocked.requests)
