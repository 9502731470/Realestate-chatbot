"""Offline replicas of darglobal.co.uk and wasalt.sa for end-to-end crawler tests.

Payload *shapes* mirror what the live sites serve in ``__NEXT_DATA__`` (as
inspected in Nov 2026), including the personal-data fields that our parsers
must never extract.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import httpx

GZIP_MAGIC = b"\x1f\x8b"


class CompressedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Body delivered as a raw byte stream, so httpx decodes ``content-encoding``
    in the client exactly as it does for a real network response."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self):
        yield self._data

    async def __aiter__(self):
        yield self._data


def compress_like_a_real_server(request: httpx.Request, resp: httpx.Response) -> httpx.Response:
    """The live sites gzip almost every response; the replicas now do too, so
    every crawler test exercises response decoding (a response copy mislabelled
    as still-compressed once broke each live fetch while the then-uncompressed
    replicas kept passing). Already-gzipped payloads (the ``.gz`` sitemap) are
    served as-is, like a CDN."""
    body = resp.content
    if "gzip" not in request.headers.get("accept-encoding", "").lower() or not body or body[:2] == GZIP_MAGIC:
        return resp
    compressed = gzip.compress(body)
    headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() != "content-length"]
    headers += [("content-encoding", "gzip"), ("content-length", str(len(compressed)))]
    return httpx.Response(resp.status_code, headers=headers, stream=CompressedStream(compressed))


DG_ROBOTS = """User-agent: *
Disallow: /sitemap_index.xml
Disallow: /*utm_source=
Disallow: /*?s=
Allow: /$
Allow: /blog/
Allow: /press/
Allow: /projects/
Sitemap: https://darglobal.co.uk/sitemap.xml
"""

WASALT_ROBOTS = """User-agent: ChatGPT-User
Allow: /search

User-agent: *
Allow: /
Disallow: /search
Disallow: /cdn-cgi/rum

Sitemap: https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz
"""

BLOG_ROBOTS = """User-agent: *
Disallow: /wp-admin/
Allow: /wp-admin/admin-ajax.php
"""


def page(next_data: dict[str, Any], title: str = "", h1: str = "", description: str = "") -> str:
    return (
        "<!DOCTYPE html><html><head>"
        f"<title>{title}</title><meta name=\"description\" content=\"{description}\">"
        "</head><body>"
        f"<header><nav>Menu</nav></header><main><h1>{h1}</h1></main>"
        f"<script id=\"__NEXT_DATA__\" type=\"application/json\">{json.dumps(next_data)}</script>"
        "</body></html>"
    )


def nd(page_route: str, page_props: dict[str, Any]) -> dict[str, Any]:
    return {"props": {"pageProps": page_props}, "page": page_route, "query": {}, "buildId": "test"}


GLOBAL = {
    "attributes": {
        "Header": {"HeaderCallBtn": {"phone": "+971 4 000 0000"}},
        "ProjectList": [{"name": "DG1", "country": "UAE"}, {"name": "Trump Tower Jeddah", "country": "Saudi Arabia"}],
        "Footer": {"Copyright": "© DarGlobal"},
    }
}


def dg_sitemap() -> str:
    paths = [
        "", "/about", "/faq", "/projects", "/dg1", "/trump-tower-jeddah", "/careers", "/pay-online",
        "/blog/unlocking-tax-free-investment-oman", "/press/dar-global-h1-2026-results",
        "/insights/branded-residences-2026", "/projects/properties-in-uae",
    ]
    urls = "".join(f"<url><loc>https://darglobal.co.uk{p}</loc><lastmod>2026-11-10</lastmod></url>" for p in paths)
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'


def dg_projects_page_without_data() -> str:
    """Exercise a successful HTML response with no embedded project data."""
    return '<html><head><title>Projects</title></head><body></body></html>'


def dg_projects_page() -> str:
    props = {
        "globalData": GLOBAL,
        "pageData": {
            "attributes": {
                "Title": "Projects",
                "Tabs": [
                    {
                        "TabHeading": "United Arab Emirates",
                        "TabData": [
                            {"Link": "/dg1", "ProjectTitle": "DG1", "Location": "Dubai, UAE", "ProjectType": "Apartments",
                             "completionDate": "December 2027", "Bedrooms": None, "ProjectDescription": None},
                        ],
                    },
                    {
                        "TabHeading": "Saudi Arabia",
                        "TabData": [
                            {"Link": "/trump-tower-jeddah", "ProjectTitle": "Trump Tower Jeddah", "Location": "Jeddah, KSA",
                             "ProjectType": "Apartments", "completionDate": "December 2029", "Bedrooms": None},
                        ],
                    },
                ],
            }
        },
    }
    return page(nd("/projects/[[...params]]", props), title="Projects | DarGlobal", h1="Find your next investment")


def dg_project_page(slug: str) -> str:
    if slug == "dg1":
        attrs = {
            "title": "DG1", "slug": "dg1", "projectCode": "DG1", "updatedAt": "2026-10-26T08:07:20.022Z",
            "publishedAt": "2026-10-26T08:07:19.970Z", "disclaimer": None, "faqContent": [],
            "ProjectDetails": {
                "ProjectBanner": {"Heading": "DG1", "Location": "Dubai, United Arab Emirates", "StartingPrice": "AED 2.2M",
                                  "HeroBanner": {"data": {"attributes": {"url": "https://cdn.darglobal.co.uk/x.mp4"}}}},
                "AboutProject": {
                    "title": "For Visionaries Only.",
                    "description": "An exclusive capsule collection of luxury residences in Business Bay, directly on the Dubai Canal.",
                    "AboutDetailsType": [
                        {"id": 1, "name": "Property Type", "value": "Residential Tower"},
                        {"id": 2, "name": "Status", "value": "Under Development"},
                        {"id": 3, "name": "Unit type", "value": "Studios, 1, 2, & 3-bedrooms "},
                        {"id": 4, "name": "Area (sqm)", "value": "45 to 210"},
                    ],
                },
                "Amenities": {"title": "Amenities", "description": "Spaces for leisure.",
                              "amenitiesList": [{"id": 1, "name": "State-of-the-art Gym", "value": "gym"},
                                                {"id": 2, "name": "Infinity Swimming Pool", "value": "infinity-pool"}]},
                "KeyFeatures": {"description": "DG1 is located by the Dubai Canal in the heart of Business Bay.",
                                "keyfeaturelist": [{"Title": "The Dubai Mall", "Duration": "6 Min"},
                                                   {"Title": "Burj Khalifa", "Duration": "6 Min"}]},
                "whyInvest": {"description": "Dubai is one of the best places in the world to invest.",
                              "investlist": [{"id": 9, "name": "Secure a 10-year Golden Visa", "value": "with a 2 million dirhams investment."}]},
                "loaction": {"description": "Situated at the heart of Business Bay along the Dubai Canal.",
                             "locationDetailsWithImages": {"lat": "25.18", "long": "55.26", "location": "Dubai, United Arab Emirates"}},
                "PaymentMode": {"title": "Payment plan", "description": "Flexible 60/40 payment plan available for this tower."},
                "PhotoGallery": [{"url": "https://cdn.darglobal.co.uk/p.jpg"}],
            },
        }
        title = "Apartments in Business Bay, Dubai for Sale - DG1 by DarGlobal"
    else:
        attrs = {
            "title": "Trump Tower Jeddah", "slug": "trump-tower-jeddah", "projectCode": "TJ",
            "updatedAt": "2026-10-06T06:48:26.984Z",
            "disclaimer": "Trump Tower Jeddah is not owned, developed or sold by The Trump Organization.",
            "faqContent": [{"title": "FAQ", "faqs": [
                {"question": "What is Trump Tower Jeddah ", "answer": [{"type": "paragraph", "children": [{"type": "text", "text": "A 47-floor Trump-branded seafront residential tower on the Jeddah Corniche, developed by DarGlobal."}]}]},
                {"question": "When will Trump Tower Jeddah be completed ", "answer": [{"type": "paragraph", "children": [{"type": "text", "text": "Completion is scheduled for December 2029. Call +966 55 123 4567 for details."}]}]},
            ]}],
            "ProjectDetails": {
                "ProjectBanner": {"Heading": "Trump Tower Jeddah", "Location": "Saudi Arabia", "StartingPrice": None},
                "AboutProject": {"title": "Seafront living", "description": "Seafront residences on the Jeddah Corniche.",
                                 "AboutDetailsType": [
                                     {"id": 1, "name": "Property Type", "value": "Seafront Residences"},
                                     {"id": 2, "name": "Unit Type ", "value": "1, 2, 3, 4 Bedroom Apartments  \n3, 4 Bedroom Penthouses"},
                                     {"id": 3, "name": "Location", "value": "Jeddah Corniche, Saudi Arabia"},
                                     {"id": 4, "name": "Number of Floors", "value": "47"},
                                     {"id": 5, "name": "Area (sqm)", "value": "70 to 365 "}]},
                "Amenities": {"amenitiesList": [{"id": 1, "name": "Private Beach", "value": "beach"}]},
            },
        }
        title = "Trump Tower Jeddah | DarGlobal"
    props = {"globalData": GLOBAL, "projectDetailsData": {"id": 1, "attributes": attrs}, "otherProjectData": {"x": "y"}}
    return page(nd("/[slug]", props), title=title, h1=attrs["title"], description=f"{attrs['title']} by DarGlobal")


def dg_faq_page() -> str:
    props = {
        "globalData": GLOBAL,
        "pageData": {"attributes": {
            "heading": "Frequently Asked Questions",
            "description": "DarGlobal creates luxury properties and branded residences that blend lifestyle, design, and long-term value.",
            "faqContent": [{"description": "Built for a new kind of global investor.", "faqs": [
                {"question": "Where does DarGlobal operate?", "answer": [{"type": "paragraph", "children": [{"type": "text", "text": "Across 14 cities in 9 countries, including Spain, the UAE, Oman, Qatar, Saudi Arabia, Greece and the UK."}]}]},
                {"question": "Who are DarGlobal's partners?", "answer": [{"type": "list", "format": "unordered", "children": [
                    {"type": "list-item", "children": [{"type": "text", "text": "Trump Organization"}]},
                    {"type": "list-item", "children": [{"type": "text", "text": "Pagani"}]}]}]},
            ]}],
        }},
    }
    return page(nd("/faq", props), title="FAQ - Frequently Asked Questions | Darglobal", h1="Frequently Asked Questions")


def dg_about_page() -> str:
    props = {"globalData": GLOBAL, "pageData": {"attributes": {
        "heading": "About DarGlobal",
        "intro": "DarGlobal is an international luxury real estate developer listed on the London Stock Exchange, founded in 2017 as the international arm of Dar Al Arkan.",
        "mission": "We create branded residences in partnership with the world's most iconic lifestyle brands across the Middle East and Europe.",
        "contact": {"email": "info@darglobal.co.uk", "telephone": "+971 4 000 0000"},
    }}}
    return page(nd("/about", props), title="About | DarGlobal", h1="About DarGlobal")


def dg_article(section: str) -> str:
    html_body = (
        "<p>Oman offers a compelling, tax-free environment for international real estate investors.</p>"
        "<h2>Why Oman</h2><p>There is no property tax and no income tax for individuals, and freehold "
        "ownership is available to foreigners in Integrated Tourism Complexes such as AIDA in Muscat.</p>"
        "<p>Investors can also qualify for residency through qualifying property purchases.</p>"
    )
    props: dict[str, Any] = {"globalData": GLOBAL}
    if section == "blog":
        props["blogDetails"] = {"data": {"id": 3, "attributes": {"title": "Unlocking Tax-Free Investment in Oman", "publishedDate": "2026-03-02"}}}
        props["blogDetailsHTML"] = f"<html><body>{html_body}</body></html>"
        props["relatedBlogs"] = [{"title": "Some other blog post that should not be merged into this article body text"}]
        h1 = "Unlocking Tax-Free Investment in Oman"
    elif section == "press":
        props["pressDetails"] = {"attributes": {"Title": "Dar Global H1 2026 results", "date": "2026-08-20",
                                                "content": "Dar Global delivered 66 percent revenue growth in H1 2026 and its portfolio gross development value reached 23 billion dollars. "
                                                           "The company launched new branded projects in Jeddah, Muscat and Doha during the period, and sales momentum remained strong across Europe and the Gulf."}}
        h1 = "Dar Global H1 2026 results"
    else:
        props["insightDetails"] = {"attributes": {"heading": "The future of branded residences",
                                                  "body": [{"type": "paragraph", "children": [{"type": "text", "text": "Branded residences command a premium of 25 to 35 percent over comparable non-branded homes in prime markets, according to industry studies."}]},
                                                           {"type": "paragraph", "children": [{"type": "text", "text": "Buyers value hotel-grade services, design pedigree and the resale liquidity that a globally recognised brand brings to a residential tower."}]}]}}
        h1 = "The future of branded residences"
    return page(nd(f"/{section}/[slug]", props), title=f"{h1} | DarGlobal", h1=h1)


# ------------------------------------------------------------------------ Wasalt
PII = {
    "propertyOwner": {"managedBy": "Ahmad Al Example", "resEmployee": "Ahmad Al Example", "regaAdvLicNo": "7100348130"},
    "leadContactInfo": {"ownerName": "Private Owner Name", "enMessageBody": "To contact Private Owner Name"},
    "contactDetails": {"phone": "+966 55 987 6543", "whatsapp": "https://wa.me/966559876543"},
}


def wasalt_listing(pid: int, purpose: str, subtype: str, city: str, district: str, price: float,
                   beds: int | None, baths: int | None, area: float, furnishing: str = "Un-Furnished") -> dict[str, Any]:
    info: dict[str, Any] = {
        "address": f"{district}, {city}", "city": city, "district": district, "territory": f"North {city}",
        "country": "Saudi Arabia", "currencyType": "SAR", "furnishingType": furnishing, "propertyFor": purpose,
        "propertyMainType": "Residential" if subtype not in ("Office", "Showroom") else "Commercial",
        "propertySubType": subtype, "propertySubTypeSlug": subtype.lower(), "slug": f"{subtype.lower()}-{pid}",
        "title": subtype, "salePrice": price if purpose == "sale" else "", "expectedRent": price if purpose == "rent" else "",
        "rentFreq": {"yearly": {"amount": price, "default_freq": True, "freq": "yearly"}} if purpose == "rent" else "",
    }
    attrs = [{"key": "builtUpArea", "value": area}, {"key": "completionYear", "value": "New"}]
    if beds is not None:
        attrs.append({"key": "noOfBedrooms", "value": beds})
    if baths is not None:
        attrs.append({"key": "noOfBathrooms", "value": baths})
    return {
        "id": pid, "propertyInfo": info, "attributes": attrs, "featureAmenities": [{"name": "Elevator"}, {"name": "Parking"}],
        "isVerified": True, "isRegaProp": True, "ageOfListing": "Updated 2 days ago",
        "publishedAt": "2026-11-10T09:45:56.298Z", "updatedAt": "2026-11-16T09:55:01.444Z",
        "regaMojDesc": "حي النعيم بمدينة جدة", **PII,
    }


def wasalt_project_item(pid: int) -> dict[str, Any]:
    return {
        "id": pid, "projectInfo": {"title": "Shams Al Riyadh ", "slug": f"shams-al-riyadh-{pid}", "city": "Riyadh",
                                   "district": "Al Rihab", "salePriceMin": 2400000, "salePriceMax": 135045300,
                                   "currencyType": "SAR", "propertyFor": "sale", "zone": "East",
                                   "unitTypes": [{"name": "Land", "slug": "residential-land"}]},
        "unitsAvailable": {"available": 47, "total": 47}, "publishedAt": "2026-11-05T11:36:10.291Z", **PII,
    }


def wasalt_category(slug: str, h1: str, items: list[dict[str, Any]], count: int) -> str:
    props = {
        "_sentryTraceData": "x", "metaData": {"h1tag": h1, "title_tag": h1},
        "searchResult": {"count": count, "properties": items, "totalPages": 99},
        "FAQSchema": {"mainEntity": [{"@type": "Question", "name": f"What types of properties are available in {h1}?",
                                       "acceptedAnswer": {"text": "<p>Apartments, villas, lands and floors. Call 0551234567 to ask.</p>"}}]},
        "para": "<h2>Explore " + h1 + "</h2><p>" + ("Riyadh is the capital and largest city of Saudi Arabia, with fast-growing northern districts. " * 6) + "</p>",
        "headerMenuLinks": {"For sale": []},
    }
    return page(nd("/[slug]", props), title=f"{h1} | Wasalt", h1=h1)


def wasalt_category_sitemap() -> bytes:
    paths = [
        "/en/properties-for-sale-in-saudi-arabia", "/en/apartments-for-rent-in-riyadh",
        "/en/villas-for-sale-in-jeddah", "/en/properties-for-sale-in-riyadh-an-narjis",
        "/en/apartments-for-sale-in-riyadh", "/en/properties-for-rent-in-riyadh",
    ]
    xml = '<?xml version="1.0"?><urlset>' + "".join(f"<url><loc>https://wasalt.sa{p}</loc></url>" for p in paths) + "</urlset>"
    return gzip.compress(xml.encode())


def wasalt_detail(pid: int) -> str:
    props = {"propertyDetailsV3": {"id": pid, "propertyInfo": {
        "description": "<p>Un-Furnished apartment for rent in An-Narjis, Riyadh. Built up Area: 150 SQM. For viewing call 0551234567 or email agent@example.com</p>",
        "facingType": "North"},
        "additionalAttributes": [{"key": "propertyMainType", "label": "Property usage", "value": "Residential"},
                                 {"key": "streetWidth", "label": "Street width", "value": "15 m"},
                                 {"key": "regaAdvLicNo", "label": "Advertiser license number", "value": "7100348130"},
                                 {"key": "adSource", "label": "Ad source", "value": "REGA"}],
        **PII}}
    return page(nd("/property/[slug]/[slug-id]", props), title="Apartment for rent")


def wasalt_project_page(pid: int) -> str:
    props = {"compoundDetails": {"data": {"projectInfo": {
        "description": "Shams Al Riyadh is a master-planned community on King Khalid Road in north Riyadh.",
        "completionDate": "2027-06-30T00:00:00.000Z"}, "developerInfo": {"name": "Dar Al Arkan", "about": None}, **PII},
        "status": True}}
    return page(nd("/project/[slug]/[projectId]", props), title="Shams Al Riyadh")


BLOG_POSTS = [
    {"id": 4449, "date": "2026-05-18T15:39:04", "modified": "2026-05-20T10:00:00",
     "link": "https://blog.wasalt.sa/en/how-to-buy-a-house-in-saudi-arabia/",
     "title": {"rendered": "How to Buy a House in Saudi Arabia: A 2026 First-Time Buyer&#8217;s Guide"},
     "content": {"rendered": "<p>Buying a home in Saudi Arabia involves choosing a REGA-licensed broker, securing financing through Sakani or a bank, and registering the deed on the Najiz platform.</p><p>Foreign residents may own property under the updated ownership regulations, subject to conditions.</p>" * 2},
     "excerpt": {"rendered": "<p>A first-time buyer's guide.</p>"}, "categories": [106]},
    {"id": 4400, "date": "2026-04-01T10:00:00", "modified": "2026-04-01T10:00:00",
     "link": "https://blog.wasalt.sa/en/best-restaurants-riyadh/",
     "title": {"rendered": "Best restaurants in Riyadh"},
     "content": {"rendered": "<p>" + "Great food in Riyadh. " * 30 + "</p>"},
     "excerpt": {"rendered": ""}, "categories": [1363]},
]


class MockSites:
    """httpx transport replaying both sites and recording every request."""

    def __init__(self, *, wasalt_blocked: bool = False, darglobal_down: bool = False) -> None:
        self.requests: list[str] = []
        self.wasalt_blocked = wasalt_blocked
        self.darglobal_down = darglobal_down
        listings_rent = [wasalt_listing(5974001 + i, "rent", "Apartment", "Riyadh", d, p, b, 2, a)
                         for i, (d, p, b, a) in enumerate([("An-Narjis", 75000, 3, 150), ("Al Malqa", 95000, 3, 180),
                                                           ("Al Yasmin", 60000, 2, 110), ("An-Narjis", 120000, 4, 220)])]
        listings_sale = [wasalt_listing(5975001 + i, "sale", s, c, d, p, b, 3, a)
                         for i, (s, c, d, p, b, a) in enumerate([("Villa", "Jeddah", "Al Naeem", 2_400_000, 5, 400),
                                                                 ("Villa", "Jeddah", "Obhur Al Shamaliyah", 3_900_000, 6, 520),
                                                                 ("Apartment", "Riyadh", "Al Malqa", 950_000, 3, 160),
                                                                 ("Land", "Riyadh", "Al Rimal", 1_200_000, None, 600)])]
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {
            "https://darglobal.co.uk/robots.txt": (200, {"content-type": "text/plain"}, DG_ROBOTS.encode()),
            "https://darglobal.co.uk/sitemap.xml": (200, {"content-type": "application/xml"}, dg_sitemap().encode()),
            "https://darglobal.co.uk/projects": (200, {}, dg_projects_page().encode()),
            "https://darglobal.co.uk/dg1": (200, {}, dg_project_page("dg1").encode()),
            "https://darglobal.co.uk/trump-tower-jeddah": (200, {}, dg_project_page("ttj").encode()),
            "https://darglobal.co.uk/faq": (200, {}, dg_faq_page().encode()),
            "https://darglobal.co.uk/about": (200, {}, dg_about_page().encode()),
            "https://darglobal.co.uk": (200, {}, dg_about_page().encode()),
            "https://darglobal.co.uk/blog/unlocking-tax-free-investment-oman": (200, {}, dg_article("blog").encode()),
            "https://darglobal.co.uk/press/dar-global-h1-2026-results": (200, {}, dg_article("press").encode()),
            "https://darglobal.co.uk/insights/branded-residences-2026": (200, {}, dg_article("insights").encode()),
            "https://wasalt.sa/robots.txt": (200, {}, WASALT_ROBOTS.encode()),
            "https://cdn.wasalt.sa/robots.txt": (404, {}, b"not found"),
            "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz": (200, {"content-type": "application/x-gzip"}, wasalt_category_sitemap()),
            "https://wasalt.sa/en/properties-for-sale-in-saudi-arabia": (200, {}, wasalt_category("properties-for-sale-in-saudi-arabia", "Properties for Sale in Saudi", [*listings_sale, wasalt_project_item(100325)], 73314).encode()),
            "https://wasalt.sa/en/apartments-for-rent-in-riyadh": (200, {}, wasalt_category("apartments-for-rent-in-riyadh", "Apartments for Rent in Riyadh", listings_rent, 5120).encode()),
            "https://wasalt.sa/en/villas-for-sale-in-jeddah": (200, {}, wasalt_category("villas-for-sale-in-jeddah", "Villas for Sale in Jeddah", listings_sale[:2], 640).encode()),
            "https://wasalt.sa/en/property/rent/apartment-5974001": (200, {}, wasalt_detail(5974001).encode()),
            "https://wasalt.sa/en/project/Riyadh/shams-al-riyadh-100325": (200, {}, wasalt_project_page(100325).encode()),
            "https://blog.wasalt.sa/robots.txt": (200, {}, BLOG_ROBOTS.encode()),
            "https://blog.wasalt.sa/en/wp-json/wp/v2/categories?per_page=100&_fields=id,name": (
                200, {"content-type": "application/json"}, json.dumps([{"id": 106, "name": "Buying Guides"}, {"id": 1363, "name": "Food"}]).encode()),
        }
        fields = "id,date,modified,link,title,content,excerpt,categories"
        self.routes[f"https://blog.wasalt.sa/en/wp-json/wp/v2/posts?per_page=100&page=1&_fields={fields}"] = (
            200, {"content-type": "application/json", "x-wp-totalpages": "1", "x-wp-total": "2"}, json.dumps(BLOG_POSTS).encode())

    def handler(self, request: httpx.Request) -> httpx.Response:
        return compress_like_a_real_server(request, self._handler(request))

    def _handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        host = request.url.host
        if self.wasalt_blocked and host == "wasalt.sa" and not url.endswith("robots.txt"):
            return httpx.Response(403, headers={"content-type": "text/html", "cf-mitigated": "challenge"},
                                  text="<html><title>Just a moment...</title></html>")
        if self.darglobal_down and host == "darglobal.co.uk" and not url.endswith("robots.txt"):
            return httpx.Response(503, text="maintenance")
        route = self.routes.get(url) or self.routes.get(url.rstrip("/"))
        if route is None:
            return httpx.Response(404, text="not found")
        status, headers, body = route
        return httpx.Response(status, headers={"content-type": "text/html; charset=utf-8", **headers}, content=body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
