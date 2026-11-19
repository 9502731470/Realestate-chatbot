"""Helpers for extracting content from Next.js pages and CMS payloads.

Both target sites are Next.js applications that embed their server-side data
in ``<script id="__NEXT_DATA__">``. Parsing that JSON is far more robust than
CSS-selector scraping: it is the exact data the page renders, it survives
visual redesigns, and it avoids executing JavaScript.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from bs4 import BeautifulSoup

from app.scraping.pii import PII_KEYS, scrub_pii
from app.textutil import clean_ws

_NEXT_DATA = re.compile(
    r"<script[^>]*\bid=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>", re.S | re.I
)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_URLISH = re.compile(r"^(?:https?:)?//|^/[\w\-./]*$|^www\.", re.I)
_FILEISH = re.compile(r"\.(?:jpe?g|png|webp|gif|svg|mp4|mov|pdf|avif|ico|css|js)(?:\?.*)?$", re.I)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+Z?)?$")
_HEXISH = re.compile(r"^[0-9a-f\-_]{16,}$", re.I)
_SLUG = re.compile(r"^(?=.*[a-z])[a-z0-9]+(?:[-_][a-z0-9]+)*$")  # e.g. "infinity-pool"

# Keys that never contain useful prose (media, layout, navigation, SEO plumbing).
DEFAULT_SKIP_KEYS = frozenset(
    {
        "globalData", "thankyouData", "thankyouPageData", "otherProjectData", "OtherProject",
        "PhotoGallery", "LiveFeed", "View360", "localizations", "versions", "formats", "SEO",
        "seo", "structuredData", "metaSocial", "metaImage", "metaViewport", "metaRobots",
        "canonicalURL", "keywords", "Header", "Footer", "DesktopMenu", "MobileMenu",
        "CountryList", "url", "href", "link", "Link", "slug", "hash", "ext", "mime",
        "provider", "provider_metadata", "alternativeText", "caption", "createdAt",
        "updatedAt", "publishedAt", "locale", "vuid", "versionComment", "versionNumber",
        "__component", "id", "className", "style", "icon", "Icon", "color", "image", "Image",
        "images", "thumbnail", "video", "videoUrl", "HeroBanner", "HeroBannerMobile",
        "desktopThumbnail", "mobileThumbnail", "Background", "bookunit", "downloadbrochure",
        "chooseYourResidence", "socialLinks", "isMobile", "headerMenuLinks", "footerLinks",
        "homeFooter", "bannersData", "dynamicBannerData", "_sentryTraceData", "_sentryBaggage",
        "reqHeaders", "internalLinks", "breadcrumbsLinks", "remoteConfigCompound",
        "remoteConfigProperty", "lat", "long", "lon", "location",
    }
) | PII_KEYS


def extract_next_data(html: str) -> dict[str, Any] | None:
    match = _NEXT_DATA.search(html or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def page_props(next_data: dict[str, Any] | None) -> dict[str, Any]:
    if not next_data:
        return {}
    props = next_data.get("props") or {}
    return props.get("pageProps") or {}


def html_to_text(html: str) -> str:
    """Convert an HTML fragment/document to readable plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "button"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    block = ["p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "table", "blockquote"]
    for tag in soup.find_all(block):
        tag.insert_after("\n")
    text = soup.get_text(" ")
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return clean_ws(text)


def html_meta(html: str) -> dict[str, str]:
    """Title / description / h1 / og:image from a full HTML page."""
    out: dict[str, str] = {}
    if not html:
        return out
    head = html[:200_000]
    soup = BeautifulSoup(head, "html.parser")
    if soup.title and soup.title.string:
        out["title"] = clean_ws(soup.title.string)
    for name, key in (("description", "description"), ("og:image", "image"), ("og:title", "og_title")):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            out[key] = clean_ws(str(tag["content"]))
    h1 = soup.find("h1")
    if h1:
        out["h1"] = clean_ws(h1.get_text(" "))
    return out


def is_strapi_blocks(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, dict) and "type" in v and ("children" in v or "text" in v) for v in value)
    )


def strapi_blocks_to_text(blocks: Iterable[Any]) -> str:
    """Flatten Strapi v4/v5 rich-text blocks into plain text paragraphs."""

    def inline(node: Any) -> str:
        if isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str):
                return node["text"]
            return "".join(inline(c) for c in node.get("children", []) or [])
        if isinstance(node, list):
            return "".join(inline(c) for c in node)
        return ""

    lines: list[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "list":
            for item in block.get("children", []) or []:
                text = inline(item).strip()
                if text:
                    lines.append(f"- {text}")
        else:
            text = inline(block).strip()
            if text:
                lines.append(text)
    return clean_ws("\n".join(lines))


def rich_to_text(value: Any) -> str:
    """Text from a value that may be a string, HTML, or Strapi blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return html_to_text(value) if _TAG.search(value) else clean_ws(value)
    if is_strapi_blocks(value):
        return strapi_blocks_to_text(value)
    if isinstance(value, list):
        return clean_ws("\n".join(rich_to_text(v) for v in value))
    return ""


def looks_human(text: str, min_len: int = 25) -> bool:
    s = text.strip()
    if len(s) < 3 or _URLISH.search(s) or _FILEISH.search(s) or _ISO_DATE.match(s) or _HEXISH.match(s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.5 * len(s.replace(" ", "")):
        return False
    if len(s) >= min_len:
        return True
    # Short strings only count if they are multi-word phrases.
    return len(s) >= 12 and s.count(" ") >= 2


def harvest_text(
    obj: Any, *, skip_keys: frozenset[str] = DEFAULT_SKIP_KEYS, min_len: int = 25
) -> list[str]:
    """Collect human-readable text from an arbitrary CMS JSON payload.

    Understands common CMS shapes: Q/A pairs, ``{name, value}`` attribute
    pairs, ``{Title, Duration}`` landmark lists and Strapi rich-text blocks.
    Output is de-duplicated and PII-scrubbed, in document order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = scrub_pii(clean_ws(text))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            question = next(
                (node[k] for k in ("question", "Question") if isinstance(node.get(k), str)), None
            )
            answer = node.get("answer", node.get("Answer"))
            if question and answer is not None:
                ans = rich_to_text(answer)
                if ans:
                    add(f"Q: {question.strip()}\nA: {ans}")
                return
            name, value = node.get("name"), node.get("value")
            if isinstance(name, str) and "value" in node and len(node) <= 4:
                sval = ""
                if isinstance(value, str | int | float) and not isinstance(value, bool):
                    sval = str(value).strip()
                if sval and not _SLUG.match(sval):
                    add(f"{name.strip()}: {sval}")
                elif len(name.strip()) >= 3:
                    add(name.strip())
                return
            title, duration = node.get("Title"), node.get("Duration")
            if isinstance(title, str) and isinstance(duration, str) and title.strip():
                add(f"{title.strip()}: {duration.strip()}")
            for key, value in node.items():
                if key in skip_keys:
                    continue
                walk(value)
        elif isinstance(node, list):
            if is_strapi_blocks(node):
                text = strapi_blocks_to_text(node)
                if text:
                    add(text)
                return
            for item in node:
                walk(item)
        elif isinstance(node, str):
            text = node.strip()
            if _TAG.search(text):
                text = html_to_text(text)
            if looks_human(text, min_len=min_len):
                add(text)

    walk(obj)
    return out


def find_first(obj: Any, keys: Iterable[str], *, max_depth: int = 6) -> Any:
    """Depth-limited search for the first non-empty value under any of ``keys``."""
    wanted = set(keys)

    def walk(node: Any, depth: int) -> Any:
        if depth > max_depth:
            return None
        if isinstance(node, dict):
            for key in wanted:
                value = node.get(key)
                if value not in (None, "", [], {}):
                    return value
            for value in node.values():
                found = walk(value, depth + 1)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(node, list):
            for item in node[:50]:
                found = walk(item, depth + 1)
                if found not in (None, "", [], {}):
                    return found
        return None

    return walk(obj, 0)
