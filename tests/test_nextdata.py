from app.scraping.nextdata import (
    extract_next_data,
    harvest_text,
    html_meta,
    html_to_text,
    rich_to_text,
    strapi_blocks_to_text,
)
from tests.fixtures.sites import nd, page


def test_extract_next_data_roundtrip():
    html = page(nd("/x", {"a": 1}), title="T", h1="H", description="D")
    data = extract_next_data(html)
    assert data["props"]["pageProps"] == {"a": 1}
    meta = html_meta(html)
    assert meta["title"] == "T" and meta["h1"] == "H" and meta["description"] == "D"


def test_extract_next_data_missing_or_broken():
    assert extract_next_data("<html></html>") is None
    assert extract_next_data('<script id="__NEXT_DATA__" type="application/json">{broken</script>') is None


def test_strapi_blocks():
    blocks = [
        {"type": "paragraph", "children": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]},
        {"type": "list", "children": [{"type": "list-item", "children": [{"type": "text", "text": "One"}]}]},
    ]
    assert strapi_blocks_to_text(blocks) == "Hello world\n- One"
    assert rich_to_text("<p>Hi <b>there</b></p>") == "Hi there"


def test_html_to_text_drops_scripts_and_keeps_lists():
    text = html_to_text("<div><script>alert(1)</script><p>Para one.</p><ul><li>A</li><li>B</li></ul></div>")
    assert "alert" not in text
    assert "Para one." in text and "- A" in text and "- B" in text


def test_harvest_understands_cms_shapes():
    payload = {
        "globalData": {"menu": "This navigation text should never be harvested at all"},
        "faq": [{"question": "Where?", "answer": [{"type": "paragraph", "children": [{"text": "In Dubai and Riyadh."}]}]}],
        "facts": [{"id": 1, "name": "Status", "value": "Under Development"}, {"id": 2, "name": "Infinity Pool", "value": "infinity-pool"}],
        "landmarks": [{"Title": "The Dubai Mall", "Duration": "6 Min"}],
        "image": {"url": "https://cdn.example.com/a.jpg"},
        "body": "A long enough descriptive sentence about the waterfront residences and their views.",
        "contactPoint": {"telephone": "+971 4 000 0000"},
    }
    out = harvest_text(payload)
    assert "Q: Where?\nA: In Dubai and Riyadh." in out
    assert "Status: Under Development" in out
    assert "Infinity Pool" in out  # slug values are dropped, names kept
    assert "The Dubai Mall: 6 Min" in out
    assert not any("navigation" in t or "cdn.example" in t or "000 0000" in t for t in out)
