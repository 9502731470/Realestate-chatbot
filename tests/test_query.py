import pytest

from app.rag.query import QueryParser, Vocabulary

VOCAB = Vocabulary(
    districts={"narjis": ("An-Narjis", "Riyadh"), "malqa": ("Al Malqa", "Riyadh")},
    projects=[
        ("trump tower jeddah", "darglobal:project:trump-tower-jeddah", "Trump Tower Jeddah"),
        ("dg1", "darglobal:project:dg1", "DG1"),
    ],
)
P = QueryParser(VOCAB)


@pytest.mark.parametrize(
    "q,expected",
    [
        ("3-bedroom apartments for rent in Riyadh under 80k SAR",
         dict(purpose="rent", property_types=["apartment"], bedrooms=3, max_price=80_000, cities=["Riyadh"])),
        ("villas for sale in Jeddah between 1.5m and 3 million",
         dict(purpose="sale", property_types=["villa"], min_price=1_500_000, max_price=3_000_000, cities=["Jeddah"])),
        ("شقق للإيجار في الرياض أقل من 80 ألف",
         dict(purpose="rent", property_types=["apartment"], max_price=80_000, cities=["Riyadh"])),
        ("فيلا للبيع في جدة 4 غرف", dict(purpose="sale", property_types=["villa"], bedrooms=4, cities=["Jeddah"])),
        ("cheapest land for sale in Al Malqa over 500 sqm",
         dict(purpose="sale", property_types=["land"], min_area=500, districts=["Al Malqa"], sort="price_asc")),
        ("2BR flats to rent in Al Khobar max 4000 per month",
         dict(purpose="rent", bedrooms=2, max_price=48_000, cities=["Al Khobar"])),
        ("at least 4 bedrooms villa", dict(min_bedrooms=4, property_types=["villa"])),
        ("furnished apartment", dict(furnishing="furnished")),
        ("unfurnished apartment", dict(furnishing="unfurnished")),
        ("office for rent in Dammam", dict(property_types=["office"], purpose="rent", cities=["Dammam"])),
    ],
)
def test_filters(q, expected):
    f = P.parse(q).filters
    for key, value in expected.items():
        assert getattr(f, key) == value, (q, key, getattr(f, key))


def test_floor_level_is_not_a_property_type():
    assert "floor" not in P.parse("apartment on the 2nd floor in Jeddah").filters.property_types


def test_three_baths_is_not_a_price():
    f = P.parse("villa with 3 baths in Riyadh").filters
    assert f.max_price is None and f.min_price is None


@pytest.mark.parametrize(
    "q,intent",
    [
        ("hi", "greeting"),
        ("مرحبا", "greeting"),
        ("thanks!", "thanks"),
        ("what can you do?", "help"),
        ("How many villas for rent are listed in Riyadh?", "stats"),
        ("What amenities does DG1 have?", "project_detail"),
        ("Tell me about Trump Tower Jeddah", "project_detail"),
        ("Which DarGlobal projects are in Dubai?", "property_search"),
        ("How do I buy property in Saudi Arabia as a foreigner?", "general"),
        ("apartments for rent in Riyadh", "property_search"),
    ],
)
def test_intents(q, intent):
    assert P.parse(q).intent == intent


def test_language_detection():
    assert P.parse("شقق في جدة").language == "ar"
    assert P.parse("flats in Jeddah").language == "en"


def test_follow_up_inherits_filters():
    history = ["villas for rent in Riyadh under 150k"]
    p = P.parse("what about in Jeddah?", history)
    assert p.is_follow_up and p.intent == "property_search"
    assert p.filters.cities == ["Jeddah"] and p.filters.property_types == ["villa"] and p.filters.purpose == "rent"
    cheaper = P.parse("cheaper ones?", history)
    assert cheaper.filters.sort == "price_asc" and cheaper.filters.cities == ["Riyadh"]


def test_new_topic_is_not_a_follow_up():
    p = P.parse("How do I get a mortgage in Saudi Arabia?", ["villas for rent in Riyadh"])
    assert not p.is_follow_up and p.intent == "general"
    assert p.filters.property_types == []


def test_ordinal_reference():
    p = P.parse("tell me more about the second one", ["villas for rent in Riyadh"])
    assert p.references_previous and p.ordinal == 2
