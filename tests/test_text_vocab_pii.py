import pytest

from app.models import Record
from app.scraping.pii import REDACTED, sanitize_record, scrub_pii
from app.textutil import is_arabic, normalize_for_search, place_key, search_terms, to_ascii_digits
from app.vocab import (
    canonical_property_type,
    detect_currency,
    find_places,
    parse_area_range,
    parse_location,
    parse_money,
    property_types_in,
)


def test_arabic_normalisation_and_digits():
    assert to_ascii_digits("٣ غرف بـ ٨٠٬٠٠٠") == "3 غرف بـ 80,000"
    assert normalize_for_search("الرِّياض") == normalize_for_search("الرياض")
    assert normalize_for_search("جدة") == "جده"
    assert is_arabic("شقق للإيجار في الرياض")
    assert not is_arabic("Apartments in Riyadh")


@pytest.mark.parametrize(
    "raw,key", [("An-Narjis", "narjis"), ("Al Malqa", "malqa"), ("As Salhiyah", "salhiyah"), ("النرجس", "نرجس")]
)
def test_place_key(raw, key):
    assert place_key(raw) == key


def test_search_terms_drop_stopwords():
    assert search_terms("What amenities does the DG1 tower have?") == ["amenities", "dg1", "tower"]


@pytest.mark.parametrize(
    "text,cities,countries",
    [
        ("villas in Jeddah", ["Jeddah"], []),
        ("شقق في الرياض", ["Riyadh"], []),
        ("بالرياض", ["Riyadh"], []),
        ("projects in Dubai, UAE", ["Dubai"], ["United Arab Emirates"]),
        ("property in Saudi Arabia", [], ["Saudi Arabia"]),
        ("a woman in town", [], []),  # 'oman' must not match inside 'woman'
    ],
)
def test_find_places(text, cities, countries):
    assert find_places(text) == (cities, countries)


def test_parse_location():
    assert parse_location("Jeddah Corniche, Saudi Arabia") == ("Jeddah", "Saudi Arabia")
    assert parse_location("Dubai, United Arab Emirates") == ("Dubai", "United Arab Emirates")
    assert parse_location("Muscat") == ("Muscat", "Oman")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("residential-land", "land"), ("Villas", "villa"), ("Apartment", "apartment"), ("شقة", "apartment"),
        ("Rest House", "rest_house"), ("villa-townhouse", "townhouse"), (None, None), ("Spaceship", None),
    ],
)
def test_canonical_property_type(raw, expected):
    assert canonical_property_type(raw) == expected


def test_property_types_in_label():
    assert property_types_in("1, 2, 3 Bedroom Apartments \n 3, 4 Bedroom Penthouses") == ["penthouse", "apartment"]


@pytest.mark.parametrize(
    "text,amount,currency",
    [
        ("AED 2.5M", 2_500_000, "AED"),
        ("Starting from SAR 1,250,000", 1_250_000, "SAR"),
        ("2,103,408 QAR", 2_103_408, "QAR"),
        ("£1.2m", 1_200_000, "GBP"),
        ("Completion Q4 2029, from AED 3M", 3_000_000, "AED"),  # the year is skipped
        ("70 to 365", None, None),
    ],
)
def test_parse_money(text, amount, currency):
    assert parse_money(text) == (amount, currency)


def test_parse_area_range_and_currency():
    assert parse_area_range("70 to 365 ") == (70.0, 365.0)
    assert parse_area_range("1000 sq.ft")[0] == pytest.approx(92.9, abs=0.1)
    assert detect_currency("budget 80k riyals") == "SAR"


@pytest.mark.parametrize(
    "text,secret",
    [
        ("Call 0551234567 now", "0551234567"),
        ("WhatsApp +966 55 123 4567", "123 4567"),
        ("write to agent@example.com", "agent@example.com"),
        ("https://wa.me/966551234567", "wa.me"),
        ("licence 7100348130", "7100348130"),
        ("tel:+97140000000", "97140000000"),
    ],
)
def test_scrub_pii_removes_contacts(text, secret):
    out = scrub_pii(text)
    assert REDACTED in out
    assert secret not in out


def test_scrub_pii_keeps_prices_and_ids():
    text = "Price 1,200,000 SAR, listing 5974656, built 2024, 3 bedrooms"
    assert scrub_pii(text) == text


def test_sanitize_record_scrubs_every_field():
    rec = Record(
        id="x", source="wasalt", kind="faq", url="https://wasalt.sa", title="Call 0551234567",
        summary="email a@b.co", text="whatsapp https://wa.me/1", extra={"note": "0551234567", "phone": "1"},
    )
    sanitize_record(rec)
    blob = rec.model_dump_json()
    assert "0551234567" not in blob and "a@b.co" not in blob and "wa.me" not in blob
    assert "phone" not in rec.extra
