"""Text normalisation shared by ingestion, indexing and query parsing.

Handles the bilingual (English / Arabic) nature of the data: Arabic-Indic
digits, diacritics, letter-form variants and common prefixes are normalised
so that "الرياض", "الریاض" and "Riyadh" style variants match consistently.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARABIC_CHARS = re.compile(r"[؀-ۿ]")
_LETTERS = re.compile(r"[A-Za-z؀-ۿ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_DIGIT_MAP.update({ord("٫"): ".", ord("٬"): ",", ord("،"): ","})

_LETTER_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ی": "ي", "ک": "ك"})

# Transliterated Arabic article prefixes seen in place names ("Al Naeem", "An-Narjis").
_PLACE_PREFIX = re.compile(r"^(?:al|el|an|ar|as|at|ash|ad|adh|az|ath|aj|ad-|al-)\s+", re.I)

EN_STOPWORDS = frozenset(
    ["a", "an", "and", "any", "are", "as", "at", "be", "best", "by", "can", "could", "do", "does", "for", "from", "get", "give", "have", "how", "i", "in", "is", "it", "its", "me", "my", "near", "need", "of", "on", "or", "our", "please", "show", "tell", "than", "that", "the", "their", "them", "there", "these", "this", "those", "to", "under", "up", "us", "want", "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your", "about", "also", "all", "am", "between", "both", "each", "few", "find", "looking", "more", "most", "other", "some", "such", "into", "over", "very", "just", "like", "much", "many", "list", "options", "option", "available", "property", "properties"]
)
AR_STOPWORDS = frozenset(
    ["في", "من", "على", "الى", "إلى", "عن", "مع", "هل", "ما", "ماذا", "كيف", "كم", "اين", "أين", "التي", "الذي", "هذا", "هذه", "ذلك", "تلك", "او", "أو", "و", "ثم", "انا", "أنا", "نحن", "هو", "هي", "لي", "لك", "عند", "عندكم", "اريد", "أريد", "ابغى", "ابي", "ممكن", "لو", "يا"]
)


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    return text.translate(_LETTER_MAP)


def normalize_for_search(text: str) -> str:
    """Lowercase, unify digits/letters and strip punctuation (keeps word chars)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_arabic(to_ascii_digits(text)).lower()
    text = text.replace("'", "").replace("’", "")
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def place_key(name: str | None) -> str:
    """Canonical key for matching city/district names across spellings."""
    if not name:
        return ""
    key = normalize_for_search(name.replace("-", " "))
    key = _PLACE_PREFIX.sub("", key)
    key = re.sub(r"^ال", "", key)  # Arabic definite article
    return key.strip()


def is_arabic(text: str) -> bool:
    letters = _LETTERS.findall(text or "")
    if not letters:
        return False
    arabic = sum(1 for ch in letters if _ARABIC_CHARS.match(ch))
    return arabic / len(letters) >= 0.3


def clean_ws(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "…"


# Arabic attaches the article and some prepositions to the word ("بالرياض" =
# "in the Riyadh", "للإيجار" = "for the rent"). Stripping these prefixes lets
# keyword search match "الشقق", "بالشقق" and "شقق" to each other.
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def light_stem_ar(token: str) -> str:
    if not token or not _ARABIC_CHARS.match(token[0]):
        return token
    for prefix in _AR_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            return token[len(prefix):]
    return token


def normalize_for_index(text: str) -> str:
    """Text as stored in the keyword (FTS5/BM25) index: normalised + Arabic light stemming."""
    return " ".join(light_stem_ar(tok) for tok in normalize_for_search(text).split())


def search_terms(text: str, max_terms: int = 24) -> list[str]:
    """Content terms for keyword (BM25) search, stopwords removed, order kept."""
    terms: list[str] = []
    for tok in normalize_for_search(text).split():
        if len(tok) < 2 or tok in EN_STOPWORDS or tok in AR_STOPWORDS:
            continue
        tok = light_stem_ar(tok)
        if tok not in terms:
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms
