"""PII scrubbing applied to every scraped text field.

Wasalt's llms.txt explicitly forbids extracting phone numbers, WhatsApp links
or contact details at scale, so we never read owner/agent/contact structures
and additionally redact anything phone/e-mail/licence-number shaped that may
appear inside free-text descriptions.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_WHATSAPP = re.compile(
    r"(?:https?://)?(?:wa\.me|api\.whatsapp\.com|chat\.whatsapp\.com|web\.whatsapp\.com)/\S*",
    re.I,
)
_TEL_LINK = re.compile(r"tel:\+?[\d\-\s()]{6,}", re.I)
# Phone-like digit runs: optional +/00 prefix, 9–14 digits with separators.
# Prices are rendered with thousands separators by our own templates, so a
# bare 9+ digit run inside free text is almost always a phone/licence number.
_PHONE = re.compile(r"(?<![\w,./])(?:\+|00)?\d(?:[\s\-()]?\d){8,13}(?![\w,/])")

REDACTED = "[contact removed]"

# Keys in source JSON that hold personal / contact information. Parsers must
# never read these; the list is also used defensively when harvesting text.
PII_KEYS = frozenset(
    {
        "propertyOwner",
        "leadContactInfo",
        "contactDetails",
        "regaVerifiedInfo",
        "managedBy",
        "resEmployee",
        "ownerName",
        "phone",
        "phoneNumber",
        "mobile",
        "whatsapp",
        "whatsappNumber",
        "email",
        "telephone",
        "contactPoint",
        "regaAdvLicNo",
        "regaAuthorizationNumber",
        "regaAdvertiserNumber",
        "poaNumber",
        "HeaderCallBtn",
    }
)


def sanitize_record(record: Any) -> Any:
    """Defence in depth: scrub every free-text field of a Record before storage.

    Parsers already avoid personal data; this central choke point guarantees
    nothing slips through via summaries, titles or extra attributes.
    """
    record.title = scrub_pii(record.title) or record.title
    record.text = scrub_pii(record.text)
    if record.summary:
        record.summary = scrub_pii(record.summary)
    record.extra = {k: scrub_pii(str(v)) for k, v in record.extra.items() if k not in PII_KEYS}
    record.amenities = [scrub_pii(a) for a in record.amenities]
    return record


def scrub_pii(text: str | None) -> str:
    if not text:
        return ""
    text = _WHATSAPP.sub(REDACTED, text)
    text = _TEL_LINK.sub(REDACTED, text)
    text = _EMAIL.sub(REDACTED, text)

    def _phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return REDACTED if 9 <= len(digits) <= 14 else match.group(0)

    text = _PHONE.sub(_phone, text)
    # Collapse repeated redaction markers.
    return re.sub(rf"(?:{re.escape(REDACTED)}[\s,/]*){{2,}}", REDACTED + " ", text).strip()
