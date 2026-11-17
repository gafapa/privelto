"""Select uncertain text and personal fields without interpreting confidence as accuracy."""
from __future__ import annotations

import re
import unicodedata

from ..types import OcrLine
from ..rules import SELECTABLE

HANDWRITING_CONFIDENCE = 0.985  # Existing measured threshold; see docs/benchmark.md.
FIELD_KINDS = {
    r"nom(?:bre|e)|apellidos?|apelidos?|name|surname|paciente|alumn[oa]|titular|solicitante": {"name"},
    r"d\.?n\.?i\.?|n\.?i\.?f\.?|n\.?i\.?e\.?|pasaporte|passport": {"dni", "nie", "cif", "passport", "id_other"},
    r"domicilio|direcci[oó]n|enderezo|address|localidad|c[oó]digo\s+postal": {"address"},
    r"tel[eé]fono|m[oó]vil|phone": {"phone"},
    r"e-?mail|correo": {"email"},
    r"iban|cuenta|conta": {"iban"},
    r"bic|swift": {"bic"},
    r"seguridad\s+social|seguridade\s+social|social\s+security|nss": {"nss"},
    r"tarjeta|tarxeta|card": {"card"},
    r"direcci[oó]n\s+ip|ip": {"ip"},
    r"nacimiento|nacemento": {"date"},
}
FIELDS = [(re.compile(r"(?i)(?<![\w@./-])(?:" + pattern + r")(?![\w@./-])"), kinds)
          for pattern, kinds in FIELD_KINDS.items()]
TEXT_CATEGORIES = set(SELECTABLE) - {'face', 'signature', 'manual'}


def normalized_reading(text: str) -> str:
    """Ignore layout whitespace and canonical Unicode encoding; preserve accents, case and digits."""
    return " ".join(unicodedata.normalize('NFC', text).split())


def _without_labels(text: str) -> str:
    for pattern, _ in FIELDS:
        text = pattern.sub('', text)
    return re.sub(r'[\W_]+', '', text).casefold()


def review_reasons(line: OcrLine, context: str, categories: set[str]) -> tuple[str, ...]:
    if not line.text.strip() or not categories & TEXT_CATEGORIES:
        return ()
    reasons = []
    if line.handwritten or line.confidence < HANDWRITING_CONFIDENCE:
        reasons.append('low-confidence')
    label_only = ('', 'y', 'e', 'and')
    external = context.replace(line.text, '', 1).strip()
    separate_label = bool(external) and _without_labels(external) in label_only
    labelled = any(kinds & categories and any(
        separate_label or context[match.end():].lstrip().startswith((':', '：', '='))
        for match in pattern.finditer(context)) for pattern, kinds in FIELDS)
    if _without_labels(line.text) not in label_only and labelled:
        reasons.append('personal-field')
    return tuple(reasons)
