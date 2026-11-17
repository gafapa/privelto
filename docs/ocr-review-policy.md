# Local OCR review policy

Privelto keeps PP-OCRv5 as its primary OCR and the installed local Qwen3-VL model as its optional second reader. The [candidate comparison](ocr-candidates-2026-10-31-rerun.md) did not justify replacing this combination with GLM-OCR or Unlimited-OCR 2.

With handwriting reading enabled, the second reader receives uncertain lines (the existing 0.985 threshold) and explicit personal fields relevant to the selected detection categories. Recognized Spanish, Galician, and English field labels must have a colon/equal delimiter or appear separately to the left of the value. Label-only lines and confident ordinary prose do not acquire a personal-field review reason. Personal fields are processed first within each page. This heuristic can increase latency and is not a guarantee that every personal field is selected.

The original reading remains available. Differences in letters, accents, case, or digits count as disagreements; only canonical Unicode encoding and layout whitespace are normalized for comparison. Empty and illegible second readings do not erase the original. Individual reader failures preserve completed results and allow subsequent lines to proceed; three consecutive failures stop further reading requests for that analysis. Pending lines are recorded as unavailable.

The local details API includes `ocrReviews`, with original and alternative text, page geometry, selection reasons, and a status: `agreed`, `disagreed`, `illegible`, `error`, or `unavailable`. Findings linked to disagreements or illegible readings expose `ocr_review_ids`. These records belong to the in-memory analysis and are not added to exported verification reports. Document processing remains local.

Both readings are checked for personal data. Alternative readings use the measured whole-line rectangle rather than invented character positions. Distinct detected values sharing a review are retained through deduplication and deferred merging. Existing review selections remain intact. Export removes every enabled region and avoids drawing overlapping replacement labels for linked findings. The mandatory output verifier checks the retained values.

In the review interface, linked findings appear in **Por revisar** even with high OCR confidence. **Comparar lecturas OCR** reveals both readings without replacing the detected value. The control supports keyboard operation and preserves its expanded state when a finding is enabled or disabled. A disagreement alone does not create a personal-data finding; inspect page-level notices and use manual regions for missed data.

## Validation

`tests/test_ocr_review.py` exercises synthetic OCR lines and stubbed local readers, including one-character disagreements, field selection, geometry, partial failures, deferred review, API privacy, and real PDF redaction/verification. These regressions do not establish accuracy on arbitrary real documents.

The focused browser smoke test uses synthetic API responses and no models or documents:

```powershell
node tools/ocr_review_ui_smoke.mjs <path-to-installed-playwright-core>
```

It checks desktop Spanish and mobile German layouts, the high-confidence review filter, keyboard interaction, preserved expansion, empty readings, and safe rendering of long Unicode and HTML-like text. Screenshots are written to the ignored `tmp/ocr-review-ui` directory.
