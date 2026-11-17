"""Regression evidence for dual OCR readings, source geometry and deferred review."""
import copy
import json
from collections import deque

import pymupdf
import pytest

from app.engine import pipeline, verify
from app.engine.ocr.review import normalized_reading, review_reasons
from app.engine.types import Finding, OcrLine, Options
from tests.test_units import StubOcr, StubServices


def line(text, y=100, confidence=0.95, handwritten=True):
    scale = 200 / 72
    return OcrLine(text, tuple(value * scale for value in (100, y, 390, y + 18)), confidence,
                   handwritten=handwritten)


def analyze(monkeypatch, lines, readings, *, deferred=False, categories=None, native_label=None):
    from app.engine.ocr import handwriting

    values = deque(readings)
    calls = []

    class Reader:
        def __init__(self, server):
            pass

        def read(self, image):
            calls.append(image.shape)
            value = values.popleft()
            if isinstance(value, Exception):
                raise value
            return value

    class Services(StubServices):
        def llm(self):
            return object()

    monkeypatch.setattr(handwriting, 'HandwritingReader', Reader)
    options = Options(categories=categories or {'email', 'handwriting'}, ai=False, faces=False, signatures=False,
                      handwriting=True, llm_review=False, propagate=False)
    services = Services(ocr=StubOcr(page_lines=lines))
    with pymupdf.open() as doc:
        page = doc.new_page()
        if native_label:
            page.insert_text((50, 111), native_label, fontsize=11)
        result = pipeline.analyze(doc.tobytes(), 'fictitious.pdf', options, services, defer_advanced=deferred)
    return result, options, services, calls


def test_selection_keeps_uncertain_lines_and_adds_confident_personal_fields():
    confident = OcrLine('Uxía Rial', (100, 100, 200, 120), 0.999)
    assert review_reasons(confident, 'Nome e apelidos: Uxía Rial', {'name'}) == ('personal-field',)
    assert not review_reasons(confident, 'Nome e apelidos: Uxía Rial', {'email'})
    assert not review_reasons(confident, 'An ordinary public sentence', {'name'})
    assert not review_reasons(OcrLine('Nombre y apellidos:', confident.rect, 0.999), 'Nombre y apellidos:', {'name'})
    assert review_reasons(line('ambiguous', confidence=0.98, handwritten=False), '', {'url'}) == ('low-confidence',)
    assert not review_reasons(line('ambiguous'), '', {'face'})


@pytest.mark.parametrize(('text', 'categories'), [
    ('persona@correo.es', {'email'}), ('https://example.es/name', {'name'}),
    ('El nombre del proyecto es Piragua', {'name'}),
])
def test_confident_nonfield_text_is_not_selected_by_label_words(monkeypatch, text, categories):
    candidate = OcrLine(text, (10, 10, 300, 30), 0.999)
    assert not review_reasons(candidate, text, categories)


def test_reading_comparison_preserves_accents_digits_and_case():
    assert normalized_reading(' Uxi\u0301a\nRial ') == normalized_reading('Uxía Rial')
    assert normalized_reading('Uxía') != normalized_reading('Uxia')
    assert normalized_reading('12345678Z') != normalized_reading('12345679Z')
    assert normalized_reading('I') != normalized_reading('l')


def test_generated_character_geometry_covers_the_measured_line():
    original = OcrLine('first', (10, 20, 200, 40), 0.95, char_rects=[(10, 20, 20, 40)] * 5)
    primary = pipeline.ocr_page_text([original])
    alternative = pipeline.ocr_page_text([original], ['maria@example.es'])
    assert primary.boxes[0] == (10, 20, 20, 40)
    assert alternative.rects_for(0, 5) == [original.rect]
    assert original.text == 'first' and original.char_rects[0] == (10, 20, 20, 40)


def test_one_character_email_disagreement_keeps_both_findings_and_verification_targets(monkeypatch):
    original, alternative = 'maria.lopez@correo.es', 'maria.lopeza@correo.es'
    result, _, _, _ = analyze(monkeypatch, [line(original)], [alternative])
    values = {finding.value for finding in result.findings if finding.type == 'email'}
    assert values == {original, alternative}
    assert {target.value for target in verify.protected_targets(result.findings)} == values
    review = result.ocr_reviews[0]
    assert review.original == original and review.alternative == alternative and review.status == 'disagreed'
    assert all(review.id in finding.ocr_review_ids for finding in result.findings if finding.value in values)
    assert any(review.rect in finding.rects for finding in result.findings if finding.value == alternative)
    assert any(name == 'ocr' and original in text.text for name, text in result.sources[0])
    assert any(name == 'ocr-alt' and alternative in text.text for name, text in result.sources[0])
    assert any('Se conservan ambas lecturas' in note for note in result.notes)


def test_high_confidence_field_is_reviewed_without_marking_all_printed_text_as_handwriting(monkeypatch):
    result, _, _, calls = analyze(monkeypatch, [line('Correo: maria.lopez@correo.es', confidence=0.999, handwritten=False)],
                                  ['Correo: maria.lopeza@correo.es'])
    assert len(calls) == 1
    assert result.ocr_reviews[0].reasons == ('personal-field',)
    assert {f.value for f in result.findings if f.type == 'email'} == {'maria.lopez@correo.es', 'maria.lopeza@correo.es'}


def test_native_personal_label_selects_confident_ocr_value(monkeypatch):
    result, _, _, calls = analyze(monkeypatch, [line('Uxia Rial', confidence=0.999, handwritten=False)], ['Uxía Rial'],
                                  native_label='Nome:', categories={'name', 'handwriting'})
    assert len(calls) == 1 and result.ocr_reviews[0].reasons == ('personal-field',)
    assert result.ocr_reviews[0].status == 'disagreed'


def test_neighboring_ocr_label_is_not_itself_sent_for_confident_field_review(monkeypatch):
    label = OcrLine('Correo:', (30 * 200 / 72, 100 * 200 / 72, 90 * 200 / 72, 118 * 200 / 72), 0.999)
    result, _, _, calls = analyze(monkeypatch, [label, line('maria@correo.es', confidence=0.999, handwritten=False)],
                                  ['mariaa@correo.es'])
    assert len(calls) == 1 and len(result.ocr_reviews) == 1
    assert result.ocr_reviews[0].original == 'maria@correo.es'


def test_personal_fields_are_reviewed_first_without_moving_their_text_or_geometry(monkeypatch):
    result, _, _, _ = analyze(monkeypatch, [line('one@correo.es'),
                                           line('Correo: two@correo.es', 150, confidence=0.999, handwritten=False)],
                              ['Correo: twoa@correo.es', 'onea@correo.es'])
    assert [review.original for review in result.ocr_reviews] == ['Correo: two@correo.es', 'one@correo.es']
    alternative = next(text for name, text in result.sources[0] if name == 'ocr-alt')
    assert alternative.text.index('onea@correo.es') < alternative.text.index('twoa@correo.es')
    first = next(f for f in result.findings if f.value == 'onea@correo.es')
    second = next(f for f in result.findings if f.value == 'twoa@correo.es')
    assert first.rects[0][1] < second.rects[0][1]


@pytest.mark.parametrize('reading', ['', '[ilegible]'])
def test_empty_or_illegible_reading_is_visible_and_never_erases_original(monkeypatch, reading):
    result, _, _, _ = analyze(monkeypatch, [line('Correo: maria@correo.es')], [reading])
    assert result.ocr_reviews[0].status == 'illegible'
    assert result.ocr_reviews[0].alternative == reading
    assert any(f.value == 'maria@correo.es' for f in result.findings)
    assert any(f.type == 'handwriting' and f.value is None for f in result.findings)


def test_failure_of_one_line_preserves_completed_readings_and_continues(monkeypatch):
    result, _, _, calls = analyze(monkeypatch, [line('one@correo.es'), line('two@correo.es', 150), line('three@correo.es', 200)],
                                  ['onea@correo.es', RuntimeError('fixture failure'), 'threea@correo.es'])
    assert len(calls) == 3
    assert [review.status for review in result.ocr_reviews] == ['disagreed', 'error', 'disagreed']
    assert {'one@correo.es', 'onea@correo.es', 'two@correo.es', 'three@correo.es', 'threea@correo.es'} <= {
        finding.value for finding in result.findings}
    assert any('fixture failure' in note for note in result.notes)


def test_unavailable_reader_is_bounded_and_every_skipped_line_is_recorded(monkeypatch):
    result, _, _, calls = analyze(monkeypatch, [line(f'person{i}@correo.es', 100 + i * 35) for i in range(6)],
                                  [RuntimeError('offline')] * 3)
    assert len(calls) == 3
    assert [review.status for review in result.ocr_reviews] == ['error'] * 3 + ['unavailable'] * 3
    assert len({finding.value for finding in result.findings}) == 6


def test_missing_advanced_model_retains_selection_evidence(monkeypatch):
    result, options, services, calls = analyze(monkeypatch, [line('maria@correo.es')], [], deferred=True)
    services.llm = lambda: None
    pipeline.advance(result, options, services)
    assert not calls and result.ocr_reviews[0].status == 'unavailable'
    assert result.ocr_reviews[0].alternative is None
    assert any(f.value == 'maria@correo.es' for f in result.findings)


def test_deferred_merge_keeps_disagreement_and_user_decision(monkeypatch):
    target, options, services, _ = analyze(monkeypatch, [line('maria@correo.es')], ['mariaa@correo.es'], deferred=True)
    base_ids = {finding.id for finding in target.findings}
    result = copy.deepcopy(target)
    pipeline.advance(result, options, services)
    target.findings[0].enabled = False
    added = pipeline.merge_advanced(target, result, base_ids)
    assert not target.findings[0].enabled
    assert any(f.value == 'mariaa@correo.es' for f in added)
    assert target.ocr_reviews and target.findings[0].ocr_review_ids
    assert not target.advanced_pending and not target.deferred


def test_unrelated_overlap_deduplication_is_unchanged():
    first = Finding('a', 0, 'name', 'Ana López', [(0, 0, 100, 20)], 0.9, 'ocr')
    second = Finding('b', 0, 'name', 'Ana', [(0, 0, 100, 20)], 0.8, 'ai')
    result = pipeline.Analysis('test', 'test.pdf', b'', [], [first, second], Options(categories={'name'}))
    pipeline._dedupe(result)
    assert result.findings == [second]  # Existing source priority is preserved.


def test_contextual_review_reads_alternative_source_with_conservative_geometry(monkeypatch):
    from app.engine import llm_review

    result, options, _, _ = analyze(monkeypatch, [line('Correo: maria@correo.es')], ['Correo: mariaa@correo.es'])
    seen = []
    monkeypatch.setattr(llm_review, 'review', lambda server, text, known: seen.append(text) or [])
    pipeline._llm_review(result, object(), options.categories, result.add, lambda event: None, lambda: False)
    assert any('maria@correo.es' in text for text in seen)
    assert any('mariaa@correo.es' in text for text in seen)


def test_local_review_api_exposes_readings_without_changing_export_report(monkeypatch):
    from app.api import Doc
    from tests.test_audit_regressions import session

    result, _, _, _ = analyze(monkeypatch, [line('maria@correo.es')], ['mariaa@correo.es'])
    api = session()
    api._state.docs['test'] = Doc('test', 'test.pdf', 0, b'', status='ready', analysis=result)
    data = api.details('test')
    assert data['ocrReviews'][0]['original'] == 'maria@correo.es'
    assert data['ocrReviews'][0]['alternative'] == 'mariaa@correo.es'
    assert data['ocrReviews'][0]['view']
    report = verify.Report()
    report.add(verify.Leak('ocr', 'page 1', 'maria@correo.es', page=0, rects=[result.ocr_reviews[0].rect]))
    exported = json.dumps(api._report_view(result, report.to_json()))
    assert 'maria@correo.es' not in exported and 'mariaa@correo.es' not in exported
    assert 'ocrReviews' not in exported


def test_verifier_checks_alternative_value_even_when_original_is_absent(monkeypatch):
    result, _, _, _ = analyze(monkeypatch, [line('maria@correo.es')], ['mariaa@correo.es'])
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), 'mariaa@correo.es')
        report = verify.verify(pdf.tobytes(), result.findings)
    assert not report.ok
    assert any(leak.value == 'mariaa@correo.es' and leak.check == 'texto' for leak in report.leaks)


def test_export_redacts_all_disagreement_regions_without_overprinting_labels(monkeypatch):
    from app.engine import redact, sanitize

    result, _, _, _ = analyze(monkeypatch, [line('maria@correo.es')], ['mariaa@correo.es'])
    pipeline.assign_tokens(result.findings)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((100, 112), 'mariaa@correo.es')
        redact.apply(pdf, result.findings)
        text = page.get_text()
        assert text.count('[EMAIL') == 1
        assert 'maria' not in text
        report = verify.verify(sanitize.sanitize(redact.render_output(pdf.tobytes())), result.findings)
    assert report.ok
