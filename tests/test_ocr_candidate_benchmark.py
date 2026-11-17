"""Check that benchmark formatting and scoring do not hide recognition errors."""
import pytest

from tools.compare_ocr_candidates import normalize, prepare_unlimited_worker, summarize, visible_text, white_page


def test_unlimited_protocol_metadata_does_not_count_as_read_text():
    raw = '<PAGE><BLOCK>title<LEVEL1>[1,2,3,4]<FORK>ANEXO II\n<BLOCK>text[4,5,6,7]<FORK>DNI 12345678Z'
    assert visible_text(raw, 'unlimited') == 'ANEXO II DNI 12345678Z'


def test_table_boundaries_preserve_separation_and_accents():
    raw = '<PAGE><BLOCK>table[0,0,1000,1000]<FORK><table><tr><td>Uxía</td><td>12345678Z</td></tr></table>'
    assert visible_text(raw, 'unlimited') == 'Uxía 12345678Z'
    assert normalize('Uxía  Rial\nLorenzo') != normalize('Uxia Rial Lorenzo')


def test_truncation_and_empty_transcription_remain_failures():
    rows = [{'group': 'example', 'cer': 1, 'exact': False, 'value_matches': [False], 'seconds': 1,
             'complete': False, 'error': None},
            {'group': 'example', 'cer': 0, 'exact': True, 'value_matches': [True], 'seconds': 3,
             'complete': False, 'error': None}]
    summary = summarize(rows)['example']
    assert summary['failures'] == 2
    assert summary['literal_value_recall'] == 0.5
    assert summary['mean_cer'] == 0.5
    assert summary['median_seconds'] == 2


def test_direct_cuda_adapter_preserves_source_and_manifest(tmp_path):
    source = tmp_path / 'unlimited-ocr.py'
    original = 'model = load(attn_implementation="eager").eval().cuda()\n'
    source.write_text(original, encoding='utf8')
    manifest = tmp_path / 'unlimited-ocr-manifest.json'
    manifest.write_text('{"files": ["pinned"]}', encoding='utf8')
    output = tmp_path / 'output'
    output.mkdir()
    assert prepare_unlimited_worker(source, output, False) == source
    target = prepare_unlimited_worker(source, output, True)
    assert source.read_text(encoding='utf8') == original
    assert target.read_text(encoding='utf8') == 'model = load(attn_implementation="eager", device_map="cuda").eval()\n'
    assert target.with_name(manifest.name).read_bytes() == manifest.read_bytes()
    source.write_text('unrecognized loader', encoding='utf8')
    with pytest.raises(ValueError, match='Unexpected'):
        prepare_unlimited_worker(source, output, True)


def test_white_page_preserves_crop_pixels_without_stretching():
    from PIL import Image
    crop = Image.new('RGB', (8, 2), (10, 20, 30))
    page = white_page(crop)
    assert page.size == (8, 8)
    assert page.getpixel((0, 0)) == (255, 255, 255)
    assert page.crop((0, 3, 8, 5)).tobytes() == crop.tobytes()
    assert crop.size == (8, 2)
