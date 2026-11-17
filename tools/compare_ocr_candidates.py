"""Compare local OCR candidates without changing production defaults.

Inputs are cached public handwriting lines and fictitious Spanish/Galician pages.
Run --help for explicitly supplied local GLM and Unlimited resources.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import queue
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def normalize(text):
    return ' '.join(text.split())


def visible_text(raw, engine):
    if engine == 'unlimited':
        # Strip only known block metadata; retain each block's actual content.
        raw = ' '.join(part.split('<FORK>', 1)[1] if '<FORK>' in part else part
                       for part in raw.replace('<PAGE>', '').split('<BLOCK>'))
    raw = re.sub(r'</(?:td|th|tr|p)>', ' ', raw, flags=re.I)
    raw = re.sub(r'<[^>]+>', '', raw)
    return normalize(html.unescape(raw))


def prepare_unlimited_worker(source, output, direct_cuda):
    """Retain a benchmark-only load variant; never edit the supplied production worker."""
    if not direct_cuda:
        return source
    original = source.read_text(encoding='utf-8-sig')
    pattern = 'attn_implementation="eager").eval().cuda()'
    if original.count(pattern) != 1:
        raise ValueError('Unexpected Unlimited worker load code')
    directory = output / 'adapter'
    directory.mkdir()
    target = directory / source.name
    target.write_text(original.replace(pattern, 'attn_implementation="eager", device_map="cuda").eval()'), encoding='utf8')
    shutil.copyfile(source.with_name('unlimited-ocr-manifest.json'), directory / 'unlimited-ocr-manifest.json')
    return target


def white_page(image):
    """Give an isolated text crop paper context without stretching its characters."""
    from PIL import Image
    image = image.convert('RGB')
    side = max(image.size)
    page = Image.new('RGB', (side, side), 'white')
    page.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return page


def prepare(directory, count):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from tools.benchmark_handwriting import census_names, fetch_rows, synthetic
    from app.engine.system import font

    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for group, rows in [('iam', fetch_rows('iam', count)), ('census', fetch_rows('census', count)),
                        ('synthetic-lines', synthetic(count))]:
        for index, (image, truth) in enumerate(rows):
            path = directory / f'{group}-{index:03}.png'
            image.save(path)
            entries.append({'id': path.stem, 'group': group, 'image': str(path.resolve()), 'sha256': digest(path),
                            'truth': truth, 'values': census_names(truth) if group == 'census' else
                            [truth] if group == 'synthetic-lines' else []})
    rows = [
        'INFORME DE DATOS FICTICIOS',
        'Nome e apelidos: Uxía Rial Lorenzo',
        'DNI: 12345678Z',
        'Correo: uxía.rial@exemplo.gal',
        'Teléfono: 612 345 678',
        'Enderezo: Rúa do Franco 14, Santiago',
        'IBAN: ES91 2100 0418 4502 0005 1332',
        'Referencia: EXP-2026-0099',
        'Fecha de nacimiento: 03/03/1990',
        'ANEXO II. Entregar copia antes del 15/11/2026.',
    ]
    values = ['Uxía Rial Lorenzo', '12345678Z', 'uxía.rial@exemplo.gal', '612 345 678',
              'Rúa do Franco 14, Santiago', 'ES91 2100 0418 4502 0005 1332', 'EXP-2026-0099',
              '03/03/1990', '15/11/2026']
    for variant in ['clean', 'degraded']:
        image = Image.new('RGB', (1240, 1754), 'white')
        draw = ImageDraw.Draw(image)
        face = ImageFont.truetype(font('sans'), 29)
        for index, row in enumerate(rows):
            draw.text((70, 100 + index * 105), row, font=face, fill=(25, 25, 25))
        if variant == 'degraded':
            image = image.rotate(1.3, fillcolor='white').filter(ImageFilter.GaussianBlur(0.9))
        path = directory / f'synthetic-page-{variant}.png'
        image.save(path)
        # Image-only PDF is retained as the source-equivalent scan fixture.
        image.save(path.with_suffix('.pdf'), 'PDF', resolution=150)
        entries.append({'id': path.stem, 'group': 'synthetic-pages', 'image': str(path.resolve()),
                        'sha256': digest(path), 'truth': ' '.join(rows), 'values': values})
    (directory / 'samples.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf8')
    return entries


class Glm:
    def __init__(self, args):
        self.args = args
        self.token = secrets.token_hex(24)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.log = (args.output / 'runtime.log').open('wb')
        command = [str(args.runtime), '-m', str(args.model), '--mmproj', str(args.projector), '--jinja',
                   '--offline', '-c', '4096', '-ngl', '99', '--flash-attn', 'off', '--no-warmup', '--fit', 'off',
                   '--host', '127.0.0.1', '--port', str(port), '--parallel', '1', '--api-key', self.token,
                   '--no-webui']
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('GLM runtime exited; inspect runtime.log')
                try:
                    with self.http.open(self.url + '/health', timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
                time.sleep(0.2)
            raise TimeoutError('GLM readiness timeout')
        except BaseException:
            self.close()
            raise

    def read(self, entry):
        from PIL import Image
        image = Image.open(entry['image']).convert('RGB')
        image.thumbnail((self.args.glm_max_side, self.args.glm_max_side))
        encoded = io.BytesIO()
        image.save(encoded, 'PNG')
        (self.args.output / (entry['id'] + '-input.png')).write_bytes(encoded.getvalue())
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                    base64.b64encode(encoded.getvalue()).decode()}},
                   {'type': 'text', 'text': 'Text Recognition:'}]
        body = {'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'seed': 42,
                'max_tokens': 2048 if entry['group'] == 'synthetic-pages' else 256, 'stream': False}
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        with self.http.open(request, timeout=180) as response:
            data = json.load(response)
        choice = data['choices'][0]
        return choice['message']['content'], choice['finish_reason'] == 'stop'

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


class Unlimited:
    def __init__(self, args):
        self.args = args
        self.workspace = args.output / 'worker'
        self.workspace.mkdir()
        worker = prepare_unlimited_worker(args.worker, args.output, args.unlimited_direct_cuda)
        self.worker_path = worker
        self.log = (args.output / 'runtime.log').open('wb')
        self.lines = queue.Queue()
        self.process = subprocess.Popen([str(args.python), '-u', str(worker), '--model', str(args.model),
                                         '--workspace', str(self.workspace.resolve())],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding='utf8',
                                        env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                             'HF_HOME': str(self.workspace / 'cache'), 'PYTHONIOENCODING': 'utf-8'},
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def receive():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=receive, daemon=True).start()
        self.sequence = 0
        try:
            self.identity = self.request({'operation': 'load'})
        except BaseException:
            self.close()
            raise

    def request(self, payload):
        self.sequence += 1
        self.process.stdin.write(json.dumps({**payload, 'id': self.sequence}) + '\n')
        self.process.stdin.flush()
        line = self.lines.get(timeout=600 if payload['operation'] == 'load' else 180)
        if line is None:
            raise RuntimeError('Unlimited worker exited; inspect runtime.log')
        data = json.loads(line)
        if data['id'] != self.sequence or data.get('error'):
            raise RuntimeError(f'Unlimited response failure: {data}')
        return data['result']

    def read(self, entry):
        image = self.workspace / 'page.png'
        if self.args.unlimited_white_page and entry['group'] != 'synthetic-pages':
            from PIL import Image
            white_page(Image.open(entry['image'])).save(image)
            shutil.copyfile(image, self.args.output / (entry['id'] + '-input.png'))
        else:
            shutil.copyfile(entry['image'], image)
        result = self.request({'operation': 'recognize', 'images': [str(image.resolve())],
                               'maximumTokens': 2048 if entry['group'] == 'synthetic-pages' else 256})
        return result['text'], result['complete']

    def close(self):
        if self.process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'], capture_output=True, check=False)
            self.process.wait(10)
        self.log.close()


def summarize(entries):
    result = {}
    for group in sorted({entry['group'] for entry in entries}):
        rows = [entry for entry in entries if entry['group'] == group]
        values = [value for entry in rows for value in entry['value_matches']]
        result[group] = {'samples': len(rows), 'mean_cer': statistics.mean(entry['cer'] for entry in rows),
                         'exact': sum(entry['exact'] for entry in rows) / len(rows),
                         'literal_value_recall': sum(values) / len(values) if values else None,
                         'value_count': len(values), 'median_seconds': statistics.median(entry['seconds'] for entry in rows),
                         'failures': sum(bool(entry.get('error')) or not entry['complete'] for entry in rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['prepare', 'ppocr', 'qwen4', 'qwen8', 'glm', 'unlimited'], required=True)
    parser.add_argument('--samples', type=Path, default=ROOT / 'eval/ocr-candidates-2026-10-31/samples')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--limit', type=int, help='Use at most this many samples per group')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--pages-only', action='store_true')
    scope.add_argument('--lines-only', action='store_true')
    parser.add_argument('--glm-max-side', type=int, default=1024, help='Bound GLM vision attention memory; retained resized PNGs')
    parser.add_argument('--qwen-context', type=int, default=4096, help='Bound Qwen KV memory for single-line comparisons')
    parser.add_argument('--unlimited-direct-cuda', action='store_true', help='Load weights directly on GPU in a retained worker copy')
    parser.add_argument('--unlimited-white-page', action='store_true', help='Place line crops on square white paper before Unlimited inference')
    for flag in ['runtime', 'model', 'projector', 'python', 'worker']:
        parser.add_argument('--' + flag, type=Path)
    args = parser.parse_args()
    if args.engine == 'prepare':
        print(f'Prepared {len(prepare(args.samples, args.n))} samples')
        return
    from PIL import Image
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from app.engine.verify import substring_distance

    args.output = args.output or args.samples.parent / args.engine
    args.output.mkdir(parents=True, exist_ok=False)
    samples = json.loads((args.samples / 'samples.json').read_text(encoding='utf8'))
    samples.sort(key=lambda entry: (entry['group'] != 'synthetic-pages', entry['id']))
    counts = {}
    selected = []
    for entry in samples:
        group = entry['group']
        if args.pages_only and group != 'synthetic-pages':
            continue
        if args.lines_only and group == 'synthetic-pages':
            continue
        counts[group] = counts.get(group, 0) + 1
        if args.limit is None or counts[group] <= args.limit:
            selected.append(entry)
    samples = selected
    report = {'engine': args.engine, 'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'glm_max_side': args.glm_max_side if args.engine == 'glm' else None,
              'qwen_context': args.qwen_context if args.engine in ['qwen4', 'qwen8'] else None,
              'unlimited_direct_cuda': args.unlimited_direct_cuda if args.engine == 'unlimited' else None,
              'unlimited_white_page': args.unlimited_white_page if args.engine == 'unlimited' else None,
              'qualification': 'Public line crops and fictitious scan fixtures; not production redaction qualification.',
              'resources': {key: {'path': str(getattr(args, key)), 'sha256': digest(getattr(args, key))}
                            for key in ['runtime', 'model', 'projector', 'worker']
                            if getattr(args, key) and getattr(args, key).is_file()}, 'results': []}
    resource = None
    started = time.perf_counter()
    try:
        if args.engine == 'glm':
            resource = Glm(args)
        elif args.engine == 'unlimited':
            resource = Unlimited(args)
            report['model_identity'] = resource.identity
            report['executed_worker'] = {'path': str(resource.worker_path), 'sha256': digest(resource.worker_path)}
        elif args.engine == 'ppocr':
            from app.engine.ocr.engine import OcrEngine
            resource = OcrEngine(False)
        else:
            from app.engine.llm_server import LlamaServer
            from app.engine.ocr.handwriting import HandwritingReader
            from app.engine import models
            prefix = 'vlm-lite' if args.engine == 'qwen4' else 'vlm'
            resource = LlamaServer(str(models.model(prefix).path), str(models.model(prefix + '-mmproj').path),
                                   context=args.qwen_context)
            resource.start()
            report['backend'] = resource.backend
            report['resources'] = {key: {'path': str(path), 'sha256': digest(path)} for key, path in {
                'model': Path(resource.model_path), 'projector': Path(resource.mmproj_path),
                'runtime': models.runtime_dir() / 'llama' / 'llama-server.exe',
            }.items()}
            reader = HandwritingReader(resource)
        report['startup_seconds'] = time.perf_counter() - started
        for entry in samples:
            if digest(entry['image']) != entry['sha256']:
                raise ValueError('Sample changed')
            before = time.perf_counter()
            error = None
            try:
                if args.engine in ['glm', 'unlimited']:
                    raw, complete = resource.read(entry)
                else:
                    rgb = np.asarray(Image.open(entry['image']).convert('RGB'))
                    if args.engine == 'ppocr':
                        raw = ' '.join(line.text for line in resource.read(rgb)) if entry['group'] == 'synthetic-pages' else resource.recognizer.recognize([rgb])[0][0]
                    elif entry['group'] == 'synthetic-pages':
                        from app.engine.llm_server import image_part
                        raw = resource.chat([{'role': 'user', 'content': [image_part(rgb, max_side=2000),
                                             {'type': 'text', 'text': 'Transcribe literalmente todo el texto de esta página. Responde solo con la transcripción.'}]}], max_tokens=2048)
                    else:
                        raw = reader.read(rgb)
                    complete = True
            except Exception as failure:
                raw, complete, error = '', False, f'{type(failure).__name__}: {failure}'
            seconds = time.perf_counter() - before
            reading = visible_text(raw, args.engine)
            truth = normalize(entry['truth'])
            if entry['group'] == 'census':
                cer = sum(substring_distance(value.lower(), reading.lower()) for value in entry['values']) / max(1, sum(map(len, entry['values'])))
            else:
                cer = Levenshtein.distance(truth, reading) / max(1, len(truth))
            complete = complete and bool(reading.strip())
            result = {**entry, 'raw': raw, 'reading': reading, 'complete': complete, 'error': error, 'seconds': seconds,
                      'cer': cer, 'exact': truth == reading,
                      'value_matches': [normalize(value).casefold() in reading.casefold() for value in entry['values']]}
            report['results'].append(result)
            report['summary'] = summarize(report['results'])
            (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            print(f"{args.engine} {entry['id']}: CER={cer:.3f} time={seconds:.2f}s complete={complete} error={error}", flush=True)
            if error and args.engine in ['glm', 'unlimited']:
                # Do not reuse a timed-out or failed stream for the next sample.
                report['aborted'] = error
                (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
                break
    except Exception as failure:
        report['aborted'] = f'{type(failure).__name__}: {failure}'
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        raise
    finally:
        if resource is not None:
            if args.engine in ['glm', 'unlimited']:
                resource.close()
            elif args.engine in ['qwen4', 'qwen8']:
                resource.stop()


if __name__ == '__main__':
    main()
