# GLM-OCR and Unlimited-OCR 2 evaluation

Follow-up: [the completed local rerun](ocr-candidates-2026-10-31-rerun.md) supersedes this incomplete attempt. The original observations below are preserved.

## Decision

No replacement is qualified by this experiment. Keep the production PP-OCRv5
recognizer and optional Qwen3-VL handwriting reader unchanged. The comparison is
**incomplete because candidate inference could not be qualified in this session**;
this is not evidence that either candidate is intrinsically less accurate.

The experiment adds a reproducible local benchmark and checks its scoring. It
does not add a production provider, update bundled models, or publish a release.

## New baseline measurements

Windows x64, NVIDIA RTX 2070 Max-Q with 8 GiB VRAM. PP-OCRv5 recognition ran on CPU;
the full-page baseline also used the CPU/mobile text detector. Twenty samples per
line dataset, selected with the existing benchmark's deterministic seeds, plus
two image-only fictitious document pages were processed successfully.

| Input | Samples | Mean character error | Exact literal value recall |
| --- | ---: | ---: | ---: |
| IAM English handwriting | 20 | 20.36% | Not assessed |
| French census names | 20 | 36.48% | 8/33 name tokens |
| Synthetic Spanish lines in handwriting-style fonts | 20 | 8.31% | 12/20 entire lines |
| Fictitious Spanish/Galician printed pages | 2 | 0.163% | 18/18 values |

The clean page had zero character errors. The rotated/blurred page had 0.327%
character error. Both preserved all nine checked values: name, DNI, email, phone,
address, IBAN, reference, birth date, and deadline. These two simple pages are not
representative of arbitrary scans, layouts, tables, or real documents.

Character error is case/punctuation/accent sensitive after collapsing whitespace.
Census CER uses the existing name-only substring distance. Value recall checks
literal substrings with whitespace normalization and case folding, preserving
accents and digits; this is stricter than the older benchmark's detection-type
check. Census tokens and entire synthetic lines have different denominators and
must not be pooled into an overall accuracy score. Synthetic fonts are not real
Spanish handwriting. No end-to-end personal-data detector recall was measured.

## Candidate attempts and blockers

- **Cached GLM-OCR Q8 conversion + llama.cpp b12001:** two IAM crops returned empty
  output despite a successful HTTP stop response. A subsequent full-page attempt
  exceeded the 90-second readiness limit. These are compatibility/operability
  observations, not a comparative accuracy score. The initial receipt incorrectly
  counted an empty stop response as complete; the retained raw output is unchanged,
  and the benchmark now counts empty output as incomplete.
- **Official ggml-org GLM conversion:** downloaded the Q8 decoder and Q8 projector
  at revision `65a42de1148dbed2297e922b5dbc7d9b70c36578`, switched to the embedded
  chat template and image-first prompt, reduced context to 4,096 and allowed a
  five-minute readiness budget. No completed transcription was obtained before
  stopping the resource-constrained trial. This is an interrupted compatibility
  attempt, not a measured model failure or an accuracy result.
- **Unlimited-OCR 2:** used the existing Reglas offline worker and pinned manifest
  for revision `c14071b549f4fcab10909f3c4acc6a2edb366947` in its separate CUDA/Python
  environment. Loading exceeded the 180-second deadline, during artifact
  preparation, before a qualified inference result. No new Unlimited OCR accuracy,
  speed, or peak-memory measurement is available. The harness now permits a
  ten-minute load budget; **that extended attempt was not run**.
- System free physical RAM fell from about 2.1 GiB to about 168 MiB during the
  session. Candidate timings are not a usable speed ranking. Only benchmark-owned
  processes were terminated; unrelated applications were left running.
- A new Qwen baseline was not run under these conditions. The older 60-line Qwen
  results in [the handwriting benchmark](benchmark.md) remain historical evidence,
  not a controlled head-to-head result against these attempted candidates.

## Suitability for Privelto

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) is a compact 0.9B recognition model;
its complete parsing SDK adds PP-DocLayoutV3 for layout. The documented text prompt
is `Text Recognition:`. The [llama.cpp GGUF distribution](https://huggingface.co/ggml-org/GLM-OCR-GGUF)
offers a plausible optional local recognizer without a Python/CUDA dependency.

[Unlimited-OCR 2](https://huggingface.co/baidu/Unlimited-OCR-2) targets long-document
parsing and offers Transformers and SGLang inference. Its long-context design alone
does not establish better preservation of Spanish names, accents, or identifiers.
The locally available adapter uses FP16 and PyTorch attention in a separate CUDA
environment; this is not a drop-in ONNX replacement or a portable installer.

Privelto needs image coordinates to place redactions. The existing PP-OCRv5 pipeline
provides detected line regions and character positions; its handwriting reader
reads within a measured line region. A future GLM experiment can retain that detector
and redact whole uncertain lines when character geometry is unavailable. Generated
Unlimited boxes must not be treated as measured character locations. Full-page
transcription quality alone cannot qualify redaction placement or export verification.

## Reproduction and retained evidence

Use `tools/compare_ocr_candidates.py --help`. The helper does not download candidate
models and takes explicit local runtime/model paths. Its `prepare` mode can download
the public IAM/French census dataset files if not already cached. It generates
fictitious image-only PDFs and matching PNGs; candidate inputs are those PNGs, not
the production PDF import pipeline. Documents are processed locally.

```powershell
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine prepare --n 20
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine ppocr --output eval/ocr-comparison-rerun/ppocr
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine glm --limit 5 --runtime '<llama-server.exe>' --model '<GLM-OCR-Q8_0.gguf>' --projector '<mmproj-GLM-OCR-Q8_0.gguf>' --output eval/ocr-comparison-rerun/glm
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine unlimited --limit 5 --python '<prepared-python.exe>' --worker '<reglas-root>/src/llm/unlimited-ocr.py' --model '<pinned-model-folder>' --output eval/ocr-comparison-rerun/unlimited
```

The Unlimited worker requires its adjacent `unlimited-ocr-manifest.json`; it checks
the model artifacts before adapting custom code. This project does not bundle that
worker or environment. Candidate requests and startup are bounded; failed streams
are not reused. Output directories must be new so earlier evidence is not overwritten.

Raw inputs, SHA-256 identities, baseline transcriptions, partial cached-GLM outputs,
logs and downloaded candidate artifacts are retained under the ignored local
`eval/ocr-candidates-2026-10-31/` directory. The tracked
[receipt](ocr-candidates-2026-10-31.json) records the limited outcome independently.
No user documents were used. No full application build or release qualification was
needed for this benchmark-only change. Scoring/protocol tests and the existing OCR
runtime tests passed (seven tests); Ruff passed on the final harness and tests.
