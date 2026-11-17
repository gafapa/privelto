# OCR candidate rerun: completed local measurements

## Outcome

GLM-OCR now completes all 62 source samples and improves on PP-OCRv5 for sampled
handwriting. This is not a universal improvement over the existing Qwen reader:
the paired tables below show the tradeoffs. Unlimited-OCR 2 also completed 62
samples, but frequently classified handwritten line crops as images without
transcribing them. It was strongest on the synthetic handwriting-style fonts.

Keep production defaults unchanged. Qwen 8B had lower name error (8.84% vs GLM's
9.75%) and lower synthetic-font error (0.56% vs 4.89%). GLM had slightly lower IAM
error (3.46% vs 4.20%), while taking roughly two to three times as long per line.
GLM also improved sampled name accuracy over Qwen 4B (17.92%), but the 4B reader
was perfect on these synthetic font lines and much faster. GLM's smaller weights
remain interesting for an optional accuracy-oriented experiment when 8B cannot
run; these results do not establish a universal replacement. No provider, bundled runtime, release or export verifier
was changed. This follow-up preserves the [incomplete first attempt](ocr-candidates-2026-10-31.md).

## Paired character error

Lower is better. Each line group contains the same 20 source crops. The printed
group contains the same two fictitious image-only scan fixtures. Qwen was tested
only on lines, matching its existing handwriting-reader role. Empty transcriptions
count as errors, rather than being excluded.

| Input | PP-OCRv5 | Qwen3-VL 4B | Qwen3-VL 8B | GLM-OCR Q8 | Unlimited 2 base |
| --- | ---: | ---: | ---: | ---: | ---: |
| IAM English handwriting | 20.36% | 4.39% | 4.20% | 3.46% | 62.56% |
| French census names | 36.48% | 17.92% | 8.84% | 9.75% | 96.79% |
| Spanish handwriting-style fonts | 8.31% | 0.00% | 0.56% | 4.89% | 1.01% |
| Fictitious printed pages | 0.16% | Not tested | Not tested | 0.00% | 0.49% |

Census CER measures names only with the existing substring-distance scorer. Other
CER values are case/accent/punctuation sensitive after whitespace normalization.
The font samples are synthetic, not real Spanish handwriting. Raw Unlimited base
had 19 empty census transcriptions and 12 empty IAM transcriptions. These numbers
describe this line-crop/base-mode integration, not a general model ranking.

## Measured request latency

Median seconds per request, including preprocessing, image I/O and local protocol.
PP-OCRv5 ran on CPU; the other readers ran on the GPU, sequentially. First requests
may include lazy shader compilation (Qwen 4B's first two name requests took 24.4
and 34.7 seconds). Desktop GPU use varied during the session. These are observed
local latencies, not controlled throughput or cold-install benchmarks.

| Input | PP-OCRv5 | Qwen3-VL 4B | Qwen3-VL 8B | GLM-OCR Q8 | Unlimited 2 base |
| --- | ---: | ---: | ---: | ---: | ---: |
| IAM English handwriting | 0.06 s | 0.60 s | 0.87 s | 2.74 s | 7.95 s |
| French census names | 0.06 s | 0.69 s | 1.18 s | 2.29 s | 5.87 s |
| Spanish handwriting-style fonts | 0.02 s | 0.43 s | 0.73 s | 2.21 s | 9.26 s |
| Fictitious printed pages | 2.64 s | Not tested | Not tested | 4.38 s | 88.11 s |

Adapter startup intervals are retained separately in the receipt. Their accounting
differs: Unlimited includes artifact preparation and pinned weight verification;
GLM artifact hashes are computed before server startup. Do not rank startup speed
from those fields alone.

## Literal preservation and failures

The two full pages contain nine checked values each. PP-OCRv5 and GLM preserve
18/18; Unlimited base preserves 15/18, losing accents affecting the name/email.
These simple pages are not representative of arbitrary scans or complex layouts.
Synthetic exact-line accuracy and census name-token recall are distinct checks;
they must not be combined into a personal-data recall score.

- PP-OCRv5: literal census name tokens 8/33; exact synthetic lines 12/20.
- Qwen3-VL 4B: literal census name tokens 19/33; exact synthetic lines 20/20.
- Qwen3-VL 8B: literal census name tokens 22/33; exact synthetic lines 19/20.
- GLM-OCR Q8: literal census name tokens 20/33; exact synthetic lines 10/20.
- Unlimited 2 base: literal census name tokens 0/33; exact synthetic lines 17/20.

GLM can still drop accents, remove identifier spacing, or substitute characters;
for example `maria.lopez@correo.es` became `maria.lopeza@correo.es` on one crop.
Qwen's exact synthetic results do not certify accuracy on real documents. PP-OCRv5
returned one empty IAM transcription, which remains in the CER denominator.

## Unlimited white-paper diagnostic

The model's base processor letterboxes into a square using gray padding. To test
whether isolated-line context caused the omissions, a second profile placed the
first five crops of each line group on square white paper, without stretching or
changing their source pixels. This 15-crop diagnostic is separate from the main
62-sample run. The following table compares exactly those same 15 source crops
for every profile, not the diagnostic against the full 20-crop groups.

| Input | PP-OCRv5 | Qwen3-VL 4B | Qwen3-VL 8B | GLM-OCR Q8 | Unlimited 2 base | Unlimited 2 white paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IAM English handwriting | 24.42% | 2.54% | 2.90% | 2.90% | 81.15% | 3.28% |
| French census names | 32.83% | 5.56% | 5.56% | 2.22% | 100.00% | 43.46% |
| Spanish handwriting-style fonts | 12.01% | 0.00% | 0.00% | 3.58% | 0.87% | 19.44% |

White paper reduced omissions but was not a reliable fix: one census crop remained
empty, another hit the 256-token cap while repeating `2.`, and one Basque name was
transcribed in Cyrillic. All failures and raw outputs remain retained. Single-page
dynamic tiling/Gundam, SGLang and larger GPUs were not tested. Results apply to the
local FP16/eager-attention/base-mode adapter, not all Unlimited deployments.

## Resource fixes and reproduction

- GLM full-resolution input initially failed with a Vulkan allocation of
  8,212,571,136 bytes. The successful profile limits the long edge to 1,024 pixels,
  retains the resized PNGs and uses the official Q8 decoder/projector, embedded
  chat template, image-first `Text Recognition:` prompt, 4,096 context and llama.cpp
  b12001. Artifact revision/hashes are retained in the receipts.
- Unlimited's stock adapter loads the model in CPU RAM before `.cuda()`. That
  attempt exhausted available RAM and was interrupted. The successful benchmark
  copy uses `device_map="cuda"` while preserving pinned weight/code validation,
  FP16, safe coordinate parsing, offline guards and the PyTorch attention fallback.
  The first completed run used a manually retained copy; the harness now provides
  the equivalent `--unlimited-direct-cuda` option without modifying Reglas.
- Qwen 4B/8B use the existing handwriting prompt/preprocessing, Q4_K_M weights,
  GPU inference and a 4,096-token context for these short line requests. The actual
  backend and artifact hashes are in the receipt. No new Qwen full-page OCR profile
  or contextual-review accuracy was qualified.
- Observed total GPU use included the desktop: approximately 4,400–4,700 MiB
  in the GLM samples and 7,700–7,900 MiB with Unlimited. These are sampled
  device totals, not isolated peak tensor measurements. The direct-CUDA Unlimited
  run retained around 9 GiB free system RAM after loading.

```powershell
# Existing identical samples; choose fresh output directories for every run.
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine ppocr --output eval/ocr-next/ppocr
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine glm --glm-max-side 1024 --runtime '<llama-server-b12001.exe>' --model '<GLM-OCR-Q8_0.gguf>' --projector '<mmproj-GLM-OCR-Q8_0.gguf>' --output eval/ocr-next/glm
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine unlimited --unlimited-direct-cuda --python '<prepared-python.exe>' --worker '<reglas-root>/src/llm/unlimited-ocr.py' --model '<pinned-Unlimited-OCR-2-folder>' --output eval/ocr-next/unlimited
# Add --lines-only --limit 5 --unlimited-white-page for the diagnostic subset.
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine qwen4 --lines-only --output eval/ocr-next/qwen4
.venv/Scripts/python.exe tools/compare_ocr_candidates.py --engine qwen8 --lines-only --output eval/ocr-next/qwen8
```

The Unlimited environment was Python 3.12 with PyTorch 2.12.0+cu126, torchvision
0.27.0+cu126, Transformers 4.52.1, Accelerate 1.13.0, tokenizers 0.21.4 and
huggingface-hub 0.36.0. GPU: RTX 2070 Max-Q 8 GiB, driver 617.42, Windows x64.
This does not qualify clean-machine installation or other platforms.

## Evidence and limits

The [machine-readable receipt](ocr-candidates-2026-10-31-rerun.json) records per-sample
CER, literal checks, failures, timings, paired source hashes, processed-image hashes
where retained, and model/runtime/worker identities. Full transcriptions, images,
worker copies and diagnostics remain under `eval/ocr-candidates-2026-10-31-rerun/`
(ignored by git). No user documents or external inference services were used.

Nine benchmark/scoring and OCR runtime tests passed; Ruff passed. Tests cover protocol
metadata removal, strict accent-sensitive normalization, incomplete output scoring,
the source-preserving CUDA adapter and pixel-preserving white-paper input. Successful
transcription does not verify redaction geometry, detector recall, end-to-end export
or arbitrary-document safety. No full application build or publication was performed.
