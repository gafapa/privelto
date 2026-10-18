# Local review workflow experiment — 2026-10-18

Started on 18 October and completed on 19 October 2026 (Europe/Madrid).

## Question and scope

Does bounded role separation improve Privelto's optional local text review enough to justify its cost? This experiment compares workflows after text extraction, using the actual rule/GLiNER base detector and the installed local language model. It does not modify production code, defaults, website or release artifacts.

The inspiration is [agency-agents](https://github.com/msitarzewski/agency-agents/tree/652b2692f8d8ec2b1f1a38e559073e287dfdc469), especially its [Evidence Collector](https://github.com/msitarzewski/agency-agents/blob/652b2692f8d8ec2b1f1a38e559073e287dfdc469/testing/testing-evidence-collector.md) and [Reality Checker](https://github.com/msitarzewski/agency-agents/blob/652b2692f8d8ec2b1f1a38e559073e287dfdc469/testing/testing-reality-checker.md). We borrow role separation and evidence validation. Prompts here are independently written; no external agent scripts are installed or executed. Automatic pessimism, defect quotas and a requirement to reject a first pass are deliberately excluded. A reviewer may correctly find nothing.

## Results and decision

**Keep the current bounded review as the default. These specialist/critic prompts do not justify an agent integration.** Three evaluation passes were completed on the same 28 reserved cases (43 annotated occurrences per pass). Repeated passes are not additional independent documents.

Initial completed pass:

| Arm | Complete spans | Character recall | Character precision | Non-target characters proposed | Calls | Review seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_only` | 31/43 | 60.6% | 75.7% | 193 | 0 | 29.12 |
| `current` | 32/43 | 79.2% | 77.1% | 233 | 28 | 74.01 |
| `single_focused` | 31/43 | 63.9% | 76.7% | 193 | 28 | 21.76 |
| `repeat_three` | 32/43 | 79.2% | 77.1% | 233 | 84 | 197.65 |
| `specialists_critic` | 31/43 | 60.6% | 75.7% | 193 | 57 | 41.60 |

Base-only seconds measure base detection; the other rows measure additional review time. Startup, warm-up and base loading are excluded from review totals. The initial timing totals include strong resource contention and must not be treated as controlled speed estimates.

Confirmation passes (same prompts, corpus, model and shared context; case/arm order differs in the second confirmation):

| Arm | Complete spans, pass 2 / pass 3 | Character recall, pass 2 / pass 3 | Non-target characters, pass 2 / pass 3 | Median review seconds, pass 2 / pass 3 |
| --- | ---: | ---: | ---: | ---: |
| `current` | 32/43 / 32/43 | 79.2% / 79.2% | 233 / 233 | 1.544 / 1.453 |
| `single_focused` | 31/43 / 31/43 | 63.9% / 63.9% | 193 / 193 | 0.350 / 0.312 |
| `repeat_three` | 32/43 / 32/43 | 79.2% / 79.2% | 233 / 233 | 3.778 / 4.007 |
| `specialists_critic` | 31/43 / 31/43 | 60.6% / 60.6% | 193 / 193 | 0.687 / 0.656 |

The current reviewer recovered indirect-identification and belief/union fragments that the specialist workflow did not preserve. In the initial pass, it covered the complete English spouse/mayor relationship in `holdout-en-injection`. It also over-redacted an ordinary public-event date in `holdout-en-negative` and the generic sentence `The child's name is not included` in `holdout-en-family`. Several relationship misses reflect partial phrase coverage or omitted articles, rather than an untouched identifier. The raw per-case spans are necessary to interpret this distinction.

The base detector already flags 5 of 8 benign controls, including general disease names and generic references to people. Every arm inherits those detections. This additive-only comparison cannot establish whether a separate human-reviewed false-positive suppression workflow would help.

The Spanish injection case also exposes a base name-boundary false positive: a finding joins `Campovento` with the following reviewer-instruction line. Its rule `value` is normalized and differs from the literal source slice. Evaluation uses source offsets, not the normalized display value; all arms inherit the same span. Language-model additions were separately checked for literal source agreement. This is another reason to inspect boundaries before adding review roles.

Prioritize a larger independently labelled corpus, context-sensitive base false-positive handling and better boundary handling for indirect identifiers. A future reviewer may propose expansions around existing findings, with explicit source anchoring and human review. Do not silently delete base findings. Evaluate that separately before changing production. The result concerns these fixed workflows and one 4B model; it does not prove that every possible multi-agent design is inferior.

Completed reserved review requests: 591. Confirmation times are also illustrative: concurrent machine activity was not frozen.

| Run | Review requests | Invalid JSON | Non-stop finishes | Critic failures | Maximum total tokens | Owned server stopped |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| holdout | 197 | 0 | 0 | 0 | 364 | True |
| confirmation | 394 | 0 | 0 | 0 | 364 | True |

Seven evaluation/security-contract tests and Ruff checks pass. The application code was not changed. All completed benchmark-owned servers were stopped and cleanup was verified.

Evidence: [readable summary](evidence/review-harness-2026-10-18/summary.json), [initial reserved raw results](evidence/review-harness-2026-10-18/holdout.json.gz), [two confirmation passes](evidence/review-harness-2026-10-18/confirmation.json.gz), [development pilot](evidence/review-harness-2026-10-18/development.json.gz) and [aborted resource run](evidence/review-harness-2026-10-18/resource-aborted.json.gz). The compressed JSON files contain synthetic text, findings and per-request measurements. The summary records their SHA-256 values and the frozen driver, corpus and model hashes.

## Design

| Arm | Review after shared base findings |
| --- | --- |
| `base_only` | No language-model review; reference only. |
| `current` | The production `llm_review.review` function, once. |
| `single_focused` | One independently authored prompt with explicit context, sensitive facts and untrusted-document instructions. |
| `repeat_three` | Three production reviews; each masks earlier proposals. Extra-call control. |
| `specialists_critic` | Indirect-identification and direct/sensitive-data specialists, followed by a candidate-only verifier when there are proposals. Two or three calls. |

All arms share base detections, model, sampling and document text. Experimental arms validate proposed fragments against literal source text. They cannot remove base findings. The verifier may select existing candidate IDs only; malformed output preserves grounded candidates and records failure. Specialists run sequentially against one server. These are fixed workflows, without autonomous planning, tools or external retrieval.

The corpus contains four development cases and 28 reserved evaluation cases in Spanish, English, Galician, French, German, Italian and Portuguese. Topics include direct identifiers, person-linked sensitive facts, identifying relationships/unique roles, benign general information and embedded adversarial instructions. Names and scenarios are invented; email addresses use `example.test`. No user documents are read. The corpus is small and author-labelled, with unequal language representation; this is exploratory evidence, not independent validation.

Development runs were used to check infrastructure and scoring. Prompts and annotations were frozen before the reserved run. Case and arm order are deterministically shuffled to reduce ordering bias, but shared server caching and concurrent machine activity remain timing confounders. A single run cannot establish statistical significance or general superiority.

## Measures

- **Full-span recall:** the fraction of annotated occurrences whose entire non-whitespace span is covered. Every repeated occurrence counts separately. Removing most of an identifying phrase may already reduce identification risk but still fails this strict boundary test.
- **Character recall:** covered annotated non-whitespace characters divided by all annotated characters.
- **Character precision:** annotated characters divided by all proposed redaction characters. Whole-paragraph redactions are penalized. This measures agreement with this corpus's policy, not actual reidentification probability.
- **Negative controls:** benign cases receiving any redaction, including base false positives that every arm inherits.
- **Review cost:** wall time, local request count, token usage and finish reasons. Base detection and model startup are recorded separately. No hosted-model fees apply, but local compute has a cost.

Literal grounding blocks invented fragments; it does not prove that a fragment is personal data. Injection cases are attack probes, not a comprehensive prompt-injection security test. The policy preserves ordinary event dates while removing dates attributed as birth dates.

Coverage is category-agnostic: a covered health span labelled `other` still counts as hidden. Category accuracy is not measured. Whole-span scores must be read alongside character scores and the actual misses: the ground truth includes complete identifying relationship phrases, while partial removal can have a different practical privacy effect. Masking base detections also means a reviewer cannot propose a longer intact literal fragment that crosses an already-masked span. This architectural constraint is shared by the compared arms and may limit indirect-identification improvements.

## Reproduction

Requires the existing hash-verified base and advanced-lite model packs and llama runtime. On the tested Windows machine the language model uses Vulkan GPU inference and GLiNER uses CPU. Production context is 8192 tokens; the completed reserved comparison uses 2048 tokens for these short cases. Temperature is zero, top-k is one, and each review has a 700-token output limit. The `current` arm uses the production function and request parameters with this smaller shared context. Its latency is not a measurement of the default production resource configuration. Requests use the application's proxy-free, redirect-blocking loopback transport. Secrets and private process command lines are not included in evidence.

Observed hardware: Intel Core i7-10750H (6 cores / 12 logical processors), 32 GB system RAM and NVIDIA RTX 2070 Max-Q (8192 MiB VRAM, driver 616.56). Python 3.12.10; ONNX Runtime DirectML 1.24.4, NumPy 2.5.4 and tokenizers 0.23.3. The base PII model SHA-256 is `e756837e13d16ef832f6cd9fca473c2c3dd1eb232cab13e6443d958d608e797b`. This is a 4B-model comparison; the installed 8B pack was not evaluated.

```powershell
uv run pytest tests/test_review_harness_evaluation.py -q
uv run ruff check tools/benchmark_review_harness.py tools/review_harness_corpus.py tests/test_review_harness_evaluation.py
uv run python tools/benchmark_review_harness.py --split development --output tmp/review-harness-pilot.json --max-seconds 900
uv run python tools/benchmark_review_harness.py --split holdout --context 2048 --output tmp/review-harness-holdout.json --max-seconds 3600
uv run python tools/benchmark_review_harness.py --split holdout --context 2048 --repeats 2 --max-seconds 900 --output tmp/review-harness-confirmation.json
```

Raw evidence includes corpus and driver SHA-256 values, model SHA-256, literal synthetic findings, per-case metrics and request diagnostics. A checkpoint is atomically replaced after every arm; a budget failure preserves partial evidence rather than claiming completion. Only the server owned by this process is stopped. Existing unrelated local model services are left running.

The development pilot completed at 8192 tokens. A first reserved run at 8192 was deliberately aborted because of resource pressure while another local model service was present. Its partial evidence is retained separately as `review-harness-resource-aborted.json`; it is excluded from quality and timing aggregates. Only the shared context configuration was changed for the restart; prompts, corpus and scoring were unchanged. The other service was neither stopped nor queried. Resource contention and cache effects remain limitations even after reducing context.

## Qualification boundary

This evaluates short extracted text on one installed 4B model, one Windows machine and a synthetic corpus. It does not test OCR, handwriting, visual identifiers, long-document chunking, PDF export, pixel verification, installer behavior or legal compliance. The annotation policy itself needs independent review before treating these numbers as a release quality gate. Further work should use blinded, independently labelled documents and repeat runs on supported resource profiles before a production change.
