# OnKith — PII Span Detection: Model Evaluation

OnKith is a privacy-first voice assistant that transcribes speech and strips
personally identifiable information entirely on-device.

**This is the public evaluation repository.** The full OnKith system lives in a
private team repository. This repo covers my own contribution — model selection,
training, evaluation, and performance characterisation of the PII span-detection
model — and contains no product code, no user data, no deployment
infrastructure, and no model weights.

**Links:** [onkith.online](https://onkith.online/) ·
[LinkedIn](https://www.linkedin.com/company/onkith/)

## The problem

Redaction is only a privacy guarantee if it happens before data leaves your
control — a cloud service that promises to delete identifiers has already
received them. So detection must run on-device: on a CPU, inside the memory
budget of a single-board computer already running a speech model. That
constraint, not accuracy alone, drove every decision below.

## My role

I owned model selection, training, evaluation, and quantisation: the base model
choice, label alignment and split construction, the training run, the evaluation
protocol, the INT8 variant, and the trade-off measured here. I did not build the
deployment integration, the audio and speech-recognition pipeline, or the product
surface — those are my teammates' work.

## Model

**Base:** `huawei-noah/TinyBERT_General_4L_312D` (4 layers, hidden 312, 512-token
context), fine-tuned for 67-label BIO token classification. Chosen because the
target is a Raspberry Pi 5 with 4 GB RAM that must hold a speech model in memory
simultaneously — small enough to leave headroom, still a transformer, which span
boundaries need.

**Alternatives considered**

- **BiLSTM tagger — trained, rejected.** An earlier baseline on a *different*
  dataset (`ai4privacy/pii-masking-300k`, 57 labels, own 80/20 split) reached
  0.605 F1 — token-level and binary (PII vs non-PII), a strictly more permissive
  metric than anything below, so it is not comparable to the TinyBERT results and
  is not in the results table.
- **`DataikuNLP/kiji-pii-model-onnx` — rejected, not trained.** Its training data
  is fully documented, so the rejection is on fit: **size** — 63.3 MB quantised
  ONNX plus a 248.9 MB `model.onnx.data` external weights file, ~312 MB against
  our 13.7 MB, roughly 23× on a Pi 5; **languages** — English, German, French,
  Spanish, Dutch, Danish only, no Arabic; **taxonomy** — 26 types under different
  names (`FIRSTNAME`/`PHONENUMBER`/`SSN` vs our
  `GIVENNAME`/`TELEPHONENUM`/`SOCIALNUM`), missing types we cover; plus a
  coreference head adding 7 unused labels at inference cost. Its card describes a
  DistilBERT encoder while its model tree lists `microsoft/deberta-v3-small` — an
  inconsistency noted, nothing inferred from it. Licence: apache-2.0.
- **TinyBERT-6 — rejected without an experiment.** A judgement call: the size
  increase was not worth a probable small accuracy gain given the on-device
  target. It was not trained, benchmarked, or compared.

## Data

`ai4privacy/pii-masking-openpii-1.5m`, revision `a785eb52…`, by **Ai4Privacy /
Ai Suisse SA**, CC BY 4.0. English rows only.

Training and evaluation use **the same source and the same pinned revision**. The
training and model-selection splits were both carved from the official `train`
split (90/10, seed 42, multilabel-stratified). The test split is the official
**`validation`** split — never trained on, never used for selection: 40,909
English rows minus 1 containing `HEIGHT` (a type absent from training), giving
**40,908 scored rows**, 3,725,659 tokens and 311,477 spans.

Schema is **BIO**: 67 labels over 33 entity types, heavily skewed — `GIVENNAME`
(37,702 spans), `DATE` (37,412) and `SURNAME` (32,068) are a third of all spans,
while 4 trained types have zero test support and 9 more have ≤9. The 33 types are
those **present in the English data as filtered**, not the publisher's documented
taxonomy, which headlines 19 labels.

## Results

Entity-level (span) F1 requires the type *and* both boundaries to match exactly;
token-level alone overstates NER-style performance.

| Variant | Entity P | Entity R | **Entity micro-F1** | Entity macro-F1 | Token micro-F1 |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | 0.9527 | 0.9603 | **0.9565** | 0.6605 | 0.9731 |
| INT8 ONNX | 0.9436 | 0.9557 | **0.9496** | 0.6553 | 0.9696 |
| Naive regex baseline | 0.6519 | 0.0983 | 0.1708 | 0.0546 | 0.2906 |
| Majority class (all `O`) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PyTorch FP32 (export sanity check) | 0.9527 | 0.9603 | 0.9565 | — | 0.9731 |

Macro-F1 is low because it weights all 33 types equally and nine score 0.000 on
≤9 spans each; over the 20 types with ≥100 spans it is 0.9578 (FP32) and 0.9502
(INT8). Micro-F1 is the honest summary. Quantisation cost **0.0069 entity F1**
and raised private-character leakage from 0.00272 to 0.00311.

| | On-disk | Peak process memory | Median latency | p95 | Throughput |
|---|---:|---:|---:|---:|---:|
| FP32 | 54.49 MB | 180.4 MB | 27.30 ms | 88.55 ms | 28.8 /s |
| INT8 | 13.72 MB | 140.8 MB | 14.40 ms | 51.72 ms | 52.8 /s |

*Measured on Intel Core i7-13700H (laptop CPU), single-threaded, batch size 1.*
250 timed iterations per variant after 25 warm-up runs. INT8 buys a 74.8% size
reduction and 47% lower median latency for 0.7 points of entity F1.

## Limitations

- **The blind spot: this measures English, synthetic, written text.** Nothing
  here measures Arabic, code-switched, or real speech-transcript input — the
  actual deployment input, so the test split does not match the deployment
  distribution.
- **High per-type F1 can be memorisation.** `EMAIL` scores 0.99, yet with the
  address and sentence fixed and only the domain varying, the model detected
  `gmail.com`, `outlook.com`, `yahoo.com` and missed `example.com`,
  `example.org`, `northwind-consulting.co.uk` — 3 of 6.
- **Failure modes** (FP32 exact on 4 of 8 qualitative examples, INT8 on 3):
  uncued space-separated phone numbers; confusing numeric identifier types (a
  card number tagged `TAXNUM`); merging given name and surname into one span;
  invented identifier formats.
- **Rare types are unmeasured, not good** — nine lack the support to evaluate.
- **Latency is model inference only**, excluding tokenisation and span merging;
  **Raspberry Pi latency was neither benchmarked nor estimated** — all figures
  come from an x86 laptop CPU.
- **Not measured:** sustained throughput, thermal behaviour, accuracy under
  transcription error, confidence calibration.
- A 0.015% gap (46 of 311,477) between the two span-counting paths resolves
  exactly to 45 malformed `AGE` digit-fragments plus 1 merged same-type span.

## Weights

**Weights are not distributed here** — a deliberate choice, as they are a team
asset and not mine alone to publish. The notebook loads from a single `MODEL_DIR`
constant at the top; point it at a local directory holding the ONNX, PyTorch and
tokenizer artefacts and it runs unchanged.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook onkith_evaluation.ipynb
```

Set `MODEL_DIR`, then run all cells. Outputs are committed to the notebook; every
number above is in `results/metrics.json`, with figures in `results/`.

## License

Copyright (c) 2026 Yahya Alsharif. All rights reserved — portfolio and reference
only; see [LICENSE](LICENSE). Third-party components keep their own licences: the
dataset is CC BY 4.0 by **Ai4Privacy / Ai Suisse SA**; the TinyBERT base model's
licence is **UNVERIFIED** — its card declares none, and Huawei's separately
licensed GitHub repository does not demonstrably cover this checkpoint.
