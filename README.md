# OnKith — Local PII Detection: Model Evolution and Evaluation

OnKith is a privacy-first system intended to detect and remove personally
identifiable information (PII) locally, before sensitive text leaves the
device.

**This is the public model-evaluation repository.** The complete OnKith system
lives in a private team repository. This repository documents my model-focused
contribution: model selection, data preparation and label alignment, training,
evaluation, quantization, and deployment-oriented model analysis. It contains
no product code, user data, deployment infrastructure, tokenizers, or trained
model weights.

**Links:** [onkith.online](https://onkith.online/) ·
[LinkedIn](https://www.linkedin.com/company/onkith/)

## Why local detection matters

Redaction protects privacy only when it happens before data is sent elsewhere.
For OnKith, the PII model therefore has to run locally on a CPU while sharing a
limited memory and compute budget with the speech and application pipeline.
Model quality matters, but so do artifact size, latency, memory headroom, and
the behavior of the complete system on its target device.

## My role

I owned the model work represented here: selecting and training the neural
models, preparing the data and label alignment, defining the evaluation
protocols, producing INT8 ONNX variants, and measuring their quality and
efficiency trade-offs. I did not build the deployment integration, audio and
speech-recognition pipeline, or product surface; those are my teammates' work.

## Model evolution

### BiLSTM baseline

An early BiLSTM tagger was trained on a different dataset,
`ai4privacy/pii-masking-300k`, using 57 labels and its own 80/20 split. It
reached 0.605 token-level binary F1. Because both the dataset and metric differ,
that result is historical context rather than a direct comparison with either
transformer model.

### TinyBERT Model V1

`huawei-noah/TinyBERT_General_4L_312D` was the first production-oriented model:
4 transformer layers, hidden size 312, and 67 BIO labels covering 33 entity
types. It was deliberately small because the deployment target was a Raspberry
Pi 5 with other pipeline components competing for the same memory and compute.
Its INT8 ONNX artifact is 13.72 MiB.

TinyBERT remains an important part of the project history. The original
[TinyBERT evaluation notebook](onkith_evaluation.ipynb) contains its detailed
FP32-versus-INT8 evaluation, baselines, per-label results, qualitative errors,
and laptop CPU measurements.

### DeBERTa-v3-xsmall Model V2

As development progressed, the system showed enough resource headroom to
evaluate a larger encoder, while the privacy evaluation suggested that more
model capacity could improve detection quality. Model V2 therefore uses
`microsoft/deberta-v3-xsmall`: 12 layers, hidden size 384, and 63 BIO labels
covering 31 entity types.

The goal was not to maximize F1 without regard to cost. It was to determine
whether a substantially stronger encoder could improve privacy detection while
remaining a realistic candidate for edge deployment. The current evidence
shows a measurable quality improvement and a substantial size and latency cost.
Integration of DeBERTa INT8 into the final pipeline and complete Raspberry Pi 5
validation are still pending.

## Dataset and evaluation protocols

Both transformer models use
[`ai4privacy/pii-masking-openpii-1.5m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m)
by Ai4Privacy / Ai Suisse SA, pinned to revision
`a785eb528e28be2693c3718a27e066970de5dadb`. The dataset card identifies it as
CC BY 4.0.

There are two related but distinct public evaluations:

- **Historical TinyBERT evaluation.** `onkith_evaluation.ipynb` filters the
  official English validation split and excludes one row containing `HEIGHT`, a
  label absent from that model's training data. It scores 40,908 rows under its
  original standalone protocol.
- **Current Model V1 versus Model V2 comparison.**
  `deberta_vs_tinybert_int8_validation.ipynb` evaluates both INT8 ONNX models on
  exactly the same 40,909 English validation rows, in the same order, against
  the same gold spans and metric definitions. This is the canonical source for
  direct TinyBERT-versus-DeBERTa claims.

The current comparison uses exact character-boundary spans from the dataset and
the 31-entity Model V2 ontology. It also repeats the comparison over the 30
entity types shared by both models; neither model emitted an out-of-shared-space
prediction, so the headline result is unchanged.

### Production-equivalent decoding

The comparison uses the correct production-equivalent decoding behavior for
each tokenizer. TinyBERT's WordPiece tokenizer separates punctuation and can
use tokenizer `word_ids()` directly. DeBERTa-v3 uses SentencePiece, where
punctuation may remain grouped with an adjacent word, so its word groups are
derived from character offsets. This prevents trailing punctuation from being
incorrectly absorbed into entity spans while preserving the same BIO repair,
whole-word promotion, span joining, and overlap-resolution rules for both
models.

## TinyBERT INT8 versus DeBERTa INT8

The direct comparison covers 40,909 held-out English validation rows.

| Metric | TinyBERT INT8 | DeBERTa INT8 |
|---|---:|---:|
| Typed precision | 0.9475 | 0.9574 |
| Typed recall | 0.9528 | 0.9601 |
| **Typed F1** | **0.9502** | **0.9588** |
| Binary F1 | 0.9645 | 0.9704 |
| Character recall | 0.9971 | 0.9983 |
| Leakage | 0.0029 | 0.0017 |
| Overmasking | 0.0044 | 0.0029 |
| Exact-row rate | 0.7413 | 0.7844 |
| Model size | 13.72 MiB | 78.47 MiB |
| Desktop single-thread median latency | 14.97 ms | 76.33 ms |

DeBERTa-v3-xsmall INT8 improves typed F1 by 0.0086 absolute while reducing
leakage from 0.0029 to 0.0017. It also reduces overmasking and raises exact-row
correctness by 0.0431. The cost is a **5.72× larger** INT8 artifact and
approximately **5.1× higher** measured single-thread median latency.

The latency figures are local desktop CPU measurements over 500 examples after
warm-up, with one inference thread. They are useful for comparing the two
artifacts under the same conditions, but they are not Raspberry Pi measurements
and do not include a final integrated pipeline benchmark.

## Per-label findings

Among the 20 entity types with at least 50 gold examples, the comparison reports
zero F1 regressions. The largest supported improvements are for `TIME`,
`SOCIALNUM`, `GIVENNAME`, `PASSPORTNUM`, `SURNAME`, and
`DRIVERLICENSENUM`.

Seven types have fewer than 50 gold spans in this split, with only one to five
examples each, so their apparent changes are not strong evidence. The executed
comparison notebook contains the complete support counts and per-label table.

## Evaluation notebooks

### Current direct comparison

[`deberta_vs_tinybert_int8_validation.ipynb`](deberta_vs_tinybert_int8_validation.ipynb)
is the primary evidence artifact for the Model V1 versus Model V2 comparison.
Its committed outputs include:

- the complete executed comparison over the shared validation rows;
- aggregate typed, binary, character, leakage, and exact-row metrics;
- shared-ontology and per-label analysis;
- model-size and local runtime comparisons;
- the full evaluation and tokenizer-aware decoder implementation; and
- interpretation and limitations.

The notebook does not contain or download model weights. Its stored outputs are
still useful to readers who do not have access to the private/team artifacts.

### Historical TinyBERT evaluation

[`onkith_evaluation.ipynb`](onkith_evaluation.ipynb) remains the original detailed
TinyBERT evaluation artifact. The figures and `metrics.json` in [`results/`](results/)
belong to that historical standalone protocol and should not be mixed with the
new direct-comparison metrics.

## Reproducing the comparison

Install the Python dependencies and open the comparison notebook in a Jupyter
environment:

```bash
pip install -r requirements.txt
```

Place these local inputs next to the notebook before running it:

```text
tinybert_int8.onnx
deberta_int8.onnx
tinybert_tokenizer/
deberta_tokenizer/
ai4privacy_validation_en.jsonl
```

The validation JSONL can be rebuilt from the pinned public dataset revision
using the helper defined in the notebook. The normal notebook run expects the
prepared file locally and does not download the full source dataset.

The ONNX models and tokenizers are not distributed in this repository. Trained
weights are a team asset rather than mine alone to publish, and the notebook
fails early with a clear list if any required local artifact is missing.

## Limitations

- The comparison evaluates English data only.
- The AI4Privacy validation distribution is largely synthetic,
  template-generated prose. It does not represent arbitrary real-world speech,
  transcription errors, code-switching, or disfluencies.
- Rare labels remain poorly measured; seven evaluated types have fewer than 50
  gold spans and only one to five examples each.
- The reported timings come from a local desktop CPU. They are not Raspberry Pi
  latency, RAM, sustained-throughput, or thermal measurements.
- Final integration and end-to-end Raspberry Pi 5 testing of the DeBERTa INT8
  pipeline are not complete.
- The comparison measures the neural models without the complete production
  pipeline's deterministic recognizers and policy layer.
- Model weights and tokenizers are not publicly distributed, so rerunning the
  notebook requires locally supplied artifacts.
- Stronger performance on this held-out distribution does not guarantee the
  same improvement on real-world or out-of-distribution input.

## License and attribution

Copyright (c) 2026 Yahya Alsharif. All rights reserved — portfolio and reference
only; see [LICENSE](LICENSE).

Third-party components retain their own licenses. The AI4Privacy dataset is
identified by its dataset card as CC BY 4.0. The TinyBERT base checkpoint's
license remains unverified: its model card declares none, and Huawei's
separately licensed GitHub repository does not demonstrably cover that
checkpoint.
