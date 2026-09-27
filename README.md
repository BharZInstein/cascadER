# cascadER

### Two-stage XGBoost for business entity resolution

cascadER links noisy business records to a deduplicated reference catalog. It combines
name and address retrieval, multilingual text comparisons, and two trained XGBoost
classifiers. Each reference can match zero, one, or many records.

**Internal validation: 0.97224 macro F₀.₅ across 11,199 held-out references.**
This is a precision-focused matching score. The original evaluation data is not
distributed, so this result is not independently reproducible from the demo.

## How it works

```mermaid
flowchart LR
    A[Reference and target records] --> B[SQLite candidate retrieval]
    B --> C[Stage 1: 37 features / 800 trees]
    C --> D[Text, context, and competing-reference features]
    D --> E[Stage 2: 108 features / 1800 trees]
    E --> F[Thresholds and unique target assignment]
    F --> G[Matches and scored candidates]
```

- **Candidate retrieval:** up to 80 candidates per reference using names, addresses,
  numeric tokens, compact strings, and phonetic hints.
- **Two-stage matching:** initial probabilities become inputs to a richer classifier.
  Training uses group out-of-fold predictions to reduce stacking leakage.
- **Text features:** local transliteration, fuzzy comparisons, character TF-IDF,
  address numbers, and support from neighboring candidates.
- **Competing references:** each target is compared with other plausible reference
  records; final assignment gives a target to at most one reference.
- **Local execution:** CPU workers, SQLite indexes, and resumable prediction batches.

This is a trained and tuned gradient-boosted system. Both classifiers were trained
from scratch; there are no pretrained neural weights.

## Quick start

Python 3.12; tested on macOS with 16 GB memory. On macOS, install OpenMP with
`brew install libomp` if it is not already available.

```sh
git clone https://github.com/BharZInstein/cascadER.git
cd cascadER
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
```

The demo uses fictional records and the included trained models. It builds indexes,
scores candidates, resolves competing assignments, and validates output structure.
Results appear in `artifacts/demo/output/matching_results.tsv`. The tiny demo is a
functional example, not a measurement of model quality.

## Match your own records

Prepare three UTF-8 TSV files named `test_source1.tsv`, `test_source2.tsv`, and
`test_source3.tsv`. Source 1 must be a deduplicated reference catalog. Each file has:

| Column | Meaning |
|---|---|
| `entity_id` | Unique ID: `S1-123`, `S2-123`, or `S3-123`, matching its source |
| `business_name` | Business or trading name |
| `business_address` | Address; may be empty |
| `country` | Country label, such as `US`, `India`, or `France` |

Use canonical decimal ID suffixes without leading zeros, in the range
`0 <= suffix < 2**30`. The cache encodes these IDs numerically. Map external IDs to
this schema before running and retain the mapping to restore them afterward.
IDs serve as join keys; they are not predictive model features.

```sh
python scripts/match.py --data-dir /path/to/records --work-dir artifacts/my-records --workers 4
```

Use a new work directory for each input dataset. Rerunning with unchanged inputs
resumes prediction. Input changes require rebuilding the indexes. An empty source 3
file with only its header is supported; the combined target catalog must be nonempty.

### Outputs

| File | Contents |
|---|---|
| `output/matching_results.tsv` | One row per reference; comma-separated matched IDs |
| `output/candidate_pairs.tsv` | Every candidate scored for each reference |
| `output/match_probabilities.f32` | Float32 model scores aligned with final matches |
| `validation.json` | Coverage, ID, ownership, and candidate-subset checks |

Empty match lists are valid. Saved scores are model outputs, not calibrated
real-world probabilities. These confidence scores can shift on a new catalog.

## Evaluation

The model and decision thresholds were chosen using 21,736 tuning references, then
evaluated on a separate group of 11,199 references excluded from fitting and selection.

| Measurement | Previous model | cascadER |
|---|---:|---:|
| Macro F₀.₅ | 0.96938 | **0.97224** |
| Link precision | 0.99003 | **0.99248** |
| Link recall | 0.93956 | **0.94370** |
| Singleton accuracy | **0.96618** | 0.96296 |

The measured macro F₀.₅ gain is 0.00286; its paired bootstrap 95% interval is
[0.00185, 0.00395]. These measurements cover US and India labels. French text was
available for unsupervised character-frequency fitting, but French matching quality
was not measured. F₀.₅ is calculated per reference and then averaged, including
references with no true matches.

See [the model card](docs/MODEL_CARD.md) for training details and limitations,
[training instructions](docs/TRAINING.md) for the fitting pipeline, and
[validation results](reports/validation.json) for exact measurements.

## Repository

```text
models/          Both trained classifiers, text vectorizers, configuration, hashes
src/             Retrieval, features, fitting, inference, and validation
scripts/         End-to-end matching and demo entry points
examples/data/   Fictional business records
docs/            Model card and training instructions
reports/         Aggregate internal validation measurements
```

The source dataset, record-level evaluation outputs, and production caches are not
included. The model artifacts total approximately 22 MB. Full-size inference was
run on a 16 GB Apple laptop; disk use and runtime depend on catalog size and whether
training feature caches are retained.

## License

Project code and model artifacts use the [MIT license](LICENSE). Dependencies retain
their own licenses. No source dataset is distributed or relicensed by this repository.
