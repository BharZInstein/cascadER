# Training a two-stage model

The released weights are ready for inference. Training requires your own labeled
records with the schema in the README. Training file names are `train_source1.tsv`,
`train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`. Ground truth
has columns `source1_entity_id` and `matched_entity_ids`; the latter contains a
comma-separated list or an empty string. Include ground truth for every reference.

The following recipe trains the smaller 108-feature model on a deterministic sample.
It does not exactly reproduce the released model's expanded training set. The
original data is not included. Sample intervals must contain enough references,
positive pairs, and negative pairs to train each fold and evaluate both splits.

## Build training indexes and base features

Run from the repository root with the virtual environment active. Set `TRAIN_DATA`
to your training directory and `INFERENCE_DATA` to an optional unlabeled catalog.

```sh
TRAIN_DATA=/path/to/train
mkdir -p artifacts
python src/sample_anchors.py --train-dir "$TRAIN_DATA" --out artifacts/sample --hash-start 125 --hash-stop 625
python src/pipeline.py index --sources "$TRAIN_DATA/train_source2.tsv" "$TRAIN_DATA/train_source3.tsv" --db artifacts/train.sqlite
python src/retrieval_extras.py --db artifacts/train.sqlite --out artifacts/train.sqlite.extra.sqlite
python src/reference_index.py --source "$TRAIN_DATA/train_source1.tsv" --out artifacts/references.sqlite
python src/precompute_competition.py --db artifacts/train.sqlite --references artifacts/references.sqlite --out artifacts/competition --workers 4
python src/cache_training.py --data artifacts/sample --db artifacts/train.sqlite --out artifacts/base --top-k 80 --max-block 1000 --workers 4
```

`cache_training.py` assigns group splits using `bucket('split:' + reference_id, 10)`:
0–5 train, 6–7 tune, 8–9 reserved. Keep the reference groups disjoint between splits.
Indexing all unlabeled target records and reference texts is separate from fitting
on labels. No ground-truth matches are inserted into the candidate list.

## First stage and contextual features

```sh
python src/crossfit_scores.py --cache artifacts/base --out artifacts/crossfit --trees 800 --depth 7
python src/build_eid_lookup.py --db artifacts/train.sqlite --out artifacts/lookup.sqlite
python src/augment_parallel.py --cache artifacts/base --lookup artifacts/lookup.sqlite --competition artifacts/competition --posterior-dir artifacts/crossfit --out artifacts/context --workers 4
python src/robust_features.py --db artifacts/train.sqlite --out artifacts/text_weights.joblib
python src/augment_robust.py --cache artifacts/context --lookup artifacts/lookup.sqlite --weights artifacts/text_weights.joblib --out artifacts/robust --workers 4
```

The base classifier generates group out-of-fold probabilities for training references.
The final base classifier, fitted on all training references, scores tuning and
reserved references. `augment_parallel.py` builds 86 features; `augment_robust.py`
adds 22 text features for a total of 108.

Character weights use every 97th indexed target, with `min_df=2`. Small datasets may
not provide enough sampled text for fitting. The released vectorizers used both
training and unlabeled inference catalogs. To use that transductive setup, build a
separate inference SQLite index and pass both database paths to `robust_features.py
--db`. Decide this protocol before evaluation, and document whether inference text
was used. The recipe above uses only training text.

## Fit, select, and evaluate

```sh
python src/fit_cached.py --cache artifacts/robust --out artifacts/my-model --skip-holdout --experiments 1200:8,1800:7
python src/evaluate_cached_model.py --cache artifacts/robust --model artifacts/my-model
```

Selection uses tuning macro F₀.₅, including threshold and country-policy searches.
The separate evaluation command reports results on reserved groups. Do not repeatedly
choose new models based on this reserved score; use a new untouched evaluation set
after further development. The supplied `pipeline.py train/predict` entry points are
legacy base-feature paths; use `fit_cached.py` and `predict_parallel.py` for 108 features.

## Expanded training used for the released weights

The released model added 132,420 training-only references to the initial 66,400.
The additional references came from hash interval [675,1675), restricted to split
buckets 0–5. The initial tuning and reserved groups were unchanged.

To implement the same procedure on your own dataset:

1. Select a disjoint reference sample and retain only training split buckets.
2. Build its 37-feature cache against the same target index.
3. Use the first-stage model from the original sample to predict the additional
   pairs. Write those scores to `crossfit_scores.npy` in a posterior directory with
   copies of `base_model.ubj` and `base_config.json`. These predictions are out of
   sample because the base model has not trained on the added references.
4. Run both augmentation steps with the same competition cache and text weights.
5. Merge the original and additional 108-feature caches:

```sh
python src/merge_training_caches.py --cache artifacts/robust artifacts/additional-robust --out artifacts/merged
python src/fit_cached.py --cache artifacts/merged --out artifacts/expanded-model --skip-holdout --experiments 1200:8,1800:7
```

The merge script keeps all retrieved positives and all negatives with first-stage
score >= 0.001. It samples remaining training negatives at probability 0.05 with
weight 20. All tuning and reserved pairs are retained. Its seeded sampling is
deterministic for a fixed cache ordering. The released model selected 1,800 trees
at depth 7, then received a separate final evaluation on hash interval [625,675).

Training caches can be much larger than model weights: 108 float32 values require
432 bytes per candidate pair before labels, IDs, and temporary arrays. Plan disk
space for every retained intermediate cache. Training and inference use CPU XGBoost;
the Apple GPU is not used by this implementation.
