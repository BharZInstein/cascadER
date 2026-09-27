# cascadER model card

## Intended use

Match noisy business names and addresses to a deduplicated reference catalog.
The pipeline assumes that each target belongs to at most one reference and allows
multiple targets per reference. If a catalog violates that assumption, the final
ownership policy needs to change.

## Model

| Component | Configuration |
|---|---|
| First classifier | XGBoost, 800 trees, maximum depth 7, 37 features |
| Final classifier | XGBoost, 1,800 trees, maximum depth 7, 108 features |
| Learning rate | 0.07 for both classifiers |
| Retrieval | At most 80 primary candidates per reference |
| Text representation | Character 3–5-gram TF-IDF for names and addresses |
| Final threshold | 0.725, with tuned top-candidate fallback policies |
| Ownership | Highest-scoring selected reference per target; input-order tie break |

The 108 features comprise 37 basic pair features, 11 rarity features, 14 candidate
context features, 15 first-stage probability/context features, 9 competing-reference
features, and 22 canonical-text/TF-IDF features. The optional auxiliary retrieval
path exists in the code but is disabled in the distributed model configuration.

## Training and selection

The original source is a non-distributed labeled business-record dataset. Names,
addresses, country labels, and record linkage labels were supplied; no external
identity lookup or geocoding was used. Source records are not included here.

- The final classifier used **198,820 training references**.
- **21,736 tuning references** selected model size and decision policies.
- The first classifier used the original **66,400 training references**. Three
  group folds produced out-of-fold probabilities for those references. The full
  first classifier scored the additional 132,420 training references, which it had
  never trained on.
- Candidate retrieval searched the full training target catalog of **10,320,219
  records**. Labeled positives were not injected into retrieval results.
- All retrieved positives and negatives with first-stage score >= 0.001 were kept.
  Other training negatives were sampled at 5% and weighted by 20. Evaluation kept
  all candidates. The resulting training cache had 2,117,895 retained pairs.
- The final comparison used **11,199 previously unused references**, after model
  selection. These references were never used for fitting or threshold tuning.

Deterministic BLAKE2b ID buckets select groups: original sample [125,625), additional
training [675,1675) restricted to training split buckets, and final validation
[625,675), all modulo 10,000. Within the original sample, a separate hash assigns
training/tuning/reserved groups. The original reserved groups remain excluded from
training. These are reference-group splits, not a business-family or time split.

The two TF-IDF vectorizers were fitted on **209,173 unlabeled records sampled from
both training and inference target catalogs**, including French text. This is a
transductive setup: character frequencies see inference text, but not inference
labels. The vocabulary is capped at 200,000 features per field. Vectorizers and
tree weights are shipped together; refitting only the vectorizers changes feature
values and requires reevaluation.

## Evaluation

On the reserved 11,199 references: macro F₀.₅ **0.97223877**, link precision
**0.99248405**, and link recall **0.94370180**. Candidate recall is **0.97038560**.
US macro F₀.₅ is **0.97795785**, India **0.96347719**. Singleton accuracy is
**0.96296296** on 621 true singletons. These metrics are internal measurements;
the underlying evaluation records are not distributed.

For a reference with true matches, F₀.₅ = `1.25*TP / (0.25*n_true + n_pred)`.
A reference with zero true matches scores 1 for an empty prediction and 0 otherwise.
The final score averages reference scores. Link precision and recall aggregate
across pairs and therefore are not interchangeable with the macro metric.

The preceding model scored 0.96937876 on exactly the same reference groups.
The paired gain interval [0.00184850, 0.00395441] comes from 1,000 bootstrap samples
of reference scores (seed 42). It describes sampling variation within this dataset,
not robustness to new countries or data sources. Competing-reference features use
the supplied unlabeled catalog; validation enforces unique ownership within its
evaluated reference groups.

## Limits

- Only US and India have labeled validation. Other countries are unmeasured.
- Same-country blocking may miss links whose country labels disagree.
- A correct match outside the retrieved candidates cannot be recovered by scoring.
- Severe corruption, missing addresses, and closely related businesses remain hard.
- Character frequencies and context depend on the input catalog. Tiny examples and
  new data distributions can behave differently from the evaluated catalog.
- IDs must follow the numeric source-prefixed format documented in the README.
- Reference records must be deduplicated; ownership otherwise suppresses valid links.
- The reported score is not classification accuracy or a guarantee on another dataset.

## Artifacts

`model.ubj` is the final classifier; `base_model.ubj` is the first classifier.
`text_weights.joblib` holds the two vectorizers. JSON configurations specify feature
order and decision policies. `SHA256SUMS.json` records artifact hashes. Load joblib
files only from a trusted source because their serialization can execute Python.
