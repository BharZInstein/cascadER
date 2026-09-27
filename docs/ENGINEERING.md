# Engineering notes

## What this project demonstrates

- Retrieval before classification to make entity matching tractable at catalog scale.
- Two-stage learning with group out-of-fold predictions for the second classifier.
- Explicit optimization for precision-focused macro F₀.₅, including true singletons.
- Bounded prediction batches, SQLite indexes, feature caches, and checkpointed runs.
- Model/data identity checks and validation of final ID coverage and unique ownership.
- A browser interface for inspecting decisions and exporting human judgments.
- Reproducible synthetic evaluation against two fixed baselines, with failure slices.

## Measured scale

A completed local run processed **1,732,544 references** and **136,792,412 candidate
pairs**, producing **5,761,204 links**, on a 16 GB Apple laptop. The recorded prediction
time was approximately **77.2 minutes**. That run reused a precomputed candidate list
and its first 86 features, then computed the final 22 features and rescored the pairs.
The timing excludes earlier retrieval, feature-cache creation, model fitting, and
final packaging. It is not the runtime of the from-scratch public demo workflow.

The source records and large caches are not published. Those historical scale figures
are an aggregate run record, not an independently reproducible public benchmark.
The synthetic benchmark in this repository is fully reproducible and reports its
own much smaller run size and timing scope.

## Why this design

**Tree models:** the inputs combine text similarity, missingness, frequencies, numeric
agreement, and context. Tree ensembles fit those mixed signals without a GPU training
dependency. Two stages let the second model use candidate confidence and neighborhood
support. Their added complexity needs to be justified against simpler baselines.

**Candidate retrieval:** pairwise comparison of every reference with every target
would be too costly. Blocking and lexical ranking bound the scored candidate list
to 80. Retrieval misses create an absolute limit: classification cannot recover a
true link that never reaches the matcher.

**Unique ownership:** the original labeled data supported assigning each target to
at most one reference. The final step enforces that property with stable tie handling.
This assumption should be revisited for datasets whose reference catalog has duplicates.

**Local report:** a static HTML artifact keeps inspection simple and reproducible.
It can show actual model decisions without introducing a web service or another model.
It currently serves an analyst workflow, not concurrent production users.

## Known failures and next experiments

The generated stress test scores 0.59940 macro F₀.₅, compared with 0.97224 on the
original internal validation. These scores measure different distributions.
Near-identical names at different address numbers cause false merges. Every true
singleton in this particular stress test receives a false match. That failure is
kept in the report rather than hidden by changing the generator after observing scores.

The next useful model work is to collect verified examples of those ambiguous cases,
train or calibrate an abstention policy on a development split, and evaluate it on
new independent data. A blanket address-number mismatch veto can also remove real
matches when addresses contain typos or have reordered components.

Further work with clear acceptance criteria:

1. **Independent public dataset:** document its license, map its fields, and publish
   fixed splits and baseline results. No suitable public benchmark has been integrated yet.
2. **Selective automation:** measure error rate versus automatic coverage for a review
   policy. The current flags are inspection cues, not a validated abstention mechanism.
3. **Serving:** add a versioned batch API with isolated jobs, resource limits, and
   request-schema tests when a remote deployment is actually needed.
4. **Input integration:** support arbitrary external IDs with reversible mappings.
   Current inference requires the numeric source-prefixed ID schema in the README.

## Resume wording grounded in the work

Use the measurements with their scope intact. For example:

> Built cascadER, a two-stage XGBoost entity-resolution system with 108 text and
> contextual features; achieved 0.972 macro F₀.₅ on 11,199 held-out business references.

> Implemented SQLite retrieval and resumable CPU inference used to score 136.8 million
> candidate pairs; added an interactive match-review interface, reproducible synthetic
> baselines, and CI tests for metrics, assignment policies, and input validation.

Be ready to explain group out-of-fold training, why F₀.₅ differs from accuracy, the
synthetic benchmark's failure cases, transductive character-frequency fitting, and
the difference between cached prediction timing and end-to-end runtime. The project
uses established methods; it does not claim a novel research architecture.
