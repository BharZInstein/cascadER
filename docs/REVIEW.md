# Reviewing a matching run

## Generate a report

```sh
python scripts/match.py --data-dir /path/to/records --work-dir artifacts/review-run --workers 2 --save-features
python scripts/review.py --work-dir artifacts/review-run
```

Open `artifacts/review-run/review.html` in a browser. The report bundles its data,
CSS, and JavaScript, so it needs no server or network connection. The export defaults
to at most 2,000 reference records; `--max-references` can raise this explicitly.
Export reads the input catalog and feature matrix locally, so use a small catalog
for interactive inspection.

The exporter verifies model artifact hashes and input file identity, rescores the
saved features, and checks that the final assignments match the saved predictions.
The model is not refitted during export. It shows the ten highest-scoring candidates
per reference plus every selected match, even if a match falls outside that top ten.

## Decision labels

| Label | Meaning |
|---|---|
| Accepted above threshold | Selected by the reference's country/global threshold and retained after ownership resolution |
| Accepted by top-candidate fallback | No candidate reached the main threshold; the best candidate reached the fallback threshold |
| Assigned to another reference | Selected before ownership resolution, then lost the target to another reference |
| Not selected by the threshold policy | Did not satisfy the reference's main threshold or fallback rule |

The interface shows the configured threshold and fallback. Scores are uncalibrated
model outputs. The evidence panel shows actual feature inputs, such as text cosine
similarity and the first-stage score. It does not assign causal importance to them.

## Review flags

- Missing reference or accepted-target addresses.
- An accepted pair with different first numeric tokens in its addresses. Reordered
  address components can trigger this too; it is an inspection cue, not a rejection rule.
- Any candidate score within 0.05 of the main threshold.
- A candidate selected before ownership resolution that was assigned elsewhere.

The flags are explicit heuristics. They have not been calibrated as an error detector.
The synthetic stress test demonstrates why address-number conflicts deserve inspection.

## Export human decisions

For each candidate, choose **Same business**, **Different business**, or **Unsure**.
Click a selected option again to clear it. **Export review decisions** downloads:

```json
{
  "schema_version": 1,
  "model_sha256": "…",
  "exported_at": "…",
  "decisions": [
    {
      "reference_id": "S1-1",
      "target_id": "S2-1",
      "decision": "match",
      "model_score": 0.98,
      "model_selected": true
    }
  ]
}
```

The example score above is illustrative. Reviews stay in browser memory until
exported and are cleared when the page closes. They do not alter the matching TSV,
feed an automatic training loop, or constitute independent ground truth without
reviewer verification. The model hash makes the reviewed model version identifiable.

All HTML-visible record text uses DOM text nodes. Embedded JSON escapes script
delimiters, including malicious text that tries to close its JSON script element.
Reports still contain the original names and addresses, so only publish reports
whose records you intend to share.
