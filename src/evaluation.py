"""Reference-level entity resolution metrics, including true singletons."""
from collections import defaultdict


def evaluate(truth, predictions, countries=None, candidates=None):
    if set(truth) != set(predictions):
        raise ValueError('Truth and predictions must contain exactly the same reference IDs')
    if candidates is not None and set(candidates) != set(truth):
        raise ValueError('Candidate coverage differs from truth')
    countries = countries or {}
    groups = defaultdict(list)
    scores, hits, predicted, actual = [], 0, 0, 0
    singletons, correct_singletons, recalled = 0, 0, 0
    for eid, expected in truth.items():
        expected, found = set(expected), set(predictions[eid])
        tp = len(expected & found)
        value = 1.25 * tp / (.25 * len(expected) + len(found)) if expected else float(not found)
        scores.append(value)
        groups[countries.get(eid, 'unknown')].append(value)
        hits += tp
        predicted += len(found)
        actual += len(expected)
        if not expected:
            singletons += 1
            correct_singletons += not found
        if candidates is not None:
            recalled += len(expected & set(candidates[eid]))
    result = dict(references=len(truth), macro_f05=sum(scores) / len(scores) if scores else None,
                  link_precision=hits / predicted if predicted else None,
                  link_recall=hits / actual if actual else None,
                  true_positive_links=hits, predicted_links=predicted, true_links=actual,
                  singletons=singletons,
                  singleton_accuracy=correct_singletons / singletons if singletons else None,
                  by_country={k: sum(v) / len(v) for k, v in sorted(groups.items())})
    if candidates is not None:
        result['candidate_recall'] = recalled / actual if actual else None
    return result
