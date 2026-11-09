import numpy as np

from shapely.geometry import LineString
from shapely.ops import unary_union

from evaluations.rplan_eval.Evaluator import corner_metric_thresh


def line_scores(pred_lines, gt_lines, tol=corner_metric_thresh):
    """Length-based line precision / recall / F1. Each line is an (N, 2) array in pixels; a stretch of
    predicted line counts as correct if it is within `tol` px of a ground-truth line (recall: the reverse).
    Returns {} for scenes without ground-truth lines."""
    if not gt_lines:
        return {}
    gt = unary_union([LineString(l) for l in gt_lines if len(l) >= 2])
    pred = unary_union([LineString(l) for l in pred_lines if len(l) >= 2])
    prec = pred.intersection(gt.buffer(tol)).length / pred.length if pred.length else 0.0
    rec = gt.intersection(pred.buffer(tol)).length / gt.length
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"line_prec": prec, "line_rec": rec, "line_f1": f1}

def line_scores_per_label(pred_polys, pred_types, gt_polys, gt_types, line_classes, tol=corner_metric_thresh):
    """line_scores for each line class with ground-truth lines in the image (keys end in _<class id>),
    plus their mean under the plain keys. Returns {} if the image has no ground-truth lines."""
    out, per = {}, []
    for c in sorted(set(gt_types) & set(line_classes)):
        s = line_scores([p for p, t in zip(pred_polys, pred_types) if t == c],
                        [p for p, t in zip(gt_polys, gt_types) if t == c], tol)
        out.update({f"{k}_{c}": v for k, v in s.items()})
        per.append(s)
    if per:
        out.update({k: float(np.mean([s[k] for s in per])) for k in per[0]})
    return out