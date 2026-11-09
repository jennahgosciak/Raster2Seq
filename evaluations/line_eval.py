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