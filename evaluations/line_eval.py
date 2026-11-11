import numpy as np
from shapely.geometry import LineString

from evaluations.rplan_eval.Evaluator import Evaluator_RPlan, corner_metric_thresh


def buffer_lines(polys, types, line_classes, radius=2.0):
    """Lines (types in line_classes) widened `radius` px on each side into polygons (flat ends, mitred corners,
    like create_coco_osm.py --line_buffer_px); other shapes pass through unchanged."""
    out = []
    for p, t in zip(polys, types):
        if t in line_classes and len(p) >= 2:
            g = LineString(p).buffer(radius, cap_style=2, join_style=2)
            if not g.is_empty:
                p = np.round(np.asarray(g.exterior.coords[:-1])).astype(np.int32)
        out.append(p)
    return out


def line_scores(pred_polys, pred_types, gt_polys, gt_types, line_classes, img_size, radius=corner_metric_thresh):
    """Line precision / recall: the lines are buffered `radius` px on each side (default: the 10 px corner
    tolerance) and matched like polygons (Evaluator_RPlan, IoU > 0.5); line_sem_* also need the right label.
    Returns {} without true lines."""
    pred = [(p, t) for p, t in zip(pred_polys, pred_types) if t in line_classes]
    gt = [(p, t) for p, t in zip(gt_polys, gt_types) if t in line_classes]
    if not gt:
        return {}
    pt, gtt = [t for _, t in pred], [t for _, t in gt]
    r = Evaluator_RPlan(disable_overlapping_filter=True, wd_as_line=False).evaluate_scene(
        room_polys=buffer_lines([p for p, _ in pred], pt, line_classes, radius), room_types=pt,
        gt_polys=buffer_lines([p for p, _ in gt], gtt, line_classes, radius), gt_polys_types=gtt, img_size=img_size)
    return {"line_prec": r["room_prec"], "line_rec": r["room_rec"],
            "line_sem_prec": r["room_sem_prec"], "line_sem_rec": r["room_sem_rec"]}
