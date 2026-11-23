import numpy as np
from shapely.geometry import LineString

from evaluations.rplan_eval.Evaluator import Evaluator_RPlan, corner_metric_thresh


def buffer_lines(polys, types, line_classes, radius=2.0, already_buffered=False):
    """Lines (types in line_classes) widened `radius` px on each side into polygons (flat ends, mitred corners,
    like create_coco_osm.py --line_buffer_px); other shapes pass through unchanged."""
    if already_buffered:
        # if already buffered, jsut return as list
        return list(polys)
    out = []
    for p, t in zip(polys, types):
        # only buffer if the type is in one of the line classes
        # otherwise don't buffer and just append
        if t in line_classes and len(p) >= 2:
            g = LineString(p).buffer(radius, cap_style=2, join_style=2)
            if g.geom_type == "MultiPolygon":
                # a line folding back on itself can split into pieces: keep the largest
                g = max(g.geoms, key=lambda x: x.area)
            if not g.is_empty:
                p = np.round(np.asarray(g.exterior.coords[:-1])).astype(np.int32)
        out.append(p)
    return out


def _scoring_helper(
    pred_polys,
    pred_types,
    gt_polys,
    gt_types,
    keep,
    line_classes,
    img_size,
    radius,
    gt_buffered=False,
    pred_buffered=False,
):
    """Helper function for scoring lines (runs the same evaluation)"""
    pred = [(p, t) for p, t in zip(pred_polys, pred_types) if keep(t)]
    gt = [(p, t) for p, t in zip(gt_polys, gt_types) if keep(t)]
    if not gt:
        # eg, if no lines or no label, just return
        return None, pred, gt
    pt, gtt = [t for _, t in pred], [t for _, t in gt]
    r = Evaluator_RPlan(
        disable_overlapping_filter=True, wd_as_line=False
    ).evaluate_scene(
        # add args for already buffered (pred + ground truth)
        room_polys=buffer_lines(
            [p for p, _ in pred],
            pt,
            line_classes,
            radius,
            already_buffered=pred_buffered,
        ),
        room_types=pt,
        gt_polys=buffer_lines(
            [p for p, _ in gt], gtt, line_classes, radius, already_buffered=gt_buffered
        ),
        gt_polys_types=gtt,
        img_size=img_size,
    )
    return r, pred, gt


def line_corner_hits(pred, gt):
    """Vertices of the true line `gt` within corner_metric_thresh px of a vertex of `pred` (each used once)."""
    pred, used = np.asarray(pred, float), set()
    for v in np.asarray(gt, float):
        d = np.linalg.norm(pred - v, axis=1)
        d[list(used)] = np.inf
        j = int(d.argmin())
        if d[j] <= corner_metric_thresh:
            used.add(j)
    return len(used)


def line_scores(
    pred_polys,
    pred_types,
    gt_polys,
    gt_types,
    line_classes,
    img_size,
    radius=corner_metric_thresh,
    gt_buffered=False,
    pred_buffered=False,
):
    """Line precision / recall: the lines are buffered `radius` px on each side (default: the 10 px corner
    tolerance) and matched like polygons (Evaluator_RPlan, IoU > 0.5); line_sem_* also need the right label.
    Also returns counts (hits, shapes, vertices) so engine.py can combine lines with polygons.
    Returns {} without true lines."""
    r, pred, gt = _scoring_helper(
        pred_polys,
        pred_types,
        gt_polys,
        gt_types,
        lambda t: t in line_classes,
        line_classes,
        img_size,
        radius,
        gt_buffered=gt_buffered,
        pred_buffered=pred_buffered,
    )
    if r is None:
        if not pred:
            return {}  # no lines at all: nothing to score
        # predicted lines but no true lines: all false positives. Precision only (recall is undefined here),
        # plus counts, so they also count against the combined all_* metrics in engine.py
        pts = sum(len(p) for p, _ in pred)
        return {"line_prec": 0.0, "line_sem_prec": 0.0, "line_corner_prec": 0.0,
                "line_hits": 0, "n_line_pred": len(pred), "n_line_gt": 0,
                "line_corner_hits": 0, "line_points_pred": pts, "line_points_gt": 0}
    pts_pred, pts_gt = sum(len(p) for p, _ in pred), sum(len(p) for p, _ in gt)
    if pred_buffered and gt_buffered:
        # bands: the evaluator's corner metric already compares band vertices
        c_hits = round(r["corner_prec"] * pts_pred)
    else:
        # raw lines: compare the line vertices of each matched pair (pred2gt indexes the size-sorted gt)
        c_hits = sum(line_corner_hits(pred[j][0], gt[r["gt_polys_sorted_indcs"][g]][0])
                     for j, g in enumerate(r["pred2gt_indices"]) if g != -1)
    # duplicates metrics but for lines
    return {
        "line_prec": r["room_prec"],
        "line_rec": r["room_rec"],
        "line_sem_prec": r["room_sem_prec"],
        "line_sem_rec": r["room_sem_rec"],
        "line_corner_prec": c_hits / max(pts_pred, 1),
        "line_corner_rec": c_hits / max(pts_gt, 1),
        # counts, for the combined metrics
        "line_hits": round(r["room_prec"] * len(pred)),
        "n_line_pred": len(pred),
        "n_line_gt": len(gt),
        "line_corner_hits": c_hits,
        "line_points_pred": pts_pred,
        "line_points_gt": pts_gt,
    }


def label_scores(
    pred_polys,
    pred_types,
    gt_polys,
    gt_types,
    line_classes,
    img_size,
    radius=corner_metric_thresh,
    gt_buffered=False,
    pred_buffered=False,
):
    """Per-label precision / recall: Evaluator_RPlan on one label's shapes at a time (lines buffered first).
    Keys: prec_<label id>, rec_<label id>, for labels with true shapes in the image; a label that is only
    predicted gets prec_<label id> = 0 (all false positives) and no recall (undefined without true shapes)."""
    out = {}
    for c in sorted({int(t) for t in gt_types}):
        r, _, _ = _scoring_helper(
            pred_polys,
            pred_types,
            gt_polys,
            gt_types,
            lambda t: t == c,
            line_classes,
            img_size,
            radius,
            gt_buffered=gt_buffered,
            pred_buffered=pred_buffered,
        )
        out[f"prec_{c}"], out[f"rec_{c}"] = r["room_prec"], r["room_rec"]
    for c in {int(t) for t in pred_types} - {int(t) for t in gt_types}:
        out[f"prec_{c}"] = 0.0  # predicted, but no true shapes with this label in the image
    return out
