"""Metrics for Ego3D-QA evaluation."""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def parse_json_answer(answer_text: str) -> dict:
    """Parse model output JSON string into dict."""
    if isinstance(answer_text, dict):
        return answer_text
    try:
        return json.loads(answer_text)
    except (json.JSONDecodeError, TypeError):
        # Try to extract JSON from text
        import re
        match = re.search(r'\{.*\}', answer_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"answer": "parse_error"}


def compute_metrics(preds: List[dict],
                     gt_qa: List[dict],
                     pred_key: str = "predicted_answer") -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        preds: list of prediction dicts (must have "id" field)
        gt_qa: list of ground truth QA dicts
        pred_key: key in pred dict that contains the predicted answer

    Returns:
        dict of metric name -> value
    """
    gt_by_id = {q["id"]: q for q in gt_qa}

    results = defaultdict(list)
    per_sample = []

    for pred in preds:
        qa_id = pred["id"]
        gt = gt_by_id.get(qa_id)
        if gt is None:
            continue

        pred_ans = parse_json_answer(str(pred.get(pred_key, "")))
        gt_ans = gt["answer"] if isinstance(gt["answer"], dict) else parse_json_answer(gt["answer"])

        sample_metrics = {"id": qa_id, "question_type": gt["question_type"]}

        # Answer accuracy
        ans_match = _match_answer(pred_ans, gt_ans)
        sample_metrics["answer_match"] = ans_match
        results["answer_accuracy"].append(1.0 if ans_match else 0.0)

        # Object category accuracy
        cat_match = _match_field(pred_ans, gt_ans, "object")
        sample_metrics["category_match"] = cat_match
        results["category_accuracy"].append(1.0 if cat_match else 0.0)

        # Position bin accuracies
        if "position" in gt_ans and "position" in pred_ans:
            x_match = pred_ans["position"].get("x") == gt_ans["position"].get("x")
            y_match = pred_ans["position"].get("y") == gt_ans["position"].get("y")
            dir_match = pred_ans["position"].get("direction") == gt_ans["position"].get("direction")
            sample_metrics["xbin_match"] = x_match
            sample_metrics["ybin_match"] = y_match
            sample_metrics["direction_match"] = dir_match
            results["xbin_accuracy"].append(1.0 if x_match else 0.0)
            results["ybin_accuracy"].append(1.0 if y_match else 0.0)
            results["direction_accuracy"].append(1.0 if dir_match else 0.0)

        # Lane relation accuracy
        lr_match = _match_field(pred_ans, gt_ans, "lane_relation")
        results["lane_relation_accuracy"].append(1.0 if lr_match else 0.0)

        # Visibility accuracy
        vis_match = _match_field(pred_ans, gt_ans, "visibility")
        results["visibility_accuracy"].append(1.0 if vis_match else 0.0)

        # Unknown / hallucination metrics
        gt_is_unknown = gt_ans.get("answer") == "unknown"
        pred_is_unknown = pred_ans.get("answer") == "unknown"

        if gt_is_unknown:
            results["unknown_accuracy"].append(1.0 if pred_is_unknown else 0.0)

        if gt["answerability"] == "unanswerable":
            results["hallucination"].append(0.0 if pred_is_unknown else 1.0)

        per_sample.append(sample_metrics)

    # Aggregate
    metrics = {}
    for key, values in results.items():
        if values:
            metrics[key] = float(np.mean(values))

    metrics["hallucination_rate"] = float(np.mean(results.get("hallucination", [0.0])))

    return metrics


def compute_grounding_error(pred_heatmaps: List[np.ndarray],
                             gt_centers: List[tuple],
                             x_range: tuple = (-20, 80),
                             y_range: tuple = (-40, 40),
                             grid_sx: int = 96,
                             grid_sy: int = 96) -> Dict[str, float]:
    """Compute BEV grounding error.

    Args:
        pred_heatmaps: list of [H, W] heatmaps
        gt_centers: list of (cx, cy) in ego coordinates
        x_range, y_range: grid ranges
        grid_sx, grid_sy: grid resolution

    Returns:
        dict with bev_center_error, top1_grounding_accuracy
    """
    errors = []
    correct = 0
    threshold = 2.0  # meters

    cell_x = (x_range[1] - x_range[0]) / grid_sx
    cell_y = (y_range[1] - y_range[0]) / grid_sy

    for hm, (cx, cy) in zip(pred_heatmaps, gt_centers):
        # Find peak
        max_idx = np.argmax(hm)
        gy = max_idx // grid_sx
        gx = max_idx % grid_sx

        pred_x = x_range[0] + (gx + 0.5) * cell_x
        pred_y = y_range[0] + (gy + 0.5) * cell_y

        error = np.sqrt((pred_x - cx) ** 2 + (pred_y - cy) ** 2)
        errors.append(error)

        if error < threshold:
            correct += 1

    return {
        "bev_center_error": float(np.mean(errors)) if errors else float("nan"),
        "top1_grounding_accuracy": correct / len(errors) if errors else 0.0,
    }


def compute_robustness_score(acc_6cam: float,
                              acc_4cam: float,
                              acc_3cam: float) -> Dict[str, float]:
    """Compute camera dropout robustness scores."""
    return {
        "r_4cam": acc_4cam / acc_6cam if acc_6cam > 0 else 0.0,
        "r_3cam": acc_3cam / acc_6cam if acc_6cam > 0 else 0.0,
    }


def compute_per_type_metrics(preds: List[dict],
                              gt_qa: List[dict]) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by question type."""
    gt_by_id = {q["id"]: q for q in gt_qa}
    by_type = defaultdict(list)

    for pred in preds:
        gt = gt_by_id.get(pred["id"])
        if gt is None:
            continue
        qtype = gt["question_type"]
        by_type[qtype].append((pred, gt))

    results = {}
    for qtype, pairs in by_type.items():
        type_preds = [p for p, _ in pairs]
        type_gt = [g for _, g in pairs]
        results[qtype] = compute_metrics(type_preds, type_gt)

    return results


def generate_results_table(all_metrics: Dict[str, Dict[str, float]],
                            split_names: List[str]) -> str:
    """Generate markdown results table.

    Args:
        all_metrics: {split_name: {metric_name: value}}
        split_names: ordered list of split names

    Returns:
        markdown string
    """
    metric_names = [
        "answer_accuracy", "xbin_accuracy", "ybin_accuracy",
        "direction_accuracy", "bev_center_error", "unknown_accuracy",
        "hallucination_rate", "r_4cam", "r_3cam",
    ]

    header = "| Method | " + " | ".join(m.replace("_", " ").title() for m in metric_names) + " |"
    sep = "|---" * (len(metric_names) + 1) + "|"

    rows = []
    for name in split_names:
        m = all_metrics.get(name, {})
        vals = []
        for mn in metric_names:
            v = m.get(mn, float("nan"))
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        rows.append(f"| {name} | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


def save_results_csv(metrics: Dict[str, float], output_path: str):
    """Save metrics as CSV."""
    with open(output_path, "w") as f:
        f.write("metric,value\n")
        for k, v in sorted(metrics.items()):
            f.write(f"{k},{v}\n")


def save_results_markdown(all_metrics: Dict[str, Dict[str, float]],
                           output_path: str,
                           split_names: Optional[List[str]] = None):
    """Save results as markdown table."""
    if split_names is None:
        split_names = list(all_metrics.keys())
    table = generate_results_table(all_metrics, split_names)
    with open(output_path, "w") as f:
        f.write("# EgoTriPlane-Adapter Results\n\n")
        f.write(table)
        f.write("\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_answer(pred: dict, gt: dict) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_val = pred.get("answer", "")
    gt_val = gt.get("answer", "")
    if gt_val == "unknown":
        return pred_val == "unknown"
    if gt_val == "visible":
        return pred_val in ("visible", "yes")
    return str(pred_val).lower() == str(gt_val).lower()


def _match_field(pred: dict, gt: dict, field: str) -> bool:
    """Check if a specific field matches."""
    pred_val = pred.get(field)
    gt_val = gt.get(field)
    if pred_val is None or gt_val is None:
        return False
    if isinstance(pred_val, str) and isinstance(gt_val, str):
        return pred_val.lower().strip() == gt_val.lower().strip()
    return pred_val == gt_val
