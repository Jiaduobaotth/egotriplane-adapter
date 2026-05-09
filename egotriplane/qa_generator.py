"""Ego3D-QA question-answer generation logic.

Generates four question types from nuScenes sample data:
  A. closest object localization
  B. ego-path blocking object
  C. spatial relation
  D. unknown / observability

All QAs include camera-dropout variants for robustness training.
"""

import json
import random
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

from .geometry import (
    compute_direction,
    bin_distance_x,
    bin_distance_y,
    check_ego_path_overlap,
    is_box_visible_in_camera,
    is_region_observed,
    REGION_DEFS,
    classify_region,
)
from .nusc_utils import compute_object_answerability


# ---------------------------------------------------------------------------
# QA generators (one per type)
# ---------------------------------------------------------------------------

def generate_closest_object_qa(sample: dict,
                                camera_subset: List[str]) -> List[dict]:
    """Type A: Closest object localization questions.

    Selects the nearest object in front and asks for its position.
    """
    objects = sample["objects"]
    cam_by_name = {c["name"]: c for c in sample["cameras"]}

    # Filter to answerable front objects (include all foreground categories)
    front_objects = []
    for obj in objects:
        if obj["category"] not in ["vehicle", "pedestrian", "cyclist",
                                     "barrier", "traffic_cone"]:
            continue
        if obj["center_ego"][0] <= 0:
            continue
        # Check visibility in this camera subset
        visible = False
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            if is_box_visible_in_camera(
                obj["box_3d"],
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32),
                cam["width"], cam["height"],
            ):
                visible = True
                break
        if not visible:
            continue
        front_objects.append(obj)

    if len(front_objects) < 1:
        return []

    # Sort by distance
    front_objects.sort(key=lambda o: o["center_ego"][0])

    # Filter: closest and second-closest must be >2m apart
    target = front_objects[0]
    if len(front_objects) > 1:
        diff = abs(front_objects[1]["center_ego"][0] - target["center_ego"][0])
        if diff < 2.0:
            return []

    qa_list = []
    direction = compute_direction(np.array(target["center_ego"]))
    category = target["category"]

    templates = [
        "Where is the closest {cat} in front of the ego vehicle?",
        "Where is the nearest object in the {dir} region?",
        "What is the relative position of the nearest {cat}?",
    ]

    for tmpl in templates:
        question = tmpl.format(cat=category, dir=direction.replace("-", " "))
        qa_id = f"{sample['sample_token']}_{uuid.uuid4().hex[:6]}"

        answer = {
            "answer": "visible",
            "object": category,
            "position": {
                "x": bin_distance_x(float(target["center_ego"][0])),
                "y": bin_distance_y(float(target["center_ego"][1])),
                "direction": direction,
            },
            "visibility": "visible",
        }

        qa_list.append({
            "id": qa_id,
            "sample_token": sample["sample_token"],
            "camera_subset": camera_subset,
            "question_type": "closest_object",
            "question": question,
            "answer_text": json.dumps(answer),
            "answer": answer,
            "grounding": {
                "object_id": target["object_id"],
                "category": target["category"],
                "box_3d": target["box_3d"],
            },
            "answerability": "answerable",
        })

    return qa_list


def generate_closest_ego_path_qa(sample: dict,
                                  camera_subset: List[str]) -> List[dict]:
    """Type A2: Closest object ON the ego path.

    Deterministically selects the closest (by longitudinal distance)
    visible object that overlaps the ego lane corridor.
    """
    objects = sample["objects"]
    cam_by_name = {c["name"]: c for c in sample["cameras"]}

    # Filter: visible, in front (x > 0), overlaps ego path
    candidates = []
    for obj in objects:
        if obj["category"] not in ["vehicle", "pedestrian", "cyclist",
                                     "barrier", "traffic_cone"]:
            continue
        if obj["center_ego"][0] <= 0:
            continue
        center = np.array(obj["center_ego"])
        size = np.array(obj["size"])
        if not check_ego_path_overlap(center, size, yaw=obj.get("yaw_ego", 0.0)):
            continue
        # Check visibility in this camera subset
        visible = False
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            if is_box_visible_in_camera(
                obj["box_3d"],
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32),
                cam["width"], cam["height"],
            ):
                visible = True
                break
        if not visible:
            continue
        candidates.append(obj)

    # Sort by longitudinal distance — closest first
    candidates.sort(key=lambda o: o["center_ego"][0])
    target = candidates[0] if candidates else None

    qa_list = []
    templates = [
        "What is the closest object on the ego path?",
        "Which object is nearest inside the ego lane?",
    ]

    for tmpl in templates:
        qa_id = f"{sample['sample_token']}_{uuid.uuid4().hex[:6]}"

        if target:
            direction = compute_direction(np.array(target["center_ego"]))
            answer = {
                "answer": target["category"],
                "object": target["category"],
                "position": {
                    "x": bin_distance_x(float(target["center_ego"][0])),
                    "y": bin_distance_y(float(target["center_ego"][1])),
                    "direction": direction,
                },
                "visibility": "visible",
            }
            grounding = {
                "object_id": target["object_id"],
                "category": target["category"],
                "box_3d": target["box_3d"],
            }
            answerability = "answerable"
        else:
            answer = {
                "answer": "none",
                "lane_relation": "clear_ego_path",
                "visibility": "visible",
            }
            grounding = None
            answerability = "answerable"

        qa_list.append({
            "id": qa_id,
            "sample_token": sample["sample_token"],
            "camera_subset": camera_subset,
            "question_type": "closest_ego_path",
            "question": tmpl,
            "answer_text": json.dumps(answer),
            "answer": answer,
            "grounding": grounding,
            "answerability": answerability,
        })

    return qa_list


def generate_ego_path_qa(sample: dict,
                          camera_subset: List[str]) -> List[dict]:
    """Type B: Ego-path blocking object questions.

    Checks if any object blocks the ego lane corridor.
    """
    objects = sample["objects"]
    cam_by_name = {c["name"]: c for c in sample["cameras"]}

    # Find objects that overlap ego path
    blocking = []
    for obj in objects:
        if obj["category"] not in ["vehicle", "pedestrian", "cyclist", "barrier",
                                     "traffic_cone"]:
            continue
        center = np.array(obj["center_ego"])
        size = np.array(obj["size"])
        if not check_ego_path_overlap(center, size, yaw=obj.get("yaw_ego", 0.0)):
            continue
        # Check visibility
        visible = False
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            if is_box_visible_in_camera(
                obj["box_3d"],
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32),
                cam["width"], cam["height"],
            ):
                visible = True
                break
        if visible:
            blocking.append(obj)

    blocking.sort(key=lambda o: o["center_ego"][0])
    target = blocking[0] if blocking else None

    qa_list = []
    templates = [
        "Is there any object blocking the ego lane?",
        "Is there any vehicle inside the ego path?",
        "Which object is most relevant to the ego vehicle's near-term driving?",
    ]

    for tmpl in templates:
        qa_id = f"{sample['sample_token']}_{uuid.uuid4().hex[:6]}"

        if target:
            direction = compute_direction(np.array(target["center_ego"]))
            answer = {
                "answer": "yes",
                "object": target["category"],
                "position": {
                    "x": bin_distance_x(float(target["center_ego"][0])),
                    "y": bin_distance_y(float(target["center_ego"][1])),
                    "direction": direction,
                },
                "lane_relation": "in_ego_path",
                "visibility": "visible",
            }
            grounding = {
                "object_id": target["object_id"],
                "category": target["category"],
                "box_3d": target["box_3d"],
            }
        else:
            answer = {
                "answer": "no",
                "lane_relation": "clear_ego_path",
                "visibility": "visible",
            }
            grounding = None

        qa_list.append({
            "id": qa_id,
            "sample_token": sample["sample_token"],
            "camera_subset": camera_subset,
            "question_type": "ego_path",
            "question": tmpl,
            "answer_text": json.dumps(answer),
            "answer": answer,
            "grounding": grounding,
            "answerability": "answerable",
        })

    return qa_list


def generate_spatial_relation_qa(sample: dict,
                                  camera_subset: List[str]) -> List[dict]:
    """Type C: Spatial relation questions.

    Asks about the relative position of an object (left/right, front/back).
    """
    objects = sample["objects"]
    cam_by_name = {c["name"]: c for c in sample["cameras"]}

    # Pick a visible, answerable object with clear direction
    candidates = []
    for obj in objects:
        if obj["category"] not in ["vehicle", "pedestrian", "cyclist"]:
            continue
        center = np.array(obj["center_ego"])
        direction = compute_direction(center)
        if direction == "near-ego":
            continue
        visible = False
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            if is_box_visible_in_camera(
                obj["box_3d"],
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32),
                cam["width"], cam["height"],
            ):
                visible = True
                break
        if visible:
            candidates.append(obj)

    if not candidates:
        return []

    target = random.choice(candidates)
    direction = compute_direction(np.array(target["center_ego"]))
    category = target["category"]

    qa_list = []
    templates = [
        "Is the {cat} to the left or right of the ego vehicle?",
        "Is the {cat} in front of or behind the ego vehicle?",
        "Is the {cat} in the front-left or front-right region?",
    ]

    for tmpl in templates:
        qa_id = f"{sample['sample_token']}_{uuid.uuid4().hex[:6]}"
        answer = {
            "answer": direction,
            "object": category,
            "position": {
                "x": bin_distance_x(float(target["center_ego"][0])),
                "y": bin_distance_y(float(target["center_ego"][1])),
                "direction": direction,
            },
            "visibility": "visible",
        }
        qa_list.append({
            "id": qa_id,
            "sample_token": sample["sample_token"],
            "camera_subset": camera_subset,
            "question_type": "spatial_relation",
            "question": tmpl.format(cat=category),
            "answer_text": json.dumps(answer),
            "answer": answer,
            "grounding": {
                "object_id": target["object_id"],
                "category": target["category"],
                "box_3d": target["box_3d"],
            },
            "answerability": "answerable",
        })

    return qa_list


def generate_unknown_qa(sample: dict,
                         camera_subset: List[str]) -> List[dict]:
    """Type D: Unknown / observability questions.

    For regions NOT observed by the current camera subset, generate
    questions whose answer should be 'unknown'.
    """
    cam_by_name = {c["name"]: c for c in sample["cameras"]}
    calibrations = [
        {
            "name": c["name"],
            "K": c["K"],
            "T_ego_cam": c["T_ego_cam"],
            "width": c["width"],
            "height": c["height"],
        }
        for c in sample["cameras"] if c["name"] in camera_subset
    ]

    qa_list = []
    categories = ["vehicle", "pedestrian", "cyclist"]

    for region_name in REGION_DEFS:
        observed = is_region_observed(region_name, camera_subset, calibrations)

        if observed:
            continue  # Only generate unknown for unobserved regions

        category = random.choice(categories)

        templates = [
            f"Is there a {category} in the {region_name} region?",
            f"Can the current camera setup observe the {region_name} area?",
        ]

        for tmpl in templates:
            qa_id = f"{sample['sample_token']}_{uuid.uuid4().hex[:6]}"
            answer = {
                "answer": "unknown",
                "visibility": "not_observed",
                "reason": f"The {region_name} region is not covered by the current camera configuration.",
            }
            qa_list.append({
                "id": qa_id,
                "sample_token": sample["sample_token"],
                "camera_subset": camera_subset,
                "question_type": "unknown",
                "question": tmpl,
                "answer_text": json.dumps(answer),
                "answer": answer,
                "grounding": None,
                "answerability": "unanswerable",
            })

    return qa_list


# ---------------------------------------------------------------------------
# Master QA generator
# ---------------------------------------------------------------------------

def generate_qa_for_sample(sample: dict,
                            camera_subset: List[str],
                            max_per_type: int = 2,
                            types: Optional[List[str]] = None,
                            debug: bool = False) -> List[dict]:
    """Generate all QA instances for one sample + camera subset.

    Args:
        sample: sample dict from index
        camera_subset: list of camera names
        max_per_type: max QA instances per type
        types: which QA types to generate (default all)
        debug: if True, print diagnostics for closest/ego_path QAs

    Returns:
        list of QA dicts
    """
    if types is None:
        types = ["closest_object", "ego_path", "closest_ego_path",
                 "spatial_relation", "unknown"]

    all_qa = []

    if "closest_object" in types:
        qas = generate_closest_object_qa(sample, camera_subset)
        if debug and qas:
            _debug_closest(qas, "closest_object")
        all_qa.extend(qas[:max_per_type])

    if "closest_ego_path" in types:
        qas = generate_closest_ego_path_qa(sample, camera_subset)
        if debug and qas:
            _debug_closest(qas, "closest_ego_path")
        all_qa.extend(qas[:max_per_type])

    if "ego_path" in types:
        qas = generate_ego_path_qa(sample, camera_subset)
        if debug and qas:
            _debug_closest(qas, "ego_path")
        all_qa.extend(qas[:max_per_type])

    if "spatial_relation" in types:
        qas = generate_spatial_relation_qa(sample, camera_subset)
        all_qa.extend(qas[:max_per_type])

    if "unknown" in types:
        qas = generate_unknown_qa(sample, camera_subset)
        all_qa.extend(qas[:max_per_type])

    return all_qa


def _debug_closest(qa_list: List[dict], qa_type: str):
    """Print debug info for closest/ego_path QA selections."""
    for qa in qa_list:
        g = qa.get("grounding", {})
        if g:
            box = g.get("box_3d", [])
            print(f"  [{qa_type}] {qa['id'][-12:]}: "
                  f"target={g.get('category','?')} "
                  f"x={box[0]:.1f} y={box[1]:.1f} "
                  f"obj_id={g.get('object_id','?')[:12]}")
        else:
            print(f"  [{qa_type}] {qa['id'][-12:]}: "
                  f"target=None answer={qa['answer'].get('answer','?')}")
