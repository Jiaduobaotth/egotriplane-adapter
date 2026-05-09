"""Camera dropout strategies for training robustness.

Provides fixed subsets, random sampling, and unseen-holdout logic
so the model learns sensor-configuration-agnostic representations.
"""

import random
from typing import List, Optional

FULL_NUSC_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_SUBSETS = {
    "6cam": FULL_NUSC_CAMERAS.copy(),
    "front3": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
    "front4": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"],
    "no_rear": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
    "no_left": ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_RIGHT"],
    "no_right": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT"],
    "front_back": ["CAM_FRONT", "CAM_BACK"],
    "back3": ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
}

# A deliberately held-out combination for unseen-subset evaluation
UNSEEN_HOLDOUT_CAMERAS = [
    "CAM_FRONT",
    "CAM_BACK",
    "CAM_FRONT_LEFT",
    "CAM_BACK_RIGHT",
]


def sample_camera_subset(all_cameras: Optional[List[str]] = None,
                          min_cams: int = 3,
                          max_cams: int = 6) -> List[str]:
    """Randomly sample a camera subset.

    Args:
        all_cameras: full list of available camera names
        min_cams: minimum number of cameras to keep
        max_cams: maximum number of cameras to keep

    Returns:
        sorted list of camera names
    """
    if all_cameras is None:
        all_cameras = FULL_NUSC_CAMERAS

    k = random.randint(min_cams, min(max_cams, len(all_cameras)))
    return sorted(random.sample(all_cameras, k))


def sample_camera_subset_excluding(all_cameras: Optional[List[str]] = None,
                                    min_cams: int = 3,
                                    max_cams: int = 6) -> List[str]:
    """Sample a subset that does NOT include the unseen holdout combination."""
    if all_cameras is None:
        all_cameras = FULL_NUSC_CAMERAS.copy()

    while True:
        subset = sample_camera_subset(all_cameras, min_cams, max_cams)
        if sorted(subset) != sorted(UNSEEN_HOLDOUT_CAMERAS):
            return subset


def generate_dropout_subsets(all_cameras: Optional[List[str]] = None,
                              num_versions: int = 3,
                              include_full: bool = True,
                              min_cams: int = 3,
                              max_cams: int = 6) -> List[List[str]]:
    """Generate multiple camera dropout versions for one sample.

    Args:
        all_cameras: full camera list
        num_versions: number of random dropout versions
        include_full: always include full camera set
        min_cams, max_cams: range for random sampling

    Returns:
        list of camera name lists
    """
    if all_cameras is None:
        all_cameras = FULL_NUSC_CAMERAS

    subsets = []
    if include_full:
        subsets.append(sorted(all_cameras))

    for _ in range(num_versions):
        subset = sample_camera_subset(all_cameras, min_cams, max_cams)
        if subset not in subsets:
            subsets.append(subset)

    return subsets


def get_test_splits() -> dict:
    """Return pre-defined evaluation camera splits.

    Returns:
        dict mapping split name to camera list
    """
    splits = {
        "6cam": FULL_NUSC_CAMERAS.copy(),
        "front3": CAMERA_SUBSETS["front3"].copy(),
        "front4": CAMERA_SUBSETS["front4"].copy(),
        "no_rear": CAMERA_SUBSETS["no_rear"].copy(),
        "no_left": CAMERA_SUBSETS["no_left"].copy(),
        "no_right": CAMERA_SUBSETS["no_right"].copy(),
    }
    return splits


def generate_random_ncam_val(n: int,
                              num_samples: int = 200,
                              seed: int = 42) -> List[List[str]]:
    """Generate random n-camera subsets for evaluation.

    Args:
        n: number of cameras per subset
        num_samples: how many subsets to generate
        seed: random seed

    Returns:
        list of camera subsets (each is List[str])
    """
    rng = random.Random(seed)
    subsets = set()
    while len(subsets) < num_samples:
        subset = tuple(sorted(rng.sample(FULL_NUSC_CAMERAS, n)))
        subsets.add(subset)
    return [list(s) for s in subsets]


def filter_by_camera_subset(sample: dict, camera_subset: List[str]) -> dict:
    """Filter a sample dict to only include cameras in the given subset."""
    filtered = dict(sample)
    filtered["cameras"] = [
        c for c in sample["cameras"] if c["name"] in camera_subset
    ]
    return filtered
