"""EgoTriPlane Adapter package."""

from .geometry import (
    transform_points,
    get_3d_box_corners,
    project_points_to_camera,
    is_box_visible_in_camera,
    compute_direction,
    bin_distance_x,
    bin_distance_y,
    check_ego_path_overlap,
    gaussian_heatmap,
    is_region_observed,
    REGION_DEFS,
    world_to_grid,
    grid_to_world,
)

from .nusc_utils import (
    load_nusc_sample,
    normalize_category,
    compute_object_answerability,
    save_index,
    load_index,
    load_qa,
    save_qa,
)

from .camera_dropout import (
    FULL_NUSC_CAMERAS,
    CAMERA_SUBSETS,
    sample_camera_subset,
    generate_dropout_subsets,
    get_test_splits,
)

from .triplane_adapter import EgoTriPlaneAdapter

from .heads import (
    BEVHeatmapHead,
    VisibilityHead,
    CenterDetHead,
    BEVSegHead,
    TextAnswerHead,
)

from .losses import (
    FocalLoss,
    DetectionLoss,
    BEVSegLoss,
    Adapter3DPretrainLoss,
    EgoTriPlaneLoss,
    BEVLoss,
    VisibilityLoss,
    ConfigConsistencyLoss,
)

from .dataset import (
    Adapter3DPretrainDataset,
    Ego3DQADataset,
    AdapterStage1Dataset,
    detection_collate_fn,
)

__all__ = [
    # geometry
    "transform_points",
    "get_3d_box_corners",
    "project_points_to_camera",
    "is_box_visible_in_camera",
    "compute_direction",
    "bin_distance_x",
    "bin_distance_y",
    "check_ego_path_overlap",
    "gaussian_heatmap",
    "is_region_observed",
    "REGION_DEFS",
    "world_to_grid",
    "grid_to_world",
    # nusc_utils
    "load_nusc_sample",
    "normalize_category",
    "compute_object_answerability",
    "save_index",
    "load_index",
    "load_qa",
    "save_qa",
    # camera_dropout
    "FULL_NUSC_CAMERAS",
    "CAMERA_SUBSETS",
    "sample_camera_subset",
    "generate_dropout_subsets",
    "get_test_splits",
    # adapter
    "EgoTriPlaneAdapter",
    # heads
    "BEVHeatmapHead",
    "VisibilityHead",
    "CenterDetHead",
    "BEVSegHead",
    "TextAnswerHead",
    # losses
    "FocalLoss",
    "DetectionLoss",
    "BEVSegLoss",
    "Adapter3DPretrainLoss",
    "EgoTriPlaneLoss",
    "BEVLoss",
    "VisibilityLoss",
    "ConfigConsistencyLoss",
    # dataset
    "Adapter3DPretrainDataset",
    "Ego3DQADataset",
    "AdapterStage1Dataset",
    "detection_collate_fn",
]
