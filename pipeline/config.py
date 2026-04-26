"""
pipeline/config.py
Speed defaults and exercise configuration.
"""

FRAME_SKIP       = 3      # process every Nth frame
FRAME_WIDTH      = 320    # native BarbellRow images are ~350px wide — never upscale
                          # 320px safe for all; videos will be downscaled, images untouched
MODEL_COMPLEXITY = 1      # 1=balanced — better pose detection on small/portrait images
                          # (0 was missing too many poses on 350x470 BarbellRow images)
MIN_REP_FRAMES   = 2      # discard reps shorter than this many frames


EXERCISE_CONFIG = {

    "Squat": {
        "clf_name":          "BackSquat",
        "data_mode":         "video",
        "label_mode":        "temporal",   # [[start_sec, end_sec], ...] per clip
        "video_dir":         "videos",
        "splits_dir":        "Labeled_Dataset/splits",
        "split_files":       {"train": "train_keys.json",
                              "val":   "val_keys.json",
                              "test":  "test_keys.json"},
        "splits_ignore":     {"traj_nan.json"},
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"knees_forward": "error_knees_forward.json",
                              "knees_inward":  "error_knees_inward.json"},
        "shallow_squat_dir": "Labeled_Dataset/labels/Shallow_Squat_Error_Dataset",
        "per_error_splits":  None,
    },

    "OHP": {
        "clf_name":          "OverheadPress",
        "data_mode":         "video",
        "label_mode":        "per_rep",    # VIDEOID_PERSONID_REPIDX: 0/1
        "video_dir":         "videos",
        "splits_dir":        "Labeled_Dataset/splits",
        "split_files":       {"train": "train_keys.json",
                              "val":   "val_keys.json",
                              "test":  "test_keys.json"},
        "splits_ignore":     set(),
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"elbow_error": "error_elbows.json",
                              "knees_error": "error_knees.json"},
        "shallow_squat_dir": None,
        "per_error_splits":  None,
    },

    "BarbellRow": {
        "clf_name":          "BarbellRow",
        "data_mode":         "image",
        "label_mode":        "per_rep",
        "image_dir":         "images",
        "splits_dir":        None,
        "split_files":       None,
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"lumbar_error":      "labels_lumbar_error.json",
                              "torso_angle_error": "labels_torso_angle_error.json"},
        "shallow_squat_dir": None,
        "per_error_splits": {
            "lumbar_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_Lumbar_error",
                "split_files": {"train": "train_Ids.json",
                                "val":   "val_Ids.json",
                                "test":  "test_Ids.json"},
            },
            "torso_angle_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_TorsoAngle_error",
                "split_files": {"train": "train_Ids.json",
                                "val":   "val_Ids.json",
                                "test":  "test_Ids.json"},
            },
        },
    },
}