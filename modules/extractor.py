
import mediapipe as mp
import numpy as np



LANDMARK_NAMES = {
    11: "left_shoulder",  12: "right_shoulder",
    13: "left_elbow",     14: "right_elbow",
    15: "left_wrist",     16: "right_wrist",
    23: "left_hip",       24: "right_hip",
    25: "left_knee",      26: "right_knee",
    27: "left_ankle",     28: "right_ankle",
}


class PoseExtractor:
    def __init__(self,
                 min_detection_conf: float = 0.3,   
                 min_tracking_conf:  float = 0.3,
                 model_complexity:   int   = 1):   
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            enable_segmentation=False,
        )
        self.min_visibility = 0.3   

    def extract(self, frame_rgb: np.ndarray):
        """
        Returns:
            landmarks: (33, 4) array [x, y, z, visibility] or None
            annotated: frame with skeleton drawn
        """
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None, frame_rgb

        lm = results.pose_landmarks.landmark
        landmarks = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in lm],
            dtype=np.float32,
        )

        # Accept frame if hips OR knees are reasonably visible
        # (side-on squats / rows often occlude one full leg)
        hip_vis  = np.mean(landmarks[[23, 24], 3])
        knee_vis = np.mean(landmarks[[25, 26], 3])
        if hip_vis < self.min_visibility and knee_vis < self.min_visibility:
            return None, frame_rgb

        # Draw skeleton
        annotated = frame_rgb.copy()
        self.mp_draw.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(0, 200, 80),  thickness=2),
        )
        return landmarks, annotated

    def close(self):
        self.pose.close()