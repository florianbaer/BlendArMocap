import pdb
import mediapipe as mp
import numpy as np
from . import cv_stream, mp_detector_node
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import logging

class HolisticDetector(mp_detector_node.DetectorNode):
    def __init__(self, stream, model_complexity: int = 2,
                 min_detection_confidence: float = .7, refine_face_landmarks: bool = False):
        #base constructor
        mp_detector_node.DetectorNode.__init__(self, stream)
        logging.getLogger().setLevel(logging.DEBUG)
        self.solution = mp.solutions.holistic
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.refine_face_landmarks = refine_face_landmarks

    # https://google.github.io/mediapipe/solutions/holistic#python-solution-api
    def update(self, data, frame):
        self.frame = frame
        with self.solution.Holistic(
                refine_face_landmarks=self.refine_face_landmarks,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                static_image_mode=False,
        ) as mp_lib:
            print(f"Frame: {frame}")
            logging.debug(f"Frame: {frame}")
            return self.exec_detection(mp_lib), frame

    def empty_data(self):
        return [[[], []], [[[]]], []]

    def detected_data(self, mp_res):
        #print(f"Detected data: {mp_res}")
        face, pose, l_hand, r_hand = [], [], [], []
        if mp_res.pose_landmarks:
            pose = self.cvt2landmark_array(mp_res.pose_landmarks)
        if mp_res.face_landmarks:
            face = self.cvt2landmark_array(mp_res.face_landmarks)
        if mp_res.left_hand_landmarks:
            l_hand = [self.cvt2landmark_array(mp_res.left_hand_landmarks, self.frame)]
        if mp_res.right_hand_landmarks:
            r_hand = [self.cvt2landmark_array(mp_res.right_hand_landmarks)]
        # TODO: recheck every update, mp hands are flipped while detecting holistic.
        return_value = [[r_hand, l_hand], [face], pose]
#        self.deep_list_analysis(return_value)

        return return_value

    def contains_features(self, mp_res):
        if not mp_res.pose_landmarks:
            return False
        return True

    def draw_result(self, s, mp_res, mp_drawings):
        mp_drawings.draw_landmarks(
            s.frame,
            mp_res.face_landmarks,
            self.solution.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.drawing_style
                .get_default_face_mesh_contours_style())
        mp_drawings.draw_landmarks(
            s.frame,
            mp_res.pose_landmarks,
            self.solution.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_style
                .get_default_pose_landmarks_style())
        mp_drawings.draw_landmarks(
            s.frame, mp_res.left_hand_landmarks, self.solution.HAND_CONNECTIONS)
        mp_drawings.draw_landmarks(
            s.frame, mp_res.right_hand_landmarks, self.solution.HAND_CONNECTIONS)


if __name__ == '__main__':
    detection_type = "image"
    detector = HolisticDetector(cv_stream.Stream(0))

    frame = 0
    for _ in range(15):
        frame += 1
        detector.update(None, frame)

    del detector
# endregion
