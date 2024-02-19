from . import mp_loader_node
import mediapipe as mp
from pose_format import Pose

from .pose_dim import restore_dimensions


class PoseLoaderNode(mp_loader_node.LoaderNode):

    def __init__(self, path, refine_face_landmarks: bool = False):
        self.path = path
        self.solution = mp.solutions.holistic
        mp_loader_node.LoaderNode.__init__(self, path)
        self.refine_face_landmarks = refine_face_landmarks

    # https://google.github.io/mediapipe/solutions/holistic#python-solution-api
    def update(self, data, frame):
        print(self.path)
        pose = None
        with open(self.path, "rb") as f:
            pose = Pose.read(f.read())
        pose = restore_dimensions(pose, width=pose.header.dimensions.width, height=pose.header.dimensions.height)
        return self.detected_data(pose), frame

    def empty_data(self):
        return [[[], []], [[[]]], []]

    def detected_data(self, mp_res):
        face, pose, l_hand, r_hand = [], [], [], []
        if mp_res.pose_landmarks:
            pose = self.cvt2landmark_array(mp_res.pose_landmarks)
        if mp_res.face_landmarks:
            face = self.cvt2landmark_array(mp_res.face_landmarks)
        if mp_res.left_hand_landmarks:
            l_hand = [self.cvt2landmark_array(mp_res.left_hand_landmarks)]
        if mp_res.right_hand_landmarks:
            r_hand = [self.cvt2landmark_array(mp_res.right_hand_landmarks)]
        # TODO: recheck every update, mp hands are flipped while detecting holistic.
        return [[r_hand, l_hand], [face], pose]

    def contains_features(self, mp_res):
        if not mp_res.pose_landmarks:
            return False
        return True

    def draw_result(self, s, mp_res, mp_drawings):
        pass


