import numpy as np

from . import mp_loader_node
import mediapipe as mp



class PoseLoaderNode(mp_loader_node.LoaderNode):

    def __init__(self, path, refine_face_landmarks: bool = False):
        self.path = path

        from .pose_dim import restore_dimensions, restore_dimensions_div
        from pose_format import Pose
        self.pose = None
        with open(self.path, "rb") as f:
            self.pose = Pose.read(f.read())
            #self.pose = restore_dimensions(self.pose, width=self.pose.body.dimensions.width, height=self.pose.header.dimensions.height)
            #self.pose.body = self.pose.body.zero_filled()
        self.pose = restore_dimensions_div(self.pose, width=self.pose.header.dimensions.width, height=self.pose.header.dimensions.height)
        mp_loader_node.LoaderNode.__init__(self, path)
        self.refine_face_landmarks = refine_face_landmarks

    # https://google.github.io/mediapipe/solutions/holistic#python-solution-api
    def update(self, data, frame):
        # check if we are at the end
        if self.pose.body.data.shape[0] <= frame:
            return None, frame

        #pose = restore_dimensions(pose, width=pose.header.dimensions.width, height=pose.header.dimensions.height)
        return self.detected_data(self.pose, frame), frame

    def empty_data(self):
        return [[[], []], [[[]]], []]

    def detected_data(self, pose_data, frame):
        pose = self.cvt2landmark_array(pose_data.get_components(['POSE_LANDMARKS']),frame)
        face = self.cvt2landmark_array(pose_data.get_components(['FACE_LANDMARKS']),frame)
        l_hand = [self.cvt2landmark_array(pose_data.get_components(['LEFT_HAND_LANDMARKS']),frame)]
        r_hand = [self.cvt2landmark_array(pose_data.get_components(['RIGHT_HAND_LANDMARKS']),frame)]


        return [[r_hand, l_hand], [face], pose]


    def cvt2landmark_array(self, landmark_list, frame):

        """landmark_list: A normalized landmark list proto message to be annotated on the image."""
        return [[idx, [landmark_list.body.data[frame,0,idx,0], landmark_list.body.data[frame,0,idx,1], landmark_list.body.data[frame,0,idx,2]]] for idx, component in enumerate(landmark_list.header.components[0].points)]


    def contains_features(self, mp_res):
        if not mp_res.pose_landmarks:
            return False
        return True


