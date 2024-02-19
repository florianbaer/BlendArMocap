from __future__ import annotations

from mediapipe import solutions
from abc import abstractmethod

from . import cv_stream
from ...cgt_core.cgt_patterns import cgt_nodes


class LoaderNode(cgt_nodes.InputNode):

    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def contains_features(self, mp_res):
        pass


    @abstractmethod
    def empty_data(self):
        pass

    @abstractmethod
    def detected_data(self, mp_res):
        pass

    def exec_detection(self, mp_lib):
        """ Runs mediapipe detection on frame:
            -> detected_data: Detection Results.
            -> empty_data: No features detected.
            -> None: EOF or Finish. """
        self.stream.update()

        return self.detected_data(mp_res)

    def cvt2landmark_array(self, landmark_list):
        """landmark_list: A normalized landmark list proto message to be annotated on the image."""
        return [[idx, [landmark.x, landmark.y, landmark.z]] for idx, landmark in enumerate(landmark_list.landmark)]

    def __del__(self):
        if self.stream is not None:
            del self.stream
