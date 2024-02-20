from __future__ import annotations

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


    def __del__(self):
        pass
