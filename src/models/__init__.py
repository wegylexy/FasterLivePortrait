# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: __init__.py.py

from .warping_spade_model import WarpingSpadeModel
from .motion_extractor_model import MotionExtractorModel
from .appearance_feature_extractor_model import AppearanceFeatureExtractorModel
from .landmark_model import LandmarkModel
from .face_analysis_model import FaceAnalysisModel
from .stitching_model import StitchingModel


def __getattr__(name):
    # imported on first use only: mediapipe is a large optional dependency the
    # TensorRT CI path never selects
    if name == "MediaPipeFaceModel":
        from .mediapipe_face_model import MediaPipeFaceModel
        return MediaPipeFaceModel
    raise AttributeError(name)
