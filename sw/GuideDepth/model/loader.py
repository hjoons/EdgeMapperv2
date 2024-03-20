import torch.nn as nn
import torch

from GuidedDepth.model.GuideDepth import GuideDepth

def load_model(model_name, weights_pth):
    model = model_builder(model_name)

    if weights_pth is not None:
        ckpt = torch.load(weights_pth, map_location='cuda:0')
        model.load_state_dict(ckpt['model_state_dict'])

    return model

def model_builder(model_name):
    if model_name == 'GuideDepth':
        return GuideDepth(True)
    if model_name == 'GuideDepth-S':
        return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])

    print("Invalid model")
    exit(0)


