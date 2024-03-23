import torch.nn as nn
import torch

from GuideDepth.model.GuideDepth import GuideDepth

# This will load the model using the weights provided in https://github.com/mic-rud/GuidedDecoding/tree/main
def load_model(model_name):
    model = model_builder(model_name)
    
    return model

def model_builder(model_name):
    if model_name == 'GuideDepth':
        return GuideDepth(True)
    if model_name == 'GuideDepth-S':
        return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])

    print("Invalid model")
    exit(0)


