from pprint import pprint

import timm

# model_names = timm.list_models("*resnet*")
# print(model_names)
pretrained = timm.create_model("resnet18", pretrained=True)
pprint(pretrained)
