import torch
from torch import nn
from torchvision.models import alexnet,resnet18,vgg11,googlenet
from collections import OrderedDict


def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool=True):
    
    
    if pretrained == True:
        if model_name == "alexnet":
            model = alexnet(pretrained = True)
            model.classifier[6].out_features = num_classes
            return model

        elif model_name == "resnet18":
            model = resnet18(pretrained = True)
            model.fc.out_features = num_classes
            return model

        elif model_name == "vgg11":
            model = vgg11(pretrained = True)
            model.classifier[6].out_features = num_classes
            return model

        elif model_name == "googlenet":
            model = googlenet(pretrained = True)
            model.fc.out_features = num_classes
            model.aux1 = nn.Sequential(
                OrderedDict(
                [
                    ("fc1", nn.Linear(1024,1024)),
                    ("fc2", nn.Linear(1024,num_classes))
                ]
                )
            )
            model.aux2 = nn.Sequential(
                OrderedDict(
                [
                    ("fc1", nn.Linear(1024,1024)),
                    ("fc2", nn.Linear(1024,num_classes))
                ]
                )
            )
            return model
    
    elif pretrained == False:
        if model_name == "alexnet":
            model = alexnet(pretrained = False)
            model.classifier[6].out_features = num_classes
            return model

        elif model_name == "resnet18":
            model = resnet18(pretrained = False)
            model.fc.out_features = num_classes
            return model

        elif model_name == "vgg11":
            model = vgg11(pretrained = False)
            model.classifier[6].out_features = num_classes
            return model

        elif model_name == "googlenet":
            model = googlenet(pretrained = False)
            model.fc.out_features = num_classes
            model.aux1 = nn.Sequential(
                OrderedDict(
                [
                    ("fc1", nn.Linear(1024,1024)),
                    ("fc2", nn.Linear(1024,num_classes))
                ]
                )
            )
            model.aux2 = nn.Sequential(
                OrderedDict(
                [
                    ("fc1", nn.Linear(1024,1024)),
                    ("fc2", nn.Linear(1024,num_classes))
                ]
                )
            )
            return model


