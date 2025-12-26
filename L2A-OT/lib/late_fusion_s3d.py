"""
This model is an extension for late fusion of the implementation of https://github.com/kylemin/S3D.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn


class late_fusion_S3D(nn.Module):
    def __init__(self, num_class, modelA, modelB, concat_features, channels):
        super(late_fusion_S3D, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc = nn.Linear(concat_features, 10)
        self.channels = channels


    def forward(self, x1x2):
        with torch.no_grad():
            x1 = self.modelA.base(x1x2[:, 0:self.channels[0]])
            x2 = self.modelB.base(x1x2[:, self.channels[0]:])
            y1 = F.avg_pool3d(x1, (2, x1.size(3), x1.size(4)), stride=1)
            y2 = F.avg_pool3d(x2, (2, x2.size(3), x2.size(4)), stride=1)
        x1 = self.modelA.fc(y1)
        x1 = x1.view(x1.size(0), x1.size(1), x1.size(2))
        x1 = torch.mean(x1, 2)

        x2 = self.modelB.fc(y2)
        x2 = x2.view(x2.size(0), x2.size(1), x2.size(2))
        x2 = torch.mean(x2, 2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(F.relu(x))
        
        return x

    def load_pretrained_unequal(self, file):
        # load the weight file and copy the parameters
        if os.path.isfile(file):
            print('Loading pre-trrained weight file.')
            weight_dict = torch.load(file)
            weight_dict = weight_dict["state_dict"] if type(weight_dict) == dict else weight_dict
            model_dict = self.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(
                            f' WARNING parameter size not equal. Skipping weight loading for: {name} '
                            f'File: {param.size()} Model: {model_dict[name].size()}')
                else:
                    print(f' WARNING parameter from weight file not found in model. Skipping {name}')

            print('Loaded pre-trained parameters from file.')
        else:
            raise ValueError(f"Weight file {file} does not exist.")



