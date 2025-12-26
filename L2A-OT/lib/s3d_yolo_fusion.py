"""
This model is an extension for late fusion with YOLO_mlp of the implementation of https://github.com/kylemin/S3D.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn


class s3d_YOLO_fusion(nn.Module):
    def __init__(self, num_class, s3d_model, yolo_model, s3d_features):
        super(s3d_YOLO_fusion, self).__init__()
        self.s3d_model = s3d_model
        self.yolo_model = yolo_model

        #self.fc = nn.Linear(s3d_features, num_class)


    def forward(self, img, det):
        with torch.no_grad(): # Freeze initial s3d network without top layers
           x1 = self.s3d_model.base(img)
           y1 = F.avg_pool3d(x1, (2, x1.size(3), x1.size(4)), stride=1)
        x1 = self.s3d_model.fc(y1)
        x1 = x1.view(x1.size(0), x1.size(1), x1.size(2))
        x_s3d = torch.mean(x1, 2)

        with torch.no_grad(): # Freeze pre-trained YOLO
            x_yolo = self.yolo_model(det)

        x = F.normalize(x_s3d) + F.normalize(x_yolo)

        return x

    def embed(self, img):
        with torch.no_grad(): # Freeze initial s3d network without top layers
           x1 = self.s3d_model.base(img)
           y1 = F.avg_pool3d(x1, (2, x1.size(3), x1.size(4)), stride=1)
           x1 = self.s3d_model.fc(y1)
           x1 = x1.view(x1.size(0), x1.size(1), x1.size(2))
           x_s3d = torch.mean(x1, 2)

        return x_s3d

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



