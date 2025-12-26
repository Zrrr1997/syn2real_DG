"""
This model is an extension for late fusion of the implementation of https://github.com/kylemin/S3D.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

class YOLO_mlp(nn.Module):
    def __init__(self, num_class, arch="SimpleNet"):
        super(YOLO_mlp, self).__init__()
        self.arch = arch
        if arch == 'BaseNet':
            self.fc1 = nn.Linear(80, 128)
            self.fc1_1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        elif arch == 'TanyaNet':
            self.fc1 = nn.Linear(80, 128)
            self.fc1_1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc2_1 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
        
        elif arch == 'PyramidNet':
            self.fc1 = nn.Linear(80, 512)
            self.fc1_1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc2_1 = nn.Linear(128,64)
            self.fc3 = nn.Linear(64,32)
            self.fc4 = nn.Linear(32, 10)

        elif arch == 'SimpleNet':
            diff_dim = 64
            self.fc1 = nn.Linear(80, diff_dim)
            self.fc2 = nn.Linear(diff_dim, 10)
        
        elif arch == 'LongNet':
            self.fc1 = nn.Linear(80, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 64)
            self.fc5 = nn.Linear(64, 32)
            self.fc6 = nn.Linear(32, 10)

        elif arch == 'LastNet':
            self.fc1 = nn.Linear(80, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, 8)
            self.fc5 = nn.Linear(8, 10)
        else:
            raise ValueError('Architecture for YOLO_mlp not supported')
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):     
        x = torch.max(x, dim=1)[0].float() # Aggregate sequence into one feature vector
        if self.arch == "BaseNet" or self.arch == "PyramidNet" or self.arch =="TanyaNet":
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc1_1(F.relu(x))
            x = self.dropout(x)
            x = self.fc2(F.relu(x))
            if self.arch != "BaseNet":
                x = self.fc2_1(F.relu(x)) # Not in BaseNets
            x = self.fc3(F.relu(x))
            logits = self.fc4(F.relu(x))
        elif self.arch == "SimpleNet":
            x = self.fc1(x)
            logits = self.fc2(F.relu(x))
        elif self.arch == "LongNet":
            x = self.fc1(x)
            x = self.fc2(F.relu(x))
            x = self.dropout(x)
            x = self.fc3(F.relu(x))
            x = self.dropout(x)
            x = self.fc4(F.relu(x))
            x = self.dropout(x)
            x = self.fc5(F.relu(x))
            logits = self.fc6(F.relu(x))
        elif self.arch == "LastNet":
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(F.relu(x))
            x = self.dropout(x)
            x = self.fc3(F.relu(x))
            x = self.fc4(F.relu(x))
            logits = self.fc5(F.relu(x))
        else:
            raise ValueError('Architecture for YOLO_mlp not supported')

        return logits

    def embed(self, x):
        with torch.no_grad():
            x = torch.max(x, dim=1)[0].float() # Aggregate sequence into one feature vector
            if self.arch == "BaseNet" or self.arch == "PyramidNet" or self.arch =="TanyaNet":
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc1_1(F.relu(x))
                x = self.dropout(x)
                x = self.fc2(F.relu(x))
                if self.arch != "BaseNet":
                    x = self.fc2_1(F.relu(x)) # Not in BaseNets
                x = self.fc3(F.relu(x))
            elif self.arch == "SimpleNet":
                x = self.fc1(x)
            elif self.arch == "LongNet":
                x = self.fc1(x)
                x = self.fc2(F.relu(x))
                x = self.dropout(x)
                x = self.fc3(F.relu(x))
                x = self.dropout(x)
                x = self.fc4(F.relu(x))
                x = self.dropout(x)
                x = self.fc5(F.relu(x))
            elif self.arch == "LastNet":
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc2(F.relu(x))
                x = self.dropout(x)
                x = self.fc3(F.relu(x))
                x = self.fc4(F.relu(x))
            else:
                raise ValueError('Architecture for YOLO_mlp not supported')
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



