import torch
import torch.nn as nn


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


def get_debug_hook_grad(name):
    def debug_hook_grad(grad):
        print("Debug hook grad {}\n"
              "Has NaN: {}\n"
              "Has inf: {}\n"
              "Has zero: {}\n"
              "Min: {}\n"
              "Max: {}\n".format(
            name,
            torch.any(torch.isnan(grad)),
            torch.any(torch.isinf(grad)),
            torch.any(grad == 0.0),
            torch.min(grad),
            torch.max(grad)
            ))

        return grad

    return debug_hook_grad


class SkeleMotionBackbone(nn.Module):
    def __init__(self, final_width, seq_len=32, debug=False):
        super(SkeleMotionBackbone, self).__init__()

        linear_hidden_map = {30: 1536, 32: 2048}
        self.linear_hidden_width = linear_hidden_map[seq_len]
        self.final_width = final_width

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1)
        # nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(32),
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(64),
        self.relu4 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=self.linear_hidden_width, out_features=self.final_width)
        self.relu5 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=self.final_width, out_features=self.final_width)

        if debug:
            self.linear1.weight.register_hook(get_debug_hook_grad("Linear 1"))
            self.linear2.weight.register_hook(get_debug_hook_grad("Linear 2"))

    def forward(self, skele_motion_data):
        fe = self.conv1(skele_motion_data)
        fe = self.relu1(fe)

        fe = self.conv2(fe)
        fe = self.maxpool2(fe)
        fe = self.relu2(fe)

        fe = self.conv3(fe)
        fe = self.maxpool3(fe)
        fe = self.relu3(fe)

        fe = self.conv4(fe)
        fe = self.maxpool4(fe)
        fe = self.relu4(fe)

        fe = self.flatten(fe)

        fe = self.linear1(fe)
        fe = self.relu5(fe)

        fe = self.linear2(fe)

        return fe
