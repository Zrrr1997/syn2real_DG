from torch.utils.data import DataLoader
import torch
from resnet import *
import argparse
from torch import nn
from utils.datasets import *
from torch.nn import functional as F
from utils.logger import Logger
from torch import optim
import os
from model import *
from utils.data_reader import *
from torch.autograd import Variable
from utils.utils import *
from utils.wassersteinLoss import *
from torchvision.utils import save_image
from torchvision import transforms
from matplotlib import pyplot as plt

import utils.augmentation


from tqdm import tqdm
import cv2



def get_args():
    train_arg_parser = argparse.ArgumentParser(description="parser")

    #################################################################
    #                   Training hyperparameters                    #
    #################################################################
    train_arg_parser.add_argument("--num_classes", type=int, default=10,
                                  help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.00005,
                                  help='')
    train_arg_parser.add_argument("--beta1", type=float, default=0.5,
                                  help='')
    train_arg_parser.add_argument("--beta2", type=float, default=0.999,
                                  help='')

    #################################################################
  

    #################################################################
    #                         Configurations                        #
    #################################################################


    train_arg_parser.add_argument('--gpu', default=[0, 1], type=int, nargs='+', help="PCI BUS IDs of the GPUs to use.")
    train_arg_parser.add_argument('--modality_indices', default=[0, 1, 2, 3], type=int, nargs='+', help="Modality indices")
    train_arg_parser.add_argument('--modalities', default=['heatmaps', 'limbs', 'optical_flow', 'rgb'], type=str, nargs='+', help="Modalities")
    train_arg_parser.add_argument('--video_folder', default=None, type=str, help="Video folder for the output video.")
    train_arg_parser.add_argument('--video_path', default=None, type=str, help="Video path of video to be processed.")

    #################################################################
    #                   Loading pre-trained models                  #
    #################################################################

    train_arg_parser.add_argument('--pretrained_model_G', default='/home/zmarinov/repos/L2A-OT/checkpoints/second_GAN/GAN_training_second_paper_lambdasG_iteration_27000.pth', type=str, help="Pre-trained Conditional Generator model (G).")


    args = train_arg_parser.parse_args()

    return args

def pad(x):
    x = "000000" + x
    return x[-5:]

class L2A_OT_Trainer(object):
    def __init__(self, args, devices):
        self.args = args
        self.devices = devices
        self.generator_device = self.devices[1] # For G, DGC
        self.classifier_device = self.devices[0] # For D, C
        print(args)



        self.modalities = self.args.modalities
        self.modality_indices = self.args.modality_indices
        channels = [1, 1, 3, 3]
        self.n_channels = sum([channels[i] for i in self.modality_indices])


        self.n_classes = args.num_classes
        self.num_domains = 4 # Must be fixed
        self.num_aug_domains = self.num_domains # these the novel domains K_n


        # model init
        print('Loading generator on GPU:', devices[1])
        # this is the conditional generator
        self.G = Generator(c_dim = 2 * self.num_domains).to(self.generator_device) # c_dim = K_n + K_s = 2 * K_s
        if self.args.pretrained_model_G is not None:
            self.G.load_state_dict(torch.load(self.args.pretrained_model_G))

        for name, param in self.G.named_parameters():
             param.requires_grad = False

        # self.D = Discriminator(c_dim = self.n_classes).to(device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.args.lr, (self.args.beta1, self.args.beta2))
    

    def useG(self, video_path):
        self.G.train()
        cap = cv2.VideoCapture(video_path)
        if not os.path.exists(self.args.video_folder):
            os.mkdir(self.args.video_folder)
        
        ret = True
        t= 0
        vid_transform = transforms.Compose([
            utils.augmentation.Scale(size=112, asnumpy=True),
            utils.augmentation.CenterCrop(size=112, consistent=True, asnumpy=True),
            utils.augmentation.ToTensor(),
            utils.augmentation.Normalize((0.5,), (0.5,))
            ])

        while ret:
            ret, frame = cap.read()
            if not ret:
               continue
            print(t)



            t += 1
            x_real = torch.from_numpy(frame).to(self.generator_device) #.unsqueeze(0).transpose(1, 3).transpose(2, 3) # Extract source modality for the input

            x_real = vid_transform([x_real.cpu().data.numpy()])[0].unsqueeze(0).to(self.generator_device)

            label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains) # Label of source domain
            label_org[:,self.modality_indices[0]] = 1.0
            label_org = label_org.to(self.generator_device)
            new_idx = self.num_domains + self.modality_indices[0]

            label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains)# label of target domain (shifted with just K_s)
            label_trg[:,new_idx] = 1.0
            label_trg = label_trg.to(self.generator_device)

            x_fake = self.G(x_real, label_trg)
            x_fake_denorm = self.denormalize(x_fake, self.generator_device) #[0].transpose(0, 2).transpose(0, 1)


            save_image(x_fake_denorm, os.path.join(self.args.video_folder, pad(str(t)) + '.png'))


            #writer_result.write(x_fake_denorm)


        return

    def denormalize(self,x, device):
        # x is a tensor
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        mean = torch.tensor(mean).cuda().to(device)
        std = torch.tensor(std).cuda().to(device)
        x *= std.view(1,3,1,1)
        x += mean.view(1, 3, 1, 1)
        return x

def main():
    args = get_args()
    devices = [torch.device("cuda:" + str(args.gpu[0])), torch.device("cuda:" + str(args.gpu[1]))]

    print("---------------------------------------------------")
    print("Using Devices:", devices)
    print("---------------------------------------------------")
    trainer = L2A_OT_Trainer(args, devices)


    trainer.useG(args.video_path)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
