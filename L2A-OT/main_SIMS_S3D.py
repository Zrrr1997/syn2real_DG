# Torch
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Torchvision
from torchvision.utils import save_image
from torchvision import transforms

# Utils 
from utils.datasets import *
from utils.logger import Logger
from utils.data_reader import *
from utils.utils import *
from utils.wassersteinLoss import *
from utils.action_encodings import sims_simple_dataset_encoding
import utils.augmentation
from utils.sims_dataset_video import SimsDataset_Video
from utils.sims_dataset_video_multiple_modalities import SimsDataset_Video_Multiple_Modalities

# Models
from resnet import *
from model import *
from lib.s3d import S3D

# Misc
import os
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm



def get_args():
    train_arg_parser = argparse.ArgumentParser(description="parser")

    #################################################################
    #                   Training hyperparameters                    #
    #################################################################
    train_arg_parser.add_argument("--batch_size", type=int, default=6,
                                  help="")
    train_arg_parser.add_argument("--validation_size", type=int, default=60,
                                  help="")
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
    train_arg_parser.add_argument("--num_iterations_D", type=int, default=10,
                                  help="Number of iterations to train the domain classifier.")
    train_arg_parser.add_argument("--num_iterations_G", type=int, default=10,
                                  help="Number of iterations to train the conditional generator.")

    #################################################################
    #                   Testing/Validation hyperparameters          #
    #################################################################
    train_arg_parser.add_argument("--test_every", type=int, default=100,
                                  help="Iteration cycle at which to test the task model.")
    train_arg_parser.add_argument("--t", type=int, default=0,
                                  help="Start iteration.")
    train_arg_parser.add_argument("--test_every_D", type=int, default=5,
                                  help="")
    train_arg_parser.add_argument("--save_img_every", type=int, default=1000,
                                  help="Iteration cycle at which to save sample images from the generator.")
    train_arg_parser.add_argument('--test_classifier_only', action='store_true', default=False,
                    help="Test only the task model classifier on the test dataset. Sets the requires_grad for D, DGC, and G to false.")
    train_arg_parser.add_argument('--train_classifier_only', action='store_true', default=False,
                    help="Train classifier DGC with a frozen generator G.")
    train_arg_parser.add_argument('--train_classifier_only_on_novel', action='store_true', default=False,
                    help="Train classifier DGC with a frozen generator G, but only on the NOVEL domains.")
    train_arg_parser.add_argument('--freeze_generator', action='store_true', default=False,
                    help="Freeze generator weights.")
    train_arg_parser.add_argument('--force_balanced', action='store_true', default=False,
                    help="Force evalutation with balanced accuracy.")
    train_arg_parser.add_argument('--no_val', action='store_true', default=False,
                    help="No validation.")
    #################################################################
    #                         Configurations                        #
    #################################################################
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='checkpoints',
                                  help='')
    train_arg_parser.add_argument('--gpu', default=[0, 1], type=int, nargs='+', help="PCI BUS IDs of the GPUs to use.")
    train_arg_parser.add_argument('--modality_indices', default=[0, 1, 2, 3], type=int, nargs='+', help="Modality indices")
    train_arg_parser.add_argument('--modalities', default=['heatmaps', 'limbs', 'optical_flow', 'rgb'], type=str, nargs='+', help="Modalities")
    train_arg_parser.add_argument('--results_folder', default=None, type=str, help="Results folder for the generated and reconstructed images.")
    train_arg_parser.add_argument("--exp_tag", type=str, default='_random-tag',
                                  help='Experiment tag for the tensorboard logs. Also the name of the directory for the models in ./checkpoints')
    train_arg_parser.add_argument("--num_workers", type=int, default=10,
                                  help="Number of workers for the DataLoaders.")

    #################################################################
    #                   Dataset directory arguments                 #
    #################################################################
    train_arg_parser.add_argument('--dataset_roots', default=None, type=str, nargs='+', help="The set of dataset root folders on which to train and validate on (e.g., /path/to/heatmaps /path/to/limbs /path/to/optical_flow /path/to/rgb).")
    train_arg_parser.add_argument('--dataset_roots_test', default=None, type=str, nargs='+', help="The set of dataset root folders on which to test on (e.g., /path/to/adl_heatmaps /path/to/adl_limbs /path/to/adl_optical_flow /path/to/adl_rgb).")

    #################################################################
    #                   Loading pre-trained models                  #
    #################################################################
    train_arg_parser.add_argument('--pretrained_model_DGC', default=None, type=str, help="Pre-trained S3D model for the DGC network (task model classifier)")
    train_arg_parser.add_argument('--pretrained_model_G', default=None, type=str, help="Pre-trained Conditional Generator model (G).")
    train_arg_parser.add_argument('--pretrained_model_C', default=None, type=str, help="Pre-trained S3D model for the C network (frozen Y^ classification network)")

    args = train_arg_parser.parse_args()

    return args

class L2A_OT_Trainer(object):
    def __init__(self, args, devices):
        # Setup GPU-devices
        self.args = args
        self.devices = devices
        self.generator_device = self.devices[1] # For G, DGC
        self.classifier_device = self.devices[0] # For D, C

        # Adam optimizer
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr = args.lr

        print(args)

        # Model save directories
        if not os.path.exists(args.model_path) and not args.test_classifier_only:
            os.makedirs(args.model_path)
        if not os.path.exists(os.path.join(args.model_path, args.exp_tag)) and not args.test_classifier_only:
            os.makedirs(os.path.join(args.model_path, args.exp_tag))


        # Modalities setup
        self.modalities = self.args.modalities
        self.modality_indices = self.args.modality_indices
        channels = [1, 1, 3, 3] # H, L, OF, RGB
        self.n_channels = sum([channels[i] for i in self.modality_indices])

   
        # Data augmentation
        vid_transform = transforms.Compose([
            utils.augmentation.Scale(size=112, asnumpy=True),
            utils.augmentation.CenterCrop(size=112, consistent=True, asnumpy=True),
            utils.augmentation.ToTensor(),
            utils.augmentation.Normalize((0.5,), (0.5,))
            ])

        # Datasets
        self.sims_train_dataset = SimsDataset_Video_Multiple_Modalities(dataset_root=None,
                             split_mode='train',
                             split_train_file=None,
                             vid_transform=vid_transform,
                             seq_len=16,
                             seq_shifts=None,
                             downsample_vid=1,
                             split_policy="frac",
                             sample_limit=None,
                             use_cache=True,
                             return_data=("vclip", "label"),
                             per_class_samples=None,
                             random_state=42,
                             n_channels=8,
                             color_jitter=False,
                             color_jitter_trans=None,
                             modalities=self.modalities,
                             dataset_roots=self.args.dataset_roots,
                             n_channels_each_modality=[1,1,3,3],
                             test_on_sims=False,
                             fine_tune_late_fusion=False)
        self.sims_val_dataset = SimsDataset_Video_Multiple_Modalities(dataset_root=None,
                             split_mode='val',
                             split_train_file=None,
                             vid_transform=vid_transform,
                             seq_len=16,
                             seq_shifts=None,
                             downsample_vid=1,
                             split_policy="frac",
                             sample_limit=None,
                             use_cache=True,
                             return_data=("vclip", "label"),
                             per_class_samples=None,
                             random_state=42,
                             n_channels=8,
                             color_jitter=False,
                             color_jitter_trans=None,
                             modalities=self.modalities,
                             dataset_roots=self.args.dataset_roots,
                             n_channels_each_modality=[1,1,3,3],
                             test_on_sims=False,
                             fine_tune_late_fusion=False)
        self.adl_test_dataset = SimsDataset_Video_Multiple_Modalities(dataset_root=None,
                             split_mode='test',
                             split_train_file=None,
                             vid_transform=vid_transform,
                             seq_len=16,
                             seq_shifts=None,
                             downsample_vid=1,
                             split_policy="frac",
                             sample_limit=None,
                             cache_folder="cache_ADL",
                             use_cache=True,
                             return_data=("vclip", "label"),
                             per_class_samples=None,
                             random_state=42,
                             n_channels=8,
                             color_jitter=False,
                             color_jitter_trans=None,
                             modalities=self.modalities,
                             dataset_roots=self.args.dataset_roots_test,
                             n_channels_each_modality=[1,1,3,3],
                             test_on_sims=True,
                             fine_tune_late_fusion=False)

        # Data loaders used for validation of the classifier
        print("Before train")
        train_sampler = torch.utils.data.SequentialSampler(self.sims_train_dataset)
        self.data_loader_train = iter(torch.utils.data.DataLoader(self.sims_train_dataset,
                                              batch_size=self.args.batch_size,
                                              sampler=train_sampler,
                                              shuffle=False,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              drop_last=True))
        print("Before val")
        val_sampler = torch.utils.data.SequentialSampler(self.sims_val_dataset)
        self.data_loader_val = iter(torch.utils.data.DataLoader(self.sims_val_dataset,
                                              batch_size=self.args.batch_size,
                                              sampler=val_sampler,
                                              shuffle=False,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              drop_last=True))
        print("Before test")
        test_sampler = torch.utils.data.SequentialSampler(self.adl_test_dataset)
        self.data_loader_test = torch.utils.data.DataLoader(self.adl_test_dataset,
                                              batch_size=self.args.batch_size,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              drop_last=True)
        print('Train/Val/Test sizes: ', len(self.data_loader_train), len(self.data_loader_val), len(self.data_loader_test))
        print('Single Frame shape', self.sims_train_dataset[0]['vclip'].shape)

        
        if self.args.results_folder is None:
            self.args.results_folder = self.args.exp_tag
        if not os.path.exists(os.path.join('results/' + self.args.results_folder)) and not self.args.test_classifier_only: # No results folder when evaluating...
            os.mkdir(os.path.join('results/' + self.args.results_folder))

        # Model hyperparameters
        self.n_classes = args.num_classes
        self.num_domains = 4 # Must be fixed
        self.num_aug_domains = self.num_domains # these the novel domains K_n
        self.Loss_cls = nn.CrossEntropyLoss() # [action] classification loss
        self.ReconstructionLoss = nn.L1Loss() # conditional generator loss
        self.lambda_domain = 1 # 1
        self.lambda_cycle = 10 # 2
        self.lambda_CE = 1 # 1
        self.ckpt_val = self.args.test_every # Validation frequency

        # Initialize domain generator
        print('Loading generator on GPU:', devices[1])
        self.G = Generator(c_dim = self.num_aug_domains + self.num_domains).to(self.generator_device) # c_dim = K_n + K_s = 2 * K_s
        if self.args.pretrained_model_G is not None:
            self.G.load_state_dict(torch.load(self.args.pretrained_model_G))
        if self.args.test_classifier_only or self.args.freeze_generator:
            for name, param in self.G.named_parameters():
                param.requires_grad = False

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, (self.beta1, self.beta2))

        # Running best loss / validation accuracy
        self.best_accuracy_val_D = 0
        self.best_accuracy_DGC = 0
        self.best_loss_DGC = 100000
    
    # Domain classifier 
    def D_init(self):
        self.D = resnet18(pretrained=False, num_classes=self.num_domains)
        weight = torch.load("./checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = self.D.state_dict()['fc.weight']
        weight['fc.bias'] = self.D.state_dict()['fc.bias']
        self.D.load_state_dict(weight)
        self.D = self.D.to(self.classifier_device)
        if self.args.test_classifier_only:
            for name, param in self.D.named_parameters():
                param.requires_grad = False
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lr, (self.beta1, self.beta2))

        return

    # Pre-trained classifier Y^, trained on the source domains -> e.g. S3D trained with early fusion on all the modalities
    def C_init(self): 
        print("Using the S3D pre-trained model.")
        self.C = S3D(num_class=self.args.num_classes, n_input_channels=self.n_channels)
        if self.args.pretrained_model_C:
            if 'tar' in self.args.pretrained_model_C and 'pth' not in self.args.pretrained_model_C:
                self.C.load_state_dict(torch.load(self.args.pretrained_model_C)['state'])
            elif 'pth' in self.args.pretrained_model_C:
                self.C.load_pretrained_unequal(self.args.pretrained_model_C)
        # Freeze pre-trained model
        for name, param in self.C.named_parameters():
            param.requires_grad = False
        self.C = self.C.to(self.classifier_device)
        #self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.lr, (self.beta1, self.beta2)) # Deprecated -> Left in case you don't have a pre-trained classifier

        return

    # Task model from the paper
    def DGC_init(self):
        self.DGC = S3D(num_class=self.args.num_classes, n_input_channels=self.n_channels)
        if self.args.pretrained_model_DGC:
            if 'tar' in self.args.pretrained_model_DGC and 'pth' not in self.args.pretrained_model_DGC:
                self.DGC.load_state_dict(torch.load(self.args.pretrained_model_DGC)['state'])
            elif 'pth' in self.args.pretrained_model_DGC:
                self.DGC.load_pretrained_unequal(self.args.pretrained_model_DGC)
        self.DGC = self.DGC.to(self.generator_device)
        if self.args.test_classifier_only:
            for name, param in self.DGC.named_parameters():
                param.requires_grad = False
        self.dgc_optimizer = torch.optim.Adam(self.DGC.parameters(), self.lr, (self.beta1, self.beta2))

        return

    # Loss with Sinkhort distance
    def Loss_distribution(self,x_ori,x_gen):
        _,f_ori = self.D(x_ori,latent_flag = True)
        _,f_gen = self.D(x_gen,latent_flag = True)
        C = cost_matrix(f_ori, f_gen).to(self.classifier_device)
        loss = sink(C, device=self.classifier_device)
        return loss


    def trainG(self, T, writer, start_iter):
        self.G.train()
        self.C.eval()
        self.D.eval()
        self.DGC_init()
        self.DGC.train()

        for t in range(start_iter, T):
            if self.args.test_classifier_only: # Test classifier and return
                no_val = self.args.no_val

                if not os.path.exists(os.path.join(self.args.logs, self.args.exp_tag)):
                    os.mkdir(os.path.join(self.args.logs, self.args.exp_tag))
                test_acc, test_acc_novel, bal_acc_test, bal_acc_test_novel = self.test_workflow_C(self.DGC, self.data_loader_test, self.args, t, self.generator_device, split='test') # Test task model
                if not self.args.no_val:
                    val_acc, val_acc_novel, bal_acc_val, bal_acc_val_novel = self.test_workflow_C(self.DGC, self.data_loader_val, self.args, t, self.generator_device, split='val') # Validate task model
                    train_acc, train_acc_novel, bal_acc_train, bal_acc_train_novel = self.test_workflow_C(self.DGC,self.data_loader_train, self.args, t, self.generator_device, split='train') # Validate task model
                print("[Normal Accuracies (Source)] =====> Test(ADL) Accuracy:", test_acc)
                print("[Balanced Accuracies (Source)] =====> Test(ADL) Accuracy:", bal_acc_test)
                print("[Normal Accuracies (Novel)] =====> Test(ADL) Accuracy:", test_acc_novel)
                print("[Balanced Accuracies (Novel)] =====> Test(ADL) Accuracy:", bal_acc_test_novel)
                
                if not self.args.no_val:
                    print("[Normal Accuracies (Source)] =====> Train(Sims) / Val(Sims) / Test(ADL) Accuracy:", train_acc, val_acc, test_acc)
                    print("[Balanced Accuracies (Source)] =====> Train(Sims) / Val(Sims) / Test(ADL) Accuracy:", bal_acc_train, bal_acc_val, bal_acc_test)
                    print("[Normal Accuracies (Novel)] =====> Train(Sims) / Val(Sims) / Test(ADL) Accuracy:", train_acc_novel, val_acc_novel, test_acc_novel)
                    print("[Balanced Accuracies (Novel)] =====> Train(Sims) / Val(Sims) / Test(ADL) Accuracy:", bal_acc_train_novel, bal_acc_val_novel, bal_acc_test_novel)

                    with open(os.path.join(self.args.logs, self.args.exp_tag, self.args.exp_tag + '_evaluation_accuracies.txt'), 'w') as f:
                        f.write(self.args.exp_tag + ' [Normal Accuracies (Source)] =====>  Train(Sims) / Val(Sims) / Test(ADL) Accuracy:' + str(train_acc) + ' ' + str(val_acc) + ' ' + str(test_acc) + '\n')
                        f.write(self.args.exp_tag + ' [Normal Accuracies (Novel)] =====>  Novel Train(Sims) / Val(Sims) / Test(ADL) Accuracy:' + str(train_acc_novel) + ' ' + str(val_acc_novel) + ' ' + str(test_acc_novel) + '\n')
                        f.write(self.args.exp_tag + ' [Balanced Accuracies (Source)] =====>  Train(Sims) / Val(Sims) / Test(ADL) Accuracy:' + str(bal_acc_train) + ' ' + str(bal_acc_val) + ' ' + str(bal_acc_test) + '\n')
                        f.write(self.args.exp_tag + ' [Balanced Accuracies (Novel)] =====>  Novel Train(Sims) / Val(Sims) / Test(ADL) Accuracy:' + str(bal_acc_train_novel) + ' ' + str(bal_acc_val_novel) + ' ' + str(bal_acc_test_novel) + '\n')
                
                return # Directly end cycle
            '''
            # TODO: fix this horrible code - remove the iterable and just use the data loader with batch_size 1 for training
            if t % self.args.test_every == 0: # reset data samplers to avoid StopIteration
                train_sampler = torch.utils.data.RandomSampler(self.sims_train_dataset)
                self.data_loader_train = iter(torch.utils.data.DataLoader(self.sims_train_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=train_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True))
                val_sampler = torch.utils.data.RandomSampler(self.sims_val_dataset)
                self.data_loader_val = iter(torch.utils.data.DataLoader(self.sims_val_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=val_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True))
                
                test_sampler = torch.utils.data.SequentialSampler(self.adl_test_dataset)
                self.data_loader_test = torch.utils.data.DataLoader(self.adl_test_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=test_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True)
            '''

            loss_CE = 0.0
            loss_diversity = 0.0
            generator_loss = 0.0

            # Sample random image sequence of the dataset
            # TODO: don't sample like this please
            random_sample_index = np.random.randint(0, len(self.sims_train_dataset))
            x_all = self.sims_train_dataset[random_sample_index]['vclip'].to(self.generator_device)
            labels_all = self.sims_train_dataset[random_sample_index]['label'].to(self.classifier_device)

            # Transpose sequence dimension with channel dimension to function as a batch
            heatmaps = x_all[0:1].transpose(0, 1)
            heatmaps = torch.cat((heatmaps, heatmaps, heatmaps), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            limbs = x_all[1:2].transpose(0, 1)
            limbs = torch.cat((limbs, limbs, limbs), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            optical_flow = x_all[2:5].transpose(0, 1)
            rgb = x_all[5:].transpose(0, 1)
            data = [heatmaps, limbs, optical_flow, rgb]


            for index in self.modality_indices: # go over all source domains
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                x_real = data[index].to(self.generator_device) # Extract source modality for the input

                label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains) # Label of source domain
                label_org[:,index] = 1.0
                label_org = label_org.to(self.generator_device)
                new_idx = self.num_domains + index

                label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains)# label of target domain (shifted with just K_s)
                label_trg[:,new_idx] = 1.0
                label_trg = label_trg.to(self.generator_device)

                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #
                # Original-to-target domain.


                x_fake = self.G(x_real, label_trg).to(self.classifier_device)
                x_real = x_real.to(self.classifier_device)
                if not self.args.train_classifier_only:
                    loss_novel = (self.Loss_distribution(x_real, x_fake) / len(self.modalities)).to(self.generator_device)

                x_fake = x_fake.to(self.generator_device)
                x_rec = self.G(x_fake, label_org)

                x_real = x_real.to(self.generator_device)
                if not self.args.train_classifier_only:
                    loss_cycle = self.ReconstructionLoss(x_rec, x_real) / len(self.modalities)
                    generator_loss += (self.lambda_cycle * loss_cycle - self.lambda_domain * loss_novel)
                    writer.add_scalar("Cycle "+ self.modalities[index] + "  Loss / Iteration", loss_cycle.item(), t)
                    writer.add_scalar("Novel "+ self.modalities[index] + "  Loss / Iteration", loss_novel.item(), t)

                # Use of classifier [pretrained Y^] - self.C
                x_fake = x_fake.to(self.classifier_device)

                if index == 0:
                # Store all fake and rec images
                    full_fakes = x_fake
                    if index in self.modality_indices:
                        fakes = torch.mean(x_fake, dim=1, keepdim=True)
                    recs = x_rec
                elif index == 1:
                    if index in self.modality_indices and self.modality_indices[0] != index:
                        fakes = torch.cat((fakes, torch.mean(x_fake, dim=1, keepdim=True)), 1) # Concatenate generated domains along channel dimension
                        full_fakes = torch.cat((full_fakes, x_fake), 1) 
                        recs = torch.cat((recs, x_rec), 1) 
                    elif self.modality_indices[0] == index: # Limbs first
                        fakes = torch.mean(x_fake, dim=1, keepdim=True)
                        full_fakes = x_fake
                        recs = x_rec
                else:
                    if index == self.modality_indices[0]: # Optical flow / RGB is first
                        full_fakes = x_fake
                        fakes = x_fake
                        recs = x_rec
                    else: # OF/RGB is not first
                        full_fakes = torch.cat((full_fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                        fakes = torch.cat((fakes, x_fake), 1)
                        recs = torch.cat((recs, x_rec), 1)

                del x_fake, x_rec, x_real, label_org, label_trg


            fakes = fakes.transpose(0, 1).unsqueeze(0)
            if not self.args.train_classifier_only:
                out_cls = self.C(fakes)
                loss_CE += self.Loss_cls(out_cls, labels_all.unsqueeze(0))


            factor = 0.0
            if not self.args.train_classifier_only and len(self.modality_indices) > 1:
                for i in range(len(self.modality_indices)):
                    for j in range(i, len(self.modality_indices)):
                        loss_diversity += self.Loss_distribution(full_fakes[:,3*i:3*i+3,:,:], full_fakes[:,3*j:3*j+3,:,:])
                        factor += 1.0
                loss_diversity = (loss_diversity / factor).to(self.generator_device)
                writer.add_scalar("Diversity Loss / Iteration", loss_diversity.item(), t)


            if not self.args.train_classifier_only:
                loss_CE = loss_CE.to(self.generator_device)
                total_loss = self.lambda_CE * loss_CE - self.lambda_domain * loss_diversity + generator_loss

            labels_all = labels_all.to(self.generator_device)
            fakes = fakes.to(self.generator_device)

            loss_DGC = 0.5 * self.Loss_cls(self.DGC(fakes), labels_all.unsqueeze(0)) 
            # TODO fix all this into one function
            if not self.args.train_classifier_only_on_novel:
                if self.modality_indices == [0]:
                    x_all = x_all[0:1]
                elif self.modality_indices == [1]:
                    x_all = x_all[1:2]
                elif self.modality_indices == [0, 1]:
                    x_all = x_all[0:2]
                elif self.modality_indices == [2]:
                    x_all = x_all[2:5]
                elif self.modality_indices == [3]:
                    x_all = x_all[5:8]
                elif self.modality_indices == [1,2]:
                    x_all = x_all[1:5]
                elif self.modality_indices == [2,3]:
                    x_all = x_all[2:8]
                elif self.modality_indices == [0,1,2]:
                    x_all = x_all[0:5] # H L OF
                elif self.modality_indices == [1,2,3]:
                    x_all = x_all[1:8] # L OF RGB
                elif self.modality_indices == [0, 2]:
                    x_all = torch.cat((x_all[0:1], x_all[2:5]), 0) # H OF
                elif self.modality_indices == [0, 3]:
                    x_all = torch.cat((x_all[0:1], x_all[5:8]), 0) # H RGB
                elif self.modality_indices == [1, 3]:
                    x_all = torch.cat((x_all[1:2], x_all[5:8]), 0) # H RGB
                elif self.modality_indices == [0, 1, 3]:
                    x_all = torch.cat((x_all[0:2], x_all[5:8]), 0) # H L RGB
                elif self.modality_indices == [0, 2, 3]:
                    x_all = torch.cat((x_all[0:1], x_all[2:8]), 0) # H OF RGB

                loss_DGC += 0.5 * self.Loss_cls(self.DGC(x_all.unsqueeze(0)), labels_all.unsqueeze(0)) 

            print('{}_Loss of task model  DGC:{}'.format(t, loss_DGC.item()))

            if loss_DGC.item() < self.best_loss_DGC:
                self.best_loss_DGC = loss_DGC.item()
                outfile = os.path.join(self.args.model_path, self.args.exp_tag, 'best_loss_DGC.tar')
                print("Saving new best-loss task model DGC!")
                torch.save({'ite': t, 'state': self.DGC.state_dict()}, outfile)

            loss_DGC.backward(retain_graph=True)
            self.dgc_optimizer.step()
            self.dgc_optimizer.zero_grad()

            if not self.args.train_classifier_only:
                total_loss.backward()
                self.g_optimizer.step() # update generator
                self.g_optimizer.zero_grad()
            
                print('{}_total_loss Generator  G:{}'.format(t, total_loss.item()))

                writer.add_scalar("Generator (Total) Loss / Iteration", total_loss.item(), t)

            writer.add_scalar("Classifier Loss / Iteration", loss_DGC.item(), t)

             
            if t % self.ckpt_val == 0: # Test task model

                if not self.args.no_val:
                    test_acc_normal, _, test_acc, test_acc_novel = self.test_workflow_C(self.DGC, self.data_loader_test, self.args, t, self.generator_device, split='test') # Test task model
                    val_acc_normal, _, val_acc, val_acc_novel = self.test_workflow_C(self.DGC, self.data_loader_val, self.args, t, self.generator_device, split='val') # Validate task model
                    train_acc_normal, _, train_acc, train_acc_novel = self.test_workflow_C(self.DGC, self.data_loader_train, self.args, t, self.generator_device, split='train') # Validate task model on training data
                    writer.add_scalars('Balanced Accuracy (Sims) TASK Classifier / Iteration', {'val_acc':val_acc, 'train_acc':train_acc}, t)
                    writer.add_scalars('Balanced Accuracy (ADL) TASK Classifier / Iteration', {'test_acc':test_acc}, t)
                    writer.add_scalars('Balanced Accuracy (Sims) TASK Classifier on Novel Domains / Iteration', {'val_acc_novel':val_acc_novel, 'train_acc_novel':train_acc_novel}, t)
                    writer.add_scalars('Balanced Accuracy (ADL) TASK Classifier on Novel Domains / Iteration', {'test_acc_novel':test_acc_novel}, t)
                    print("Train/Val/Test Accuracy (Source):", train_acc, val_acc, test_acc)

                # Denormalize modalities and save them as images
                if t % self.args.save_img_every == 0:
                    print("SAVING DGC", t)

                    if not self.args.train_classifier_only:
                        torch.save(self.G.state_dict(),  f="checkpoints/" + self.args.exp_tag + "/G_iteration_{}.pth".format(t))
                    torch.save(self.DGC.state_dict(),  f="checkpoints/" + self.args.exp_tag + "/DGC_iteration_{}.pth".format(t))

                    
                    denormalized_modalities = np.array([self.denormalize(data[i], data[i].get_device()) for i in self.modality_indices])
                    fake_modalities = np.array([full_fakes[:,3*i:3*i+3,:,:] for i in range(len(self.modality_indices))])
                    denormalized_fakes = np.array([self.denormalize(fake_modality, fake_modality.get_device()) for fake_modality in fake_modalities])


                    reconstructed_modalities = [recs[:,3*i:3*i+3,:,:] for i in range(len(self.modality_indices))]
                    denormalized_recs = np.array([self.denormalize(rec_modality, rec_modality.get_device()) for rec_modality in reconstructed_modalities])

                    for eva_idx in range(len(self.modality_indices)): # Go over all modalities
                        save_image(denormalized_modalities[eva_idx], 'results/' + self.args.results_folder + '/' + str(t) + '_' + self.args.exp_tag + self.modalities[eva_idx] + '_real.jpg')
                        save_image(denormalized_fakes[eva_idx], 'results/' + self.args.results_folder + '/' + str(t) + '_' + self.args.exp_tag + self.modalities[eva_idx] + '_fake.jpg')
                        save_image(denormalized_recs[eva_idx], 'results/' + self.args.results_folder + '/' + str(t) +  '_' + self.args.exp_tag + self.modalities[eva_idx] + '_rec.jpg')

        return
    ###############################################################################################################
    # Deprecated, because Classifier is fully pre-trained (left in case a new classifier is to be trained)
    '''
    def trainC(self,T, writer):
        self.C.train()

        for t in range(T):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
            random_sample_index = np.random.randint(0, len(self.sims_train_dataset))
            x_all = self.sims_train_dataset[random_sample_index]['vclip']


            labels_all = self.sims_train_dataset[random_sample_index]['label']


                # =================================================================================== #
                #                               2. Train the discriminator                            #
                # =================================================================================== #
             
            out_cls = self.C(x_all.unsqueeze(0).to(self.devices[0]))
            labels_all = labels_all.to(self.devices[0])

            loss_CE = self.Loss_cls(out_cls, labels_all.unsqueeze(0))

            writer.add_scalar("Training Loss / Iteration", loss_CE, t)

            # loss_CE = loss_CE/len(self.batImageGenTrainsDg)

            loss_CE.backward()
            self.c_optimizer.step()
            self.c_optimizer.zero_grad()
            print('{}_total_loss:{}'.format(t, loss_CE.item()))

            if t % self.ckpt_val == 0 and t > 0:
                val_acc, _ = self.test_workflow_C(self.C, self.data_loader_val, self.args, t, self.devices[1])
                train_acc, _ = self.test_workflow_C(self.C, self.data_loader_train, self.args, t, self.devices[1])
                writer.add_scalars('Train-Val Accuracy / Iteration', {'train_acc': train_acc, 'val_acc': val_acc}, t)
                print('Validation accuracy / Training Accuracy', val_acc, train_acc)
                # torch.save(self.D.state_dict(),  f="checkpoints/C_iteration_{}.pth".format(t))
        return
     '''
     ###############################################################################################################



    # 
    def trainD(self,T, writer):
        self.D.train()
        for t in range(T):
            if t % 4 == 0: # reset data samplers to avoid StopIteration
                train_sampler = torch.utils.data.RandomSampler(self.sims_train_dataset)
                self.data_loader_train = iter(torch.utils.data.DataLoader(self.sims_train_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=train_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True))
                val_sampler = torch.utils.data.RandomSampler(self.sims_val_dataset)
                self.data_loader_val = iter(torch.utils.data.DataLoader(self.sims_val_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=val_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True))
                test_sampler = torch.utils.data.RandomSampler(self.adl_test_dataset)
                self.data_loader_test = iter(torch.utils.data.DataLoader(self.adl_test_dataset,
                                                      batch_size=self.args.batch_size,
                                                      sampler=val_sampler,
                                                      shuffle=False,
                                                      num_workers=self.args.num_workers,
                                                      pin_memory=True,
                                                      drop_last=True))

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            random_sample_index = np.random.randint(0, len(self.sims_train_dataset))
            x_all = self.sims_train_dataset[random_sample_index]['vclip'].to(self.classifier_device)
            heatmaps = x_all[0:1].transpose(0, 1)
            heatmaps = torch.cat((heatmaps, heatmaps, heatmaps), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            limbs = x_all[1:2].transpose(0, 1)
            limbs = torch.cat((limbs, limbs, limbs), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            optical_flow = x_all[2:5].transpose(0, 1)
            rgb = x_all[5:].transpose(0, 1)
            data = [heatmaps, limbs, optical_flow, rgb]
            iteration_loss = 0.0
            for index in range(4): # go over all source domains ---> don't skip any modality
                x_real = data[index]
                domain_labels = torch.tensor(int(index)).repeat(x_real.size(0)).to(self.classifier_device) # domain label is just the index of the domain!

                # =================================================================================== #
                #                               2. Train the discriminator                            #
                # =================================================================================== #

                out_cls = self.D(x_real)
                loss_CE = self.Loss_cls(out_cls, domain_labels) / len(self.modalities)
                loss_CE.backward()
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
                iteration_loss += loss_CE.item()
            print('{}_total_loss D Domain Classifier:{}'.format(t, iteration_loss))
            writer.add_scalar('Training loss Domain Classifier / Iteration', iteration_loss, t)

            if t % self.args.test_every_D == 0:
                domain_classifier_accuracy = self.test_workflow_D(self.data_loader_val, self.args, t)
                domain_classifier_accuracy_train = self.test_workflow_D(self.data_loader_train, self.args, t)
                print('{}_Accuracy of D Domain Classifier:{}'.format(t, domain_classifier_accuracy))
                writer.add_scalars('Accuracy Domain Classifier / Iteration', {'val_acc':domain_classifier_accuracy, 'train_acc':domain_classifier_accuracy_train}, t)
                # torch.save(self.D.state_dict(),  f="checkpoints/D_iteration_{}.pth".format(t))

        return

    def test_workflow_C(self, model, sims_val, flags, ite, model_device=None, split='train'):
        if False:
            return 15.0, 15.0, 15.0, 15.0

        accuracy_val, accuracy_trg, balanced_acc_src, balanced_acc_trg = self.test_C(model=model, flags=flags, ite=ite,
                                 log_dir=flags.logs, test_set=sims_val, model_device=model_device, split=split)
        mean_acc = accuracy_val
        if mean_acc > self.best_accuracy_DGC and not flags.test_classifier_only: # Don't save if testing only
            self.best_accuracy_DGC = mean_acc
            outfile = os.path.join(flags.model_path, self.args.exp_tag, 'best_val_DGC.tar')
            print("Saving new best task model DGC!", mean_acc)
            torch.save({'ite': ite, 'state': model.state_dict()}, outfile)
        return mean_acc, accuracy_trg, balanced_acc_src, balanced_acc_trg


    def test_C(self, model, flags, ite, log_dir='logs/', test_set=None, model_device=None, split='None'): 
        # switch on the network test mode
        model.train()

        res_dict = {"vid_id": [], "label": [], "pred1": [], "pred2": [], "pred3": [], "pred4": [], "pred5": []}
        res_dict_target = {"vid_id": [], "label": [], "pred1": [], "pred2": [], "pred3": [], "pred4": [], "pred5": []}
        accuracy = 0
        accuracy_trg = 0

        # For balanced accuracy
        acc_classes_target = {}
        acc_classes_source = {}
        for c in range(10):
            acc_classes_target[c] = []
            acc_classes_source[c] = []

        with torch.no_grad():
            #for i in tqdm(range(self.args.validation_size)):
            i = 0
            for out in tqdm(test_set):
                i += 1
                #if i == flags.validation_size:
                #    break
                #out = next(test_set)
                
                if flags.modality_indices == [0]:
                    input_batch = out['vclip'].to(model_device)[:,0:1]
                elif flags.modality_indices == [1]:
                    input_batch = out['vclip'].to(model_device)[:,1:2]
                elif flags.modality_indices == [0, 1]:
                    input_batch = out['vclip'].to(model_device)[:,0:2]
                elif flags.modality_indices == [2]:
                    input_batch = out['vclip'].to(model_device)[:,2:5]
                elif flags.modality_indices == [3]:
                    input_batch = out['vclip'].to(model_device)[:,5:8]
                elif flags.modality_indices == [1,2]:
                    input_batch = out['vclip'].to(model_device)[:,1:5]
                elif flags.modality_indices == [2,3]: # OF RGB
                    input_batch = out['vclip'].to(model_device)[:,2:8]
                elif flags.modality_indices == [0,1,2]: # H L OF
                    input_batch = out['vclip'].to(model_device)[:,0:5] 
                elif flags.modality_indices == [0,1,2,3]: # H L OF RGB
                    input_batch = out['vclip'].to(model_device)
                elif flags.modality_indices == [1,2,3]: # H L OF RGB
                    input_batch = out['vclip'].to(model_device)[:,1:8]
                elif flags.modality_indices == [0,2]: # H OF
                    input_batch = out['vclip'].to(model_device)
                    input_batch = torch.cat((input_batch[:,0:1], input_batch[:,2:5]), 1)
                elif flags.modality_indices == [0,3]: # H RGB
                    input_batch = out['vclip'].to(model_device)
                    input_batch = torch.cat((input_batch[:,0:1], input_batch[:,5:8]), 1)
                elif flags.modality_indices == [1,3]: # L RGB
                    input_batch = out['vclip'].to(model_device)
                    input_batch = torch.cat((input_batch[:,1:2], input_batch[:,5:8]), 1)
                elif flags.modality_indices == [0,1,3]: # H L RGB
                    input_batch = out['vclip'].to(model_device)
                    input_batch = torch.cat((input_batch[:,0:2], input_batch[:,5:8]), 1)
                elif flags.modality_indices == [0,2,3]: # H OF RGB
                    input_batch = out['vclip'].to(model_device)
                    input_batch = torch.cat((input_batch[:,0:1], input_batch[:,2:8]), 1)

                predictions = model(input_batch)
                labels = out['label'].to(model_device)

                # Save ranking for Borda Count
                _, preds = torch.topk(predictions, k=5, dim=1)
                for k in range(preds.shape[1]):
                    res_dict[f"pred{k + 1}"].extend(list(preds[:, k].cpu().numpy()))
                res_dict["vid_id"].extend(out['vid_id'])
                res_dict["label"].extend(list(out['label'].numpy()))
                res_dict_target["vid_id"].extend(out['vid_id'])
                res_dict_target["label"].extend(list(out['label'].numpy()))


                predictions = torch.max(predictions, dim=1)[1]

                if flags.test_classifier_only or flags.train_classifier_only or flags.force_balanced:
                    predictions_target = model(self.transform_to_novel_batch(out['vclip']).to(model_device))
                    _, preds = torch.topk(predictions_target, k=5, dim=1)
                    for k in range(preds.shape[1]):
                        res_dict_target[f"pred{k + 1}"].extend(list(preds[:, k].cpu().numpy()))
                    predictions_target = torch.max(predictions_target, dim=1)[1]
                    assert predictions_target.shape == labels.shape
                    final_pred_target = (labels == predictions_target)
                    temp_var_trg = torch.sum(final_pred_target) / torch.numel(predictions_target)
                    temp_var_trg = temp_var_trg.cpu().data.numpy()
                    accuracy_trg += temp_var_trg
                    
                    # balanced accuracy
                    for j in range(flags.batch_size):
                        acc_classes_target[int(labels[j].cpu().data.numpy())].append(final_pred_target[j].cpu().data.numpy())
                assert predictions.shape == labels.shape
                final_pred_source = (labels == predictions)
                temp_var = torch.sum(final_pred_source) / torch.numel(predictions)
                temp_var = temp_var.cpu().data.numpy()
                accuracy += temp_var

                # balanced accuracy
                for j in range(flags.batch_size):
                    acc_classes_source[int(labels[j].cpu().data.numpy())].append(final_pred_source[j].cpu().data.numpy())
            #accuracy /= self.args.validation_size
            accuracy /= i

            balanced_acc_source = -1.0
            balanced_acc_target = -1.0
            if flags.test_classifier_only or flags.train_classifier_only or flags.force_balanced:
                if not os.path.exists(flags.logs):
                   os.mkdir(flags.logs)
                if not os.path.exists(os.path.join(flags.logs, flags.exp_tag)):
                   os.mkdir(os.path.join(flags.logs, flags.exp_tag))
                accuracy_trg /= i # Normal Accuracy
                actions = list({val: key for key, val in sims_simple_dataset_encoding.items()}.values())
                # Balanced Accuracy
                balanced_acc_source, ica_src = self.balanced_accuracy(acc_classes_source)
                balanced_acc_target, ica_trg = self.balanced_accuracy(acc_classes_target)
                print("Normal Accuracy Source / Novel:", accuracy, accuracy_trg)
                print("Balanced accuracy Source / Novel ", balanced_acc_source, balanced_acc_target)
                '''
                plt.bar(range(10), ica_src.values(), label='Balanced Accuracies Top1 for Source Domains')
                plt.title("Class accuracies on the source domains")
                plt.ylim(0, 1.0)
                plt.xticks(range(10), actions, rotation=45)
                plt.plot(np.ones(10) * balanced_acc_source, 'r--', label = 'Mean balanced accuracy Top1: ' + str(round(balanced_acc_source, 2)))
                plt.plot(np.ones(10) * accuracy, 'g--', label = 'Normal accuracy Top1: ' + str(round(accuracy, 2)))
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(flags.logs, flags.exp_tag, split + '_source_balanced_accs.svg'))
                plt.clf()
                plt.close()
                plt.bar(range(10), ica_trg.values(), label='Balanced Accuracies Top1 for Target Domains')
                plt.title("Class accuracies on the novel domains")
                plt.ylim(0, 1.0)
                plt.xticks(range(10), actions, rotation=45)
                plt.plot(np.ones(10) * balanced_acc_target, 'r--', label = 'Mean balanced accuracy Top1: ' + str(round(balanced_acc_target, 2)))
                plt.plot(np.ones(10) * accuracy_trg, 'g--', label = 'Normal accuracy Top1: ' + str(round(accuracy_trg, 2)))
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(flags.logs, flags.exp_tag, split + '_target_balanced_accs.svg'))
                plt.clf()
                plt.close()
                '''
        '''
        if split == 'test':
            res_df = pd.DataFrame(res_dict)
            res_df_target = pd.DataFrame(res_dict_target)
            if not os.path.exists(os.path.join(flags.logs, 'ranking', flags.exp_tag)):
                os.mkdir(os.path.join(flags.logs, 'ranking', flags.exp_tag))
            csv_save_path = os.path.join(flags.logs, 'ranking', flags.exp_tag, 'ranking.csv')
            csv_save_path_target = os.path.join(flags.logs, 'ranking', flags.exp_tag, 'ranking_target.csv')
            res_df.to_csv(csv_save_path, index=False)
            res_df_target.to_csv(csv_save_path_target, index=False)
            print('-------------------------------')
            print('Saved ranking to csv in:', csv_save_path)
            print('-------------------------------')
        '''

        return accuracy, accuracy_trg, balanced_acc_source, balanced_acc_target

    # Compute balanced accuracy for input dictionary with [0,0,1,0...] arrays for each class
    def balanced_accuracy(self, acc_classes):
        #print(acc_classes)
        balanced_accuracy = 0.0
        #return 15.0, [15.0] * 10
        eval_class_count = 0
        individual_class_accuracies = {}
        for c in range(10):
            if len(acc_classes[c]) > 0:
                eval_class_count += 1
                balanced_accuracy += sum(acc_classes[c]) / len(acc_classes[c]) # Add the individual class accuracy
                individual_class_accuracies[c] = sum(acc_classes[c]) / len(acc_classes[c])
            else:
                individual_class_accuracies[c] = -0.1
        if eval_class_count == 0:
            eval_class_count += 1
        return (balanced_accuracy / eval_class_count), individual_class_accuracies

    # Transform a batch of source domain images to novel domains with the Conditional Generator (G)
    def transform_to_novel_batch(self, x_all_batch):
        self.G.train() 

        with torch.no_grad():
            x_all_batch = x_all_batch.transpose(1, 2) # switch Sequence and Channel dimension to combine sequence and batch dim
            x_all_batch = x_all_batch.contiguous().view(-1, *x_all_batch.shape[2:]).transpose(0, 1) # (batch_size, 8, 16, 112, 122) -> (8, 16*batch_size, 112, 112)

            heatmaps = x_all_batch[0:1].transpose(0, 1)
            heatmaps = torch.cat((heatmaps, heatmaps, heatmaps), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            limbs = x_all_batch[1:2].transpose(0, 1)
            limbs = torch.cat((limbs, limbs, limbs), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
            optical_flow = x_all_batch[2:5].transpose(0, 1)
            rgb = x_all_batch[5:].transpose(0, 1)
            data = [heatmaps, limbs, optical_flow, rgb]

        
            for index in self.args.modality_indices: # go over all source domains

                x_real = data[index].to(self.generator_device) # Extract source modality for the input

                label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains) # Label of source domain
                label_org[:,index] = 1.0
                label_org = label_org.to(self.generator_device)
                new_idx = self.num_domains + index

                label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains)# label of target domain (shifted with just K_s)
                label_trg[:,new_idx] = 1.0
                label_trg = label_trg.to(self.generator_device)

                # Original-to-target domain.

                x_fake = self.G(x_real, label_trg).to(self.classifier_device)


                # Store all fake images
                if index == 0 or (self.args.modality_indices[0] == 1 and index == 1):
                    full_fakes = x_fake
                    #fakes = x_fake[:,0,:,:].unsqueeze(1)
                    fakes = torch.mean(x_fake, dim=1, keepdim=True)
                elif index == 1:
                    full_fakes = torch.cat((full_fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                    #fakes = torch.cat((fakes, x_fake[:,0,:,:].unsqueeze(1)), 1) # Concatenate generated domains along channel dimension
                    fakes = torch.cat((fakes, torch.mean(x_fake, dim=1, keepdim=True)), 1) # Concatenate generated domains along channel dimension
                else:
                    if index == self.args.modality_indices[0]:
                        full_fakes = x_fake
                        fakes = x_fake
                    else:

                        full_fakes = torch.cat((full_fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                        fakes = torch.cat((fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                del x_fake, x_real, label_org, label_trg

            fakes = fakes.view(self.args.batch_size, 16, self.n_channels, 112, 112).transpose(1, 2) # bring back to original dimensions and swap channels and sequence
            #x_all_batch = x_all_batch.transpose(0, 1) # (C, B*S, W, H) -> (B*S, C, W, H)
            #x_all_batch = x_all_batch.view(self.args.batch_size, self.n_channels, 16, 112, 112)
            assert fakes.shape == (self.args.batch_size, self.n_channels, 16, 112, 112)
        return fakes


    def test_workflow_D(self, sims_val, flags, ite):

        accuracy_val = self.test_D(test_set=sims_val, flags=flags, ite=ite,
                                  log_dir=flags.logs, log_prefix='val_index_{}'.format(0))

        if accuracy_val > self.best_accuracy_val_D:
            self.best_accuracy_val_D = accuracy_val

            if not os.path.exists(flags.model_path) and not flags.test_classifier_only:
                os.makedirs(flags.model_path)

                outfile = os.path.join(flags.model_path, self.args.exp_tag, 'best_val_D.tar')
                print("Saving new best domain classifier D!")
                torch.save({'ite': ite, 'state': self.D.state_dict()}, outfile)
        return accuracy_val

    def test_D(self, flags, ite, log_prefix, log_dir='logs/', test_set=None):
        # switch on the network test mode
        model_device = self.classifier_device
        self.D.eval()

        accuracy = 0

        with torch.no_grad():
            for i in tqdm(range(self.args.validation_size)):
                x_all = next(test_set)['vclip'][0].to(model_device)
                heatmaps = x_all[0:1].transpose(0, 1)
                heatmaps = torch.cat((heatmaps, heatmaps, heatmaps), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
                limbs = x_all[1:2].transpose(0, 1)
                limbs = torch.cat((limbs, limbs, limbs), 1) # pad to 3 channels ---> Generator requires 3-channel input, e.g. RGB
                optical_flow = x_all[2:5].transpose(0, 1)
                rgb = x_all[5:].transpose(0, 1)
                data = [heatmaps, limbs, optical_flow, rgb]

                for index in range(4):
                    x_real = data[index]
                    
                    domain_labels = torch.tensor(int(index)).repeat(x_real.size(0)).to(self.classifier_device) # domain label is just the index of the domain!
                    predictions = self.D(x_real)
                    predictions = torch.max(predictions, dim=1)[1]
                    assert predictions.shape == domain_labels.shape
                    temp_var = torch.sum(domain_labels == predictions) / torch.numel(predictions)
                    temp_var = temp_var.cpu().data.numpy()
                    accuracy += temp_var
            accuracy /= (self.args.validation_size * len(self.modalities))

        return accuracy

    def G_visualize(self):
        self.G.eval()
        self.G.load_state_dict(torch.load('checkpoints/G_iteration_450.pth'))

        for index, batImageGenVal in enumerate(self.batImageGenVals):
            x_real, cls = batImageGenVal.get_images_labels_batch()
            x_real = torch.from_numpy(np.array(x_real, dtype=np.float32))

            # wrap the inputs and labels in Variable
            x_real = Variable(x_real, requires_grad=False).cuda()
            x_real = x_real.to(self.devices[1])
            label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_org[:, index] = 1.0
            new_idx = self.num_domains + index

            label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_trg[:, new_idx] = 1.0
            x_fake = self.G(x_real, label_trg)
            x_rec = self.G(x_fake, label_org)
            x_real = self.denormalize(x_real)
            x_fake = self.denormalize(x_fake)
            x_rec = self.denormalize(x_rec)
            save_image(x_real[0],'results/real.jpg')
            save_image(x_fake[0],'results/fake.jpg')
            save_image(x_rec[0], 'results/rec.jpg')

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
    if args.test_classifier_only:
        writer = SummaryWriter('./runs/eval/' + args.exp_tag)
    else:
        writer = SummaryWriter('./runs/' + args.exp_tag)
    devices = [torch.device("cuda:" + str(args.gpu[0])), torch.device("cuda:" + str(args.gpu[1]))]

    print("---------------------------------------------------")
    print("Using Devices:", devices)
    print("---------------------------------------------------")
    trainer = L2A_OT_Trainer(args, devices)

    trainer.C_init()
    #trainer.trainC(10000, writer) # Not needed for our S3D model
    trainer.D_init()
    if not args.test_classifier_only:
        trainer.trainD(args.num_iterations_D, writer)
        trainer.trainG(args.num_iterations_G, writer, args.t)
    # trainer.G_visualize()
    else:
        
        print("---------------------------------------------------")
        print("Testing only!")
        print("---------------------------------------------------")
        trainer.trainG(1, writer, args.t)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
