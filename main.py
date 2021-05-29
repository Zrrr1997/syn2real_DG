import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils.augmentation
from datasets.adl_dataset import ADLDataset
from datasets.sims_dataset import SimsDataset
from datasets.sims_dataset_video import SimsDataset_Video
from lib.C3D import C3D
from lib.i3d_hassony import I3D, Unit3Dpy
from lib.s3d import S3D
from testing.test_video_stream import test_video_stream as test_vs
from training.train_video_stream import training_loop_video_stream as train_vs
from utils.utils import copy_file_backup

# This way, cuda optimizes for the hardware available, if input size is always equal.
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--exp-num', default=None, type=int,
                    help='Experiment number. This is used in the folder name: exp-{exp-num}. '
                         'Filled automatically if not set.')

parser.add_argument('--exp-suffix', default=None, type=int,
                    help='Experiment suffix for annotations. '
                         'If set, this is also used in the folder name: exp-{exp-num}_{exp-suffix}.')

parser.add_argument('--exp-root', default=os.path.expanduser("./experiments"), type=str,
                    help='Root folder to which to save the experiment results.')

parser.add_argument('--gpu', default=[0, 1], type=int, nargs='+', help="PCI BUS IDs of the GPUs to use.")
parser.add_argument('--loader_workers', default=16, type=int,
                    help='Number of data loader workers to pre load batch data. Main thread used if 0.')

parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run.')
parser.add_argument('--batch_size', default=28, type=int, help="Make batch size divisible by GPU count.")

parser.add_argument('--dataset', default='sims', choices=["sims", "adl", "nturgbd", "sims_video"], type=str)
parser.add_argument('--num_classes', default=10, type=int, help='Number of classes for the classification task.')

parser.add_argument('--split-policy', default='frac',
                    choices=["frac", "file", "cross-subject", "cross-view-1", "cross-view-2"], type=str,
                    help="Decide if splits are made randomly with fraction or chosen via explicit files.")
parser.add_argument('--train_split_file', default=None, type=str, help="File with the ids for the train split.")
parser.add_argument('--val_split_file', default=None, type=str, help="File with the ids for the val split.")
parser.add_argument('--split-val-frac', default=0.1, type=float, help="Fraction of the val split.")
parser.add_argument('--split-test-frac', default=0.1, type=float, help="Fraction of the test split.")

# Sample Settings
parser.add_argument('--sampling_shift', default=None, type=int,
                    help='Defines a fixed shift for sampling clips from a full length video.')
parser.add_argument('--max_samples', default=None, type=int, help='Dataset size limit. Useful for debugging.')

parser.add_argument('--seq_len', default=32, type=int, help='This is the base number of frames per clip. '
                                                            'The actual number might be smaller due to downsampling.')

parser.add_argument('--ds_vid', default=1, type=int, help='Downsampling rate. (use every n-th frame).')

parser.add_argument('--img_dim', default=128, type=int, help="The image dimension of the frames (will be squared).")

parser.add_argument('--modality', default="heatmaps", type=str, help="The modality on which to train on.")

parser.add_argument('--model_vid', default="s3d", type=str, choices=["s3d", "s3dg", "i3d", "r2+1d", "r18"],
                    help="The model for the video backbone.")
parser.add_argument('--pretrained_model_i3d', default=None, type=str, help="Pre-trained I3D model.")
parser.add_argument('--pretrained_model_s3d', default=None, type=str, help="Pre-trained S3D model.")
parser.add_argument('--pretrained_model_c3d', default=None, type=str, help="Pre-trained C3D model.")

parser.add_argument('--model_body', default="skelemotion", type=str, choices=["skelemotion"],
                    help="The model for the body motion backbone.")

parser.add_argument('--score_function', default='cross-entropy', choices=["cross-entropy"], type=str,
                    help="The loss function.")

parser.add_argument('--training_focus', default='all', type=str, choices=["all", "last"],
                    help='This applies to fine-tuning. Whether to train the full model or only the last layers.')
parser.add_argument('--optim', default="Adam", type=str, choices=["Adam", "SGD"], help='Which optimizer is used.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--sgd_momentum', default=0.9, type=float, help='SGD Momentum.')

parser.add_argument('--lr_scheduler_steps', type=int, default=None, nargs='+',
                    help='The epochs on which a lr decrease is performed (one or multiple ints.)')
parser.add_argument('--lr_step_decrease', type=int, default=0.1, help='LR multiplier on scheduler step.')

parser.add_argument('--print_freq', default=10, type=int, help='Iteration frequency of printing output during training')

parser.add_argument('--no_cache', action='store_true', default=False, help='Do not use cached dataset info (reindex).')
parser.add_argument('--test_only', action='store_true', default=False, help='Skip training.')
parser.add_argument('--test_prediction_only', action='store_true', default=False, help='Skip training.')
parser.add_argument('--test_on_sims', action='store_true', default=False, help='Test domain adaptation on a dataset (e.g. ADL) with a model trained on Sims')
parser.add_argument('--ignore_old_metrics', action='store_true', default=False, help='Ignore old metrics.')
parser.add_argument('--new_optimizer', action='store_true', default=False, help='Ignore stored optimizer on resuming.')
parser.add_argument('--last_layer_only', action='store_true', default=False, help='Only train last layer.')
parser.add_argument('--skip_val', action='store_true', default=False, help='Skip validation ->')
parser.add_argument('--per_class_samples', default=None, type=int, help='Per class samples for few shot.')

parser.add_argument('--save_best_val_loss', type=bool, default=False, help='Save model with best Val Loss.')
parser.add_argument('--save_best_val_acc', type=bool, default=True, help='Save model with best Val Accuracy.')
parser.add_argument('--save_best_train_loss', type=bool, default=True, help='Save model with best Train Loss.')
parser.add_argument('--save_best_train_acc', type=bool, default=False, help='Save model with best Train Accuracy.')
parser.add_argument('--save_interval', type=bool, default=200,
                    help='Save current model on fixed intervals.')
parser.add_argument('--save_interval_best_val_acc', type=bool, default=None,
                    help='Save best val acc model on fixed intervals.')

parser.add_argument('--training_stream', type=str, default="vid",
                    choices=["vid"], help='Streams for training.')

parser.add_argument('--resume', default=None, type=str, help='path of model to resume from')

parser.add_argument('--start_epoch', default=0, type=int, help='Explicit epoch to start from.')
parser.add_argument('--start_iteration', default=0, type=int, help='Explicit iteration to start form.')

parser.add_argument('--dataset-video-root', default=os.path.expanduser("~/datasets/sims_dataset/frames"), type=str,
                    help="Root folder of the video frames for this dataset.")
parser.add_argument('--dataset-skele-motion-root', default=os.path.expanduser("~/datasets/sims_dataset/skele-motion"),
                    type=str, help="Root folder for the skelemotion data for this dataset.")

parser.add_argument('--aug_rotation_range', default=[20.], type=float, nargs='+')

parser.add_argument('--aug_hue_range', default=[0.5], type=float, nargs='+')
parser.add_argument('--aug_saturation_range', default=[1], type=float, nargs='+')
parser.add_argument('--aug_value_range', default=[0.8], type=float, nargs='+')
parser.add_argument('--aug_contrast_range', default=[0.0], type=float, nargs='+')

parser.add_argument('--aug_crop_min_area', default=0.05, type=float)
parser.add_argument('--aug_crop_max_area', default=1., type=float)
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='Do not apply data augmentation to the training dataset.')


def argument_checks(args):
    """
    This function performs non-obvious checks on the arguments provided. Fail fast.
    """
    assert args.batch_size % len(args.gpu) == 0, "Batch size has to be divisible by GPU count."
    assert args.loader_workers >= 0

    return args


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()

    args = argument_checks(args)

    augmentation_settings = {
        "rot_range":      (-abs(min(args.aug_rotation_range)), abs(max(args.aug_rotation_range))),
        "hue_range":      (0. - abs(min(args.aug_hue_range)), 0. + abs(max(args.aug_hue_range))),
        "sat_range":      (1. - abs(min(args.aug_saturation_range)), 1. + abs(max(args.aug_saturation_range))),
        "val_range":      (1. - abs(min(args.aug_value_range)), 1. + abs(max(args.aug_value_range))),
        "hue_prob":       1.,
        "cont_range":     (1. - abs(min(args.aug_contrast_range)), 1. + abs(max(args.aug_contrast_range))),
        "crop_arr_range": (args.aug_crop_min_area, args.aug_crop_max_area)
        }

    print(args)
    print(augmentation_settings)
    if args.test_only:
       print('Testing only!')

    # setup tools
    args.log_path, args.model_path, args.exp_path = set_path(args)

    # Setup cuda
    cuda_device, args.gpu = check_and_prepare_cuda(args.gpu)

    # Prepare model
    model = select_and_prepare_model(args)

    # Data Parallel uses a master device (default gpu 0)
    # and performs scatter gather operations on batches and resulting gradients.
    # Distributes batches on mutiple devices to train model in parallel automatically.
    # -> Batch size has to be divisible by number of gpus.
    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model = model.to(cuda_device)  # Sends model to device 0, other gpus are used automatically.

    check_and_prepare_parameters(model, args)

    if args.lr_scheduler_steps is not None:
        lr_schedule = step_schedule
    else:
        lr_schedule = None

    optimizer = initialize_optimizer(model, args)

    # Prepare Loss
    # Contrastive loss can be implemented with CrossEntropyLoss with vector similarity.
    criterion = torch.nn.CrossEntropyLoss()  # Be aware that this includes a final Softmax.

    if args.resume:  # Resume a training which was interrupted.
        model, optimizer, args = prepare_on_resume(model, optimizer, args)

    # Normal case, no resuming, not pretraining.
    if not hasattr(args, 'best_train_loss'):
        args.best_train_loss = None
    if not hasattr(args, 'best_train_acc'):
        args.best_train_acc = None
    if not hasattr(args, 'best_val_loss'):
        args.best_val_loss = None
    if not hasattr(args, 'best_val_acc'):
        args.best_val_acc = None

    vid_transform, test_transform = prepare_augmentations(augmentation_settings, args)

    writer_train, writer_val = get_summary_writers(args.log_path, f"exp_{args.exp_num}_{args.exp_suffix}")

    write_settings_file(args, args.exp_path)

    if not args.test_only:
        train_loader, train_len = get_data(vid_transform, 'train', args)
        val_loader, val_len = get_data(vid_transform, 'val', args)
    test_loader, test_len = get_data(test_transform, 'test', args)

    if args.training_stream == "vid":
        if not args.test_only:
            train_vs(model, optimizer, lr_schedule, criterion, train_loader, val_loader, writer_train, writer_val, args,
                     cuda_device)

        test_vs(test_loader, model, criterion, epoch=args.epochs, cuda_device=cuda_device, args=args)


def step_schedule(lr, ep, args):
    step_eps = sorted(args.lr_scheduler_steps)

    i, _ = next(filter(lambda ise: ep < ise[1], enumerate(step_eps)), (len(step_eps), float("inf")))

    return lr * (args.lr_step_decrease ** i)


def select_and_prepare_model(args):
    if args.model_vid == 's3d':
        print("Using the S3D model.")
        model = S3D(num_class=args.num_classes)
        if args.pretrained_model_s3d:
            model.load_pretrained_unequal(args.pretrained_model_s3d)
    elif args.model_vid == "i3d":
        print("Using the I3D model.")
        if not args.pretrained_model_i3d:
            model = I3D(num_classes=args.num_classes, use_softmax=False)
        else:
            # Loading the pretrained Kinetics model. In this case we assume 400 classes first
            # and change it after loading the weights.
            model = I3D(num_classes=10, use_softmax=False)
            model.load_pretrained_unequal(args.pretrained_model_i3d)
            print("Loaded pretrained weights for I3D.")

            # This changes the number of classes after loading the pretrained weights.
            model.conv3d_0c_1x1 = Unit3Dpy(
                 in_channels=1024,
                 out_channels=args.num_classes,
                 kernel_size=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 use_bn=False)

    elif args.model_vid == "c3d":
        print("Using the C3D model.")
        model = C3D(use_softmax=False)

        if args.pretrained_model_c3d:
            print("Loading pretrained model for C3D.")
            model.load_state_dict(torch.load(args.pretrained_model_c3d))
            print("Successfully loaded model.")

        model.fc8 = torch.nn.Linear(4096, args.num_classes)
    else:
        raise ValueError(f'model {args.model_vid} not implemented!')

    return model


def prepare_augmentations(augmentation_settings, args):

    transform_train = transforms.Compose([
        utils.augmentation.RandomRotation(degree=augmentation_settings["rot_range"]),
        utils.augmentation.RandomSizedCrop(size=args.img_dim, crop_area=augmentation_settings["crop_arr_range"],
                                           consistent=True, force_inside=True),
        utils.augmentation.ColorJitter(brightness=augmentation_settings["val_range"], contrast=0,
                                       saturation=augmentation_settings["sat_range"],
                                       hue=augmentation_settings["hue_range"]),
        utils.augmentation.ToTensor(),
        utils.augmentation.Normalize()
        ])
    if args.no_augmentation:
        print("Not using any data augmentation for the heatmap/limbs/optical_fow modality")
        transform_train = transforms.Compose([
            utils.augmentation.Scale(size=args.img_dim),
            utils.augmentation.CenterCrop(size=args.img_dim, consistent=True),
            utils.augmentation.ToTensor(),
            utils.augmentation.Normalize()
            ])

    transform_test = transforms.Compose([
        utils.augmentation.Scale(size=args.img_dim),
        utils.augmentation.CenterCrop(size=args.img_dim, consistent=True),
        utils.augmentation.ToTensor(),
        utils.augmentation.Normalize()
        ])

    return transform_train, transform_test


def get_data(vid_transform, mode='train', args=None, random_state=42):
    if args.dataset == 'sims':
        dataset = SimsDataset(dataset_root=args.dataset_video_root,
                              split_mode=mode,
                              split_train_file=args.train_split_file,
                              vid_transform=vid_transform,
                              seq_len=args.seq_len,
                              seq_shifts=args.sampling_shift,
                              downsample_vid=args.ds_vid,
                              split_policy=args.split_policy,
                              sample_limit=args.max_samples,
                              use_cache=not args.no_cache,
                              per_class_samples=args.per_class_samples,
                              random_state=random_state)
    elif args.dataset == 'adl':
        dataset = ADLDataset(dataset_root=args.dataset_video_root,
                             split_mode=mode,
                             split_train_file=args.train_split_file,
                             vid_transform=vid_transform,
                             seq_len=args.seq_len,
                             seq_shifts=args.sampling_shift,
                             downsample_vid=args.ds_vid,
                             split_policy=args.split_policy,
                             sample_limit=args.max_samples,
                             use_cache=not args.no_cache,
                             per_class_samples=args.per_class_samples,
                             test_on_sims=args.test_on_sims,
                             random_state=random_state)
    elif args.dataset == 'sims_video':
        dataset = SimsDataset_Video(dataset_root=args.dataset_video_root,
                             split_mode=mode,
                             split_train_file=args.train_split_file,
                             vid_transform=vid_transform,
                             seq_len=args.seq_len,
                             seq_shifts=args.sampling_shift,
                             downsample_vid=args.ds_vid,
                             split_policy=args.split_policy,
                             sample_limit=args.max_samples,
                             use_cache=not args.no_cache,
                             per_class_samples=args.per_class_samples,
                             random_state=random_state,
                             modality=args.modality)
    else:
        raise ValueError('dataset not supported')

    if mode == 'train':
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=args.loader_workers,
                                              pin_memory=True,
                                              drop_last=True)

    return data_loader, len(data_loader)


def check_and_prepare_cuda(device_ids):
    # NVIDIA-SMI uses PCI_BUS_ID device order, but CUDA orders graphics devices by speed by default (fastest first).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(dev_id) for dev_id in device_ids])

    print('Cuda visible devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print('Available device count: {}'.format(torch.cuda.device_count()))

    device_ids = list(range(torch.cuda.device_count()))  # The device ids restart from 0 on the visible devices.

    print("Note: Device ids are reindexed on the visible devices and not the same as in nvidia-smi.")

    for i in device_ids:
        print("Using Cuda device {}: {}".format(i, torch.cuda.get_device_name(i)))

    print("Cuda is available: {}".format(torch.cuda.is_available()))

    cudev = torch.device('cuda')

    return cudev, device_ids


def check_and_prepare_parameters(model, args):
    named_params = model.named_parameters()

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        if args.last_layer_only:
            if args.model_vid == "s3d":
                if "fc.0" not in name:
                    param.requires_grad = False
            if args.model_vid == "i3d":
                if "conv3d_0c_1x1" not in name:
                    param.requires_grad = False

        print(name, param.requires_grad)
    print('=================================\n')

    print(f"Number of parameters for this model: {count_parameters(model)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_optimizer(model, args):
    params = model.module.parameters()

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd, amsgrad=False)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.wd)
    else:
        raise ValueError

    return optimizer


def prepare_on_resume(model, optimizer, args):
    if not os.path.isfile(args.resume):
        print("####\n[Warning] no checkpoint found at '{}'\n####".format(args.resume))
        raise FileNotFoundError
    else:
        print("=> loading resumed checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

        if not args.model_vid == checkpoint['model_vid']:
            print(f"WARNING: Loading wights of a different model: {checkpoint['model_vid']} (file) vs {args.model_vid}")

        #  args.lr = checkpoint['lr']

        args.start_epoch = checkpoint['epoch']
        args.start_iteration = checkpoint['iteration']

        if args.ignore_old_metrics:
            args.best_train_loss = None
            args.best_train_acc = None
            args.best_val_loss = None
            args.best_val_acc = None
        else:
            args.best_train_loss = checkpoint['best_train_loss']
            args.best_train_acc = checkpoint['best_train_acc']
            args.best_val_loss = checkpoint['best_val_loss']
            args.best_val_acc = checkpoint['best_val_acc']

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if "module.prototypes" not in checkpoint['state_dict']:  # Backwards compytibility for model with linear layer.
            if "module.prototypes.weight" in checkpoint['state_dict']:
                model.state_dict()["module.prototypes"].copy_(checkpoint['state_dict']["module.prototypes.weight"])
                print(
                    "Loaded parameter from module.prototypes.weight to module.prototypes. "
                    "This is necessary when loading an older model due to "
                    + "a change in the model architecture.")

        if args.new_optimizer:
            optimizer = initialize_optimizer(model, args)
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        model.cuda()
        # from utils import optimizer_to
        # optimizer_to(optimizer, cuda_device)

        print("=> loaded resumed checkpoint (epoch {}) '{}' ".format(args.start_epoch, args.resume))

        return model, optimizer, args


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        if args.exp_num is None and os.path.exists(args.exp_root):
            # We need to find the next unused number.
            exps = os.listdir(args.exp_root)
            import re
            exp_nums = [int(re.match(r"exp_(\d+).*", ex).group(1)) for ex in exps if re.match(r"exp_(\d+).*", ex)]
            if len(exp_nums) > 0:
                args.exp_num = max(exp_nums) + 1
            else:
                args.exp_num = 1

        exp_path = os.path.join(f"{args.exp_root}", f"exp_{args.exp_num}"
                                + ("_{args.exp_suffix}" if args.exp_suffix is not None else ""))

    log_path = os.path.join(exp_path, 'logs')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return log_path, model_path, exp_path


def get_summary_writers(img_path, prefix):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tboard_str = '{time}-{mode}-{prefix}'
    val_name = tboard_str.format(prefix=prefix, mode="val", time=time_str)
    train_name = tboard_str.format(prefix=prefix, mode="train", time=time_str)

    writer_val = SummaryWriter(log_dir=os.path.join(img_path, val_name))
    writer_train = SummaryWriter(log_dir=os.path.join(img_path, train_name))

    print(f"\n### Tensorboard Path###\n{img_path}\n")

    return writer_train, writer_val


def write_settings_file(args, exp_path):
    args_d = vars(args)
    training_description = ["{}: {}".format(key, args_d[key]) for key in sorted(args_d.keys()) if
                            args_d[key] is not None]
    training_description = "\n".join(training_description)

    target_file = os.path.join(exp_path, "training_description.txt")

    if os.path.exists(target_file):
        copy_file_backup(target_file, target_file)  # First make a backup of old file.

    with open(target_file, 'w') as f:
        import subprocess
        label = subprocess.check_output(["git", "describe", "--always"]).decode("utf-8").strip()

        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        f.write(f"Start of training: {time_str}\n")

        f.write("Git describe of repo: {}".format(label))

        f.write("\n\n")

        f.write(training_description)


if __name__ == '__main__':
    main()
