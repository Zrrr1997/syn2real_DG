import os
from collections import deque
from datetime import datetime
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from utils.augmentation import Denormalize

plt.switch_backend('agg')


def copy_file_backup(original, dest, iteration=0, max_iter=20):
    if iteration > max_iter:
        raise ValueError("Too many history files present.")

    dest_and_suff = dest + (f".{iteration}" if iteration > 0 else "")

    if not os.path.exists(dest_and_suff):
        copyfile(original, dest_and_suff)
    else:
        copy_file_backup(original, dest, iteration + 1)


def write_out_images(img_seqs, writer, iteration):
    de_normalize = Denormalize()

    for i, img_seq in enumerate(img_seqs):
        # C, Sqlen, H, W
        img_dim = img_seq.shape[-1]
        img_seq_hor = img_seq.transpose(0, 1).contiguous().view(-1, 3, img_dim, img_dim)
        num_imgs = img_seq_hor.shape[0]

        img_grid = vutils.make_grid(img_seq_hor, nrow=num_imgs)
        de_norm_imgs = de_normalize.denormalize(img_grid)

        writer.add_image(f'input_seq_{i}', de_norm_imgs, iteration)


def save_checkpoint(state, model_name="model_last.pth.tar", model_path='models'):
    model_last = os.path.join(model_path, model_name)

    torch.save(state, model_last + ".tmp")  # This way there is always a valid file.
    torch.save(state, model_last)

    os.remove(model_last + ".tmp")


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    total = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # Returns the k largest indices (not values) per prediction.
    pred = pred.t()  # Each column a sample, 5 largest 'classes'
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # Each column contains each class maximally one, so maximally a single 'True' per column
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / total))
    return res


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, locality=5):
        self.locality = locality
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0):
        if val is None:
            self.val = None
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if self.locality > 0:
            self.local_history.append(val)
            if len(self.local_history) > self.locality:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''

    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count': 0, 'correct': 0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                  % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


class ConfusionMeter(object):
    '''compute and show confusion matrix'''

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p, t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        width, height = self.mat.shape
        import matplotlib
        import seaborn as sn
        import pandas as pd
        #matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        sn.set(rc={'figure.figsize': (1.4*0.4*width, 0.4*width)})

        cats = range(width) if dictionary is None else [dictionary[i] for i in range(width)]

        df_cm = pd.DataFrame(self.mat, cats, cats)

        sn.set(font_scale=0.8)  # for label size
        sn.heatmap(df_cm, cmap=plt.cm.jet, annot=True, fmt=".0f", annot_kws={"size": 6})  # font size

        plt.savefig(path, format='svg')

        return

        plt.figure(dpi=600, figsize=(1.4 * width, 1.4 * width))
        plt.imshow(self.mat,
                   cmap=plt.cm.jet,
                   interpolation=None,
                   extent=(0.5, np.shape(self.mat)[0] + 0.5, np.shape(self.mat)[1] + 0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y + 1, x + 1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i + 1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i + 1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))



def balanced_acc(csv_path='results_test_sims_frac_s3d_32f.csv'):
	sims_simple_dataset_encoding = {
	    "Cook":         0,
	    "Drink":        1,
	    "Eat":          2,
	    "Getup":        3,
	    "Readbook":     4,
	    "Usecomputer":  5,
	    "Usephone":     6,
	    "Usetablet":    7,
	    "Walk":         8,
	    "WatchTV":      9
	    }
	actions = list({val: key for key, val in sims_simple_dataset_encoding.items()}.values())
	import pandas as pd
	import numpy as np
	from matplotlib import pyplot as plt
	df = pd.read_csv(csv_path)
	acc_dict = {}
	classes = sorted(df['label'].unique())
	balanced_accs = []
	for c in classes:
		x = df[df['label'] == c] 
		balanced_acc = len(x[x['pred1'] == c]) / len(x)
		balanced_accs.append(balanced_acc)
	mean_balanced_acc = sum(balanced_accs) / len(balanced_accs)
	normal_acc = len(df[df['label'] != df['pred1']]) / len(df)

	print('Balanced accuracies:', balanced_accs)
	print('Mean balanced accuracy:', mean_balanced_acc)
	print('Normal accuracy:', normal_acc)

	acc_dict['Action_Name'] = actions
	acc_dict['Action_ID'] = range(len(actions))
	acc_dict['Balanced_Accuracies'] =  np.array(balanced_accs)
	acc_dict['Mean_Balanced_Accuracy'] = mean_balanced_acc
	acc_dict['Normal Accuracy'] = normal_acc

	acc_df = pd.DataFrame(acc_dict)

	plt.bar(range(10), balanced_accs, label='Balanced Accuracies')
	plt.plot(np.ones(10) * mean_balanced_acc, 'r--', label = 'Mean balanced accuracy',)
	plt.xticks(range(10), actions, rotation=45)
	plt.legend()
	plt.tight_layout()
	plt.savefig(csv_path[:-3] +'_balanced_accuracies.png')

	acc_df.to_csv(csv_path[:-3] + '_balanced_accuracies.csv', index=False)


def write_out_checkpoint(epoch, iteration, model, optimizer, args, train_loss, train_acc, val_loss, val_acc,
                         best_train_loss, best_train_acc, best_val_loss, best_val_acc, alt_path=None):
    state = {'epoch':           epoch + 1,
             'iteration':       iteration,
             'model_vid':       args.model_vid,
             'state_dict':      model.state_dict(),
             'optimizer':       optimizer.state_dict(),
             'lr':              args.lr,
             'best_train_loss': best_train_loss,
             'best_train_acc':  best_train_acc,
             'best_val_loss':   best_val_loss,
             'best_val_acc':    best_val_acc,
             'train_loss':      train_loss,
             'train_acc':       train_acc,
             'val_loss':        val_loss,
             'val_acc':         val_acc
             }

    path = args.model_path if alt_path is None else alt_path

    save_checkpoint(state=state,
                    model_name="model_last.pth.tar",
                    model_path=path)

    if args.save_best_val_acc and val_acc == best_val_acc:
        save_checkpoint(state=state,
                        model_name="model_best_val_acc.pth.tar",
                        model_path=path)

    if args.save_best_val_loss and val_loss == best_val_loss:
        save_checkpoint(state=state,
                        model_name="model_best_val_loss.pth.tar",
                        model_path=path)

    if args.save_best_train_acc and train_acc == best_train_acc:
        save_checkpoint(state=state,
                        model_name="model_best_train_acc.pth.tar",
                        model_path=path)

    if args.save_best_train_loss and train_loss == best_train_loss:
        save_checkpoint(state=state,
                        model_name="model_best_train_loss.pth.tar",
                        model_path=path)

    if args.save_interval is not None and epoch % args.save_interval == 0:
        save_checkpoint(state=state,
                        model_name=f"model_ep_{epoch}.pth.tar",
                        model_path=path)

        if os.path.exists("model_best_train_acc.pth.tar"):
            copyfile("model_best_train_acc.pth.tar", f"model_best_train_acc_ep_{epoch}.pth.tar")

    with open(os.path.join(path, "training_state.log"), 'a') as f:
        f.write(f"Epoch: {(epoch + 1):4} | "
                f"Loss Train: {train_loss:1.4f} | Acc Train: {train_acc:1.4f} | "
                f"Loss Val: {val_loss:1.4f} | Acc Val: {val_acc:1.4f} | "
                f"Best Loss Train: {best_train_loss:1.4f} | Best Acc Train: {best_train_acc:1.4f} | "
                f"Best Loss Val: {best_val_loss:1.4f} | Best Acc Val: {best_val_acc:1.4f}\n")
