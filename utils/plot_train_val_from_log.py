from matplotlib import pyplot as plt
import argparse, os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None, type=str,
                    help='Name of model to plot validation and train accuracy.')
parser.add_argument('--log_path', default=None, type=str, 
                    help='Path of logs to plot from.')
parser.add_argument('--print_best_val_acc', action="store_true", default=False, 
                    help='Print the epoch with the best validation accuracy without plotting anything.')
parser.add_argument('--max_epoch', default=200, type=int,
                    help='Maximum epoch to reach when plotting or printing.')


args = parser.parse_args()
assert args.model_name is not None and args.log_path is not None

f = open(os.path.join(args.log_path, 'training_state.log'), 'r')
lines = f.readlines()
train_acc = [float(line.split('|')[2].split(' ')[3]) for line in lines]
val_acc = [float(line.split('|')[4].split(' ')[3])  for line in lines]
if args.print_best_val_acc:
    print("Best validation accuracy at epoch: ", np.argmax(np.array(val_acc[:args.max_epoch])))
    exit()
plt.plot(train_acc[:args.max_epoch], label='Training Accuracy')
plt.plot(val_acc[:args.max_epoch], label = 'Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.1)
plt.title(args.model_name)
plt.grid()
plt.savefig(os.path.join(args.log_path, args.model_name + '_val_train_acc.svg'))

