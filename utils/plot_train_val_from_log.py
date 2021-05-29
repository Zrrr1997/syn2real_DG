from matplotlib import pyplot as plt
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None, type=str,
                    help='Name of model to plot validation and train accuracy.')
parser.add_argument('--log_path', default=None, type=str, 
                    help='Path of logs to plot from.')


args = parser.parse_args()
assert args.model_name is not None and args.log_path is not None

f = open(os.path.join(args.log_path, 'training_state.log'), 'r')
lines = f.readlines()
train_acc = [float(line.split('|')[2].split(' ')[3]) for line in lines]
val_acc = [float(line.split('|')[4].split(' ')[3])  for line in lines]
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.1)
plt.title(args.model_name)
plt.grid()
plt.savefig(os.path.join(args.log_path, args.model_name + '_val_train_acc.svg'))

