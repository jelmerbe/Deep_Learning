import argparse, os, pickle
import torch

import HW3_methods # imports training file 

# Part (a) - setup for parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=100, help='batchsize')
parser.add_argument('--device', type=str, default='cuda', help='device to be used')
parser.add_argument('--num_epochs', type=int, default=200, help='device to be used')
parser.add_argument('--print_freq', type=int, default=5, help='device to be used')
parser.add_argument('--log', type=str, default='log.txt', help='the log file name, appended to save_dir')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--dropout', type=float, default=0.50, help='Dropout Rate')
parser.add_argument('--n_channels', type=int, default=5, help='Channels')


# all parsed arguments are stored in dictionary 
args = parser.parse_args()

if args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

lr = args.lr
dropout = args.dropout
n_channels = args.n_channels

# Run entire model with given parameters 

data_train, data_test = HW3_methods.load_data(batch_size=args.batchsize)
model = HW3_methods.CIFAR10Model(num_outputs=10, p=dropout, n_channels=n_channels)
model.to(device)
print(f"dropout={model.p}, n_channels={model.n_channels}, lr={lr}, device={device}")
_, train_accuracies = HW3_methods.train(model, data_train, lr=lr, batch_size=args.batchsize, device=device, num_epochs=args.num_epochs, print_freq=args.print_freq)
mean_test_accuracy = HW3_methods.test(model, data_test, batch_size=args.batchsize)
print(f"mean test accuracy = {mean_test_accuracy}%.")