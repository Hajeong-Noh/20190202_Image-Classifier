import functions
import argparse

parser = argparse.ArgumentParser(description='parser_for_train')
parser.add_argument('data_dir',  default="./flowers/", type=str)
parser.add_argument('--arch', dest="arch", default="vgg16", type=str)
parser.add_argument('--hidden_units', dest="hidden_units", default = 256, type=int)
parser.add_argument('--dropout', dest="dropout", default=0.2)
parser.add_argument('--learning_rate', dest="learning_rate", default=0.001)
parser.add_argument('--gpu', dest="gpu", default=True, type=bool)
parser.add_argument('--epochs', dest="epochs", default=2, type=int)
parser.add_argument('--print_every', dest="print_every", default=10, type=int)
parser.add_argument('--save_dir', dest="save_dir", default="./checkpoint.pth", type=str)
parsed = parser.parse_args()
data_dir = parsed.data_dir
arch = parsed.arch
hidden_units = parsed.hidden_units
dropout = parsed.dropout
learning_rate = parsed.learning_rate
gpu = parsed.gpu
epochs = parsed.epochs
print_every = parsed.print_every
save_dir = parsed.save_dir

trainloader, validloader, class_to_idx = functions.load_data(data_dir)
model, criterion, optimizer = functions.setup(arch, hidden_units, dropout, learning_rate, gpu)
functions.train_network(model, criterion, optimizer, trainloader, validloader, epochs,  print_every, gpu)
functions.save_checkpoint(save_dir, arch, hidden_units, dropout, learning_rate, gpu, model, optimizer, class_to_idx)
