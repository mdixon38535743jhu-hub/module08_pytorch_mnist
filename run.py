
import argparse
from utilities import ConvNet, load_data, train_model, plot

def main():
    parser = argparse.ArgumentParser(description='MNIST Classification with PyTorch')
    parser.add_argument('-e', '--num_epochs', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.batch_size)
    model = ConvNet()
    loss_list, acc_train, acc_test = train_model(model, train_loader, test_loader, args.num_epochs, args.learning_rate)
    plot(loss_list, acc_train, acc_test)

if __name__ == '__main__':
    main()