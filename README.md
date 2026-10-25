# module08_pytorch_mnist
MNIST Classification Using PyTorch

# MNIST Classification with PyTorch

This project implements a convolutional neural network (ConvNet) for MNIST handwritten digit classification using PyTorch. The model follows Module 7B, Section 3, and uses argparse to configure training hyperparameters through the command line.

## Project Files
- `utilities.py`: Model architecture, dataset loading, training, evaluation, and plotting.
- `run.py`: Main script using argparse to configure training.
- `figure.png`: Generated training loss and accuracy plots.

## Dependencies
- Python
- PyTorch
- Torchvision
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision matplotlib
```

## Usage
Run with default parameters:
```bash
python run.py
```

Run with custom parameters:
```bash
python run.py -e 3 -b 500 -l 0.0005
```

| Argument | Description | Default |
|---|---|---|
| `-e`, `--num_epochs` | Number of training epochs | 2 |
| `-b`, `--batch_size` | Training batch size | 1000 |
| `-l`, `--learning_rate` | Adam learning rate | 0.001 |

The program automatically downloads MNIST, trains the model using CPU or GPU when available, evaluates classification accuracy, and saves the results to `figure.png`.

## Experimental Results

| Configuration | Training Accuracy | Test Accuracy |
|---|---|---|
| Default (2 epochs, batch 1000, LR 0.001) | 97.31% | 97.40% |
| Custom (3 epochs, batch 500, LR 0.0005) | 98.98% | 98.62% |

Results may vary due to random initialization and data shuffling.