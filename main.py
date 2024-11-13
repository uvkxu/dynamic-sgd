import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import csv
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from data_handlers import CIFAR10
from utils.trainers_new import DynamicSGD
# from utils.trainers import DynamicSGD
from wideresnet import WideResNet
from resnet import ResNet20
from ema_pytorch import EMA

"""
TODO: implement Parameter averaging using EMA
with https://github.com/lucidrains/ema-pytorch/tree/main/ema_pytorch
beta = 0.9999 as said in paper Unlocking High Accuracy
"""

matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a differentially private neural network"
    )

    parser.add_argument("--model", default="WideResNet")

    # add algorithm option
    parser.add_argument(
        "--algo",
        default="DPSGD",
        type=str,
        help="algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)",
    )

    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        help="optimizer you want for the training: sgd, adam, rmsprop",
    )

    # Clipping thresholds for DiceSGD
    parser.add_argument(
        "--C", default=1, nargs="+", type=float, help="clipping threshold"
    )

    parser.add_argument("--save_results", default=True)
    parser.add_argument(
        "--subset_size",
        default=None,
        type=int,
        help="check optimizer and model capacity",
    )
    parser.add_argument("--dry_run", default=False, help="check a single sample")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )

    parser.add_argument(
        "--epsilon", type=float, default=3, help="epsilon = privacy budget (default: 1)"
    )

    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs to train (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="force CPU training"
    )

    parser.add_argument(
        "--dp", type=bool, default=True, help="Train with DP (True) or without DP (False)"
    )

    parser.add_argument(
        "--new", type=bool, default=True, help="Train with opacus==0.14.0 (False) or opacus==1.5.2 (True)"
    )

    parser.add_argument(
        "--sens_decay", type=float, default=0.3, help="Set the sensitivity decay rate between 0 and 1"
    )

    parser.add_argument(
        "--mu_decay", type=float, default=0.75, help="Set the decay rate between 0 and 1"
    )

    parser.add_argument(
        "--save_experiment",
        action="store_true",
        default=True,
        help="Save experiment details",
    )
    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")


model = WideResNet(depth=16, num_classes=10, widen_factor=4).to(device) if args.model == "WideResNet" else ResNet20(num_classes=10, num_groups=16)
ema = EMA(
    model,
    beta = 0.9999,              # exponential moving average factor
    update_after_step = 100,    # only after this number of .update() calls will it start updating
    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)


if __name__ == "__main__":
    # for name, param in model.named_parameters():
    # print(name, param.requires_grad)

        # Initialize the CIFAR10 class
    cifar10_data = CIFAR10(
        val_size=10000,
        batch_size=args.batch_size,
        subset_size=args.subset_size
    )

    # Access the DataLoaders
    train_dl = cifar10_data.train_dl
    test_dl = cifar10_data.val_dl

    delta = 10**-5


    # dr_sens = np.linspace(0.1,0.7,4)
    # dr_mus = np.linspace(0.2,0.8,4)
    dysgd = DynamicSGD(
        model,
        train_dl,
        test_dl,
        args.batch_size,
        args.epsilon,
        delta,
        args.epochs,
        args.C,
        device,
        args.lr,
        args.optimizer,
        args.sens_decay,
        args.mu_decay,
        ema,
        args.dp
        )
    
    test_losses = dysgd.test_losses
    train_losses = dysgd.train_losses
    test_accuracies = dysgd.test_accuracies
    train_accuracies = dysgd.train_accuracies

    # Create directory for experiment
    if args.save_experiment:
        current_date = datetime.now().strftime("%Y-%b-%d %Hh%Mmin")
        experiment_dir = f"./experiments/{current_date}"
        os.makedirs(experiment_dir, exist_ok=True)

        # Save args to a file
        args_file = os.path.join(experiment_dir, "config.txt")
        with open(args_file, "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

    # Save losses and accuracies to a CSV
    csv_file = os.path.join(experiment_dir, "metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"])
        for epoch in range(1, args.epochs + 1):
            train_loss = train_losses[epoch - 1]
            test_loss = test_losses[epoch - 1]
            train_accuracy = train_accuracies[epoch - 1]
            test_accuracy = test_accuracies[epoch - 1]
            writer.writerow([epoch, train_loss, test_loss, train_accuracy, test_accuracy])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Curves")

    # Plot and save the accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, args.epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Testing Accuracy Curves")

    # Save the plots as a file
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "metrics_curves.png"))

    # Save model
    if args.save_experiment:
        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
    