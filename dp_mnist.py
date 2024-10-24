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
from opacus import PrivacyEngine
from utils.trainers import DynamicSGD
import logging
from wideresnet import WideResNet

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

    # add algorithm option
    parser.add_argument(
        "--algo",
        default="DPSGD",
        type=str,
        help="algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)",
    )

    # Clipping thresholds for DiceSGD
    parser.add_argument(
        "--C", default=1, nargs="+", type=float, help="clipping threshold"
    )

    parser.add_argument("--save_results", default=True)
    parser.add_argument("--optimizer", default="SGD")
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
        "--lr", type=float, default=0.15, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="force CPU training"
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


model = WideResNet(depth=28, num_classes=10).to(device)

OPTIMIZERS = {
    "SGD": optim.SGD(model.parameters(), lr=args.lr),
    "SGDM": optim.SGD(model.parameters(), lr=args.lr, momentum=0.1),
    "Adam": optim.Adam(model.parameters(), lr=args.lr),
}

optimizer = OPTIMIZERS[args.optimizer]


if __name__ == "__main__":
    # for name, param in model.named_parameters():
    # print(name, param.requires_grad)

        # Initialize the CIFAR10 class
    cifar10_data = CIFAR10(
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
        "sgd",
        0.3,
        0.8
        )
    
    test_losses = dysgd.test_losses
    train_losses = dysgd.train_losses

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

    # Save losses to a CSV
    if args.save_experiment:
        loss_file = os.path.join(experiment_dir, "losses.csv")
        with open(loss_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Test Loss"])
            for epoch, (train_loss, test_loss) in enumerate(
                zip(train_losses, test_losses), 1
            ):
                writer.writerow([epoch, train_loss, test_loss])

    # Plot and save the loss curves
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Curves")

    if args.save_experiment:
        plt.savefig(os.path.join(experiment_dir, "loss_curves.png"))

    # Save model
    if args.save_experiment:
        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
    