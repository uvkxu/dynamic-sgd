import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

random_seed = 42
torch.manual_seed(random_seed)


class MNIST_Digits:

    train_data = datasets.MNIST(
        root="data", train=True, transform=ToTensor(), download=True
    )

    test_data = datasets.MNIST(
        root="data", train=False, transform=ToTensor(), download=True
    )

    loaders = {
        "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }


from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url
import tarfile
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class CIFAR10:

    def __init__(self, val_size=50, batch_size=100, subset_size=None):

        self.dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        self.data_dir = "./data/cifar10"

        self.download_data()

        self.dataset = ImageFolder(self.data_dir + "/train", transform=ToTensor())

        if subset_size is not None:
            if val_size > subset_size:
                raise ValueError(
                    f"Validation size ({val_size}) cannot be larger than subset size ({subset_size})."
                )
            total_indices = torch.randperm(len(self.dataset)).tolist()
            subset_indices = total_indices[:subset_size]
            self.dataset = Subset(self.dataset, subset_indices)

        from torch.utils.data import random_split

        train_size = len(self.dataset) - val_size
        train_ds, val_ds = random_split(self.dataset, [train_size, val_size])

        self.train_dl = DataLoader(
            train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

    def download_data(self):
        download_url(self.dataset_url, "./")
        with tarfile.open("./cifar10.tgz", "r:gz") as tar:
            tar.extractall(path="./data")

    def show_example(self, img, label):
        import matplotlib

        matplotlib.rcParams["figure.facecolor"] = "#ffffff"

        print("Label: ", self.dataset.classes[label], "(" + str(label) + ")")
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    def show_example_by_idx(self, idx):
        self.show_example(*self.dataset[idx])

    def show_batch(self, dl=None):
        if dl is None:
            dl = self.train_dl

        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break
        plt.show()


class CIFAR100:

    def __init__(self, val_size=50, batch_size=4096, subset_size=None):
        self.dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"
        self.data_dir = "./data/cifar100"

        # Uncomment to download the dataset if it's not already in the dir
        # self.download_data()

        self.dataset = ImageFolder(self.data_dir + "/train", transform=ToTensor())

        if subset_size is not None:
            if val_size > subset_size:
                raise ValueError(
                    f"Validation size ({val_size}) cannot be larger than subset size ({subset_size})."
                )
            total_indices = torch.randperm(len(self.dataset)).tolist()
            subset_indices = total_indices[:subset_size]
            self.dataset = Subset(self.dataset, subset_indices)

        from torch.utils.data import random_split

        train_size = len(self.dataset) - val_size
        train_ds, val_ds = random_split(self.dataset, [train_size, val_size])

        self.train_dl = DataLoader(
            train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

    def download_data(self):
        download_url(self.dataset_url, "./")
        with tarfile.open("./cifar100.tgz", "r:gz") as tar:
            tar.extractall(path="./data")

    def show_example(self, img, label):
        import matplotlib

        matplotlib.rcParams["figure.facecolor"] = "#ffffff"

        print("Label: ", self.dataset.classes[label], "(" + str(label) + ")")
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    def show_example_by_idx(self, idx):
        self.show_example(*self.dataset[idx])

    def show_batch(self, dl=None):
        if dl is None:
            dl = self.train_dl

        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break
        plt.show()


if __name__ == "__main__":
    cf = CIFAR100()
    print(len(cf.dataset))

    print("done")
