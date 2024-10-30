import math
import numpy as np
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .privacy_engine import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from .GaussianCalibrator import calibrateAnalyticGaussianMechanism
import math
from .poisson_sampler import poisson_sampler
from .mu_search import mu0_search,cal_step_decay_rate
from scipy.stats import norm
from scipy import optimize
from ema_pytorch import EMA

from opacus.schedulers import LambdaGradClip, LambdaNoise

class DynamicSGD(): 
    def __init__(
            self, 
            model,
            train_dl,
            test_dl,
            batch_size,
            epsilon,
            delta,
            epochs,
            C,
            device,
            lr,
            method,
            decay_rate_sens = None, 
            decay_rate_mu = None,
            ema=None,
            dp = True):
        if method == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif method == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif method == "rmsprop":
            optimizer = torch.optim.RMSProp(model.parameters(), lr=lr)
        elif method == "adagrad":
            optimizer == torch.optim.Adagrad(model.parameters(), )
        else:
            raise RuntimeError("Unknown Optimizer!")
        
        self.test_dl = test_dl  # Testing data loader
        self.batch_size = batch_size  # Size of each batch
        max_per_sample_grad_norm = C  # Some constant or hyperparameter
        self.device = device  # Device to train on (e.g., 'cpu' or 'cuda')
        self.dp = dp
        step = 0

        self.test_losses = []
        self.test_accuracies = []

        self.train_losses = []
        self.train_accuracies = []
        
        num_data = len(train_dl.dataset)
        print(f'Training_dataset length: {num_data}')

        sampling_rate = batch_size/num_data
        iteration = int(epochs/sampling_rate)
        
        if delta is None:
            delta = 1.0/num_data
        mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = 1, tol = 1.e-12)
        mu_t = math.sqrt(math.log(mu**2/(sampling_rate**2*iteration)+1))
        sigma = 1/mu_t

        if decay_rate_mu is not None:
            decay_rate_mu = cal_step_decay_rate(decay_rate_mu,iteration)
            mu_0 = mu0_search(mu, iteration, decay_rate_mu, sampling_rate,mu_t=mu_t)
            
        if decay_rate_sens is not None:
            decay_rate_sens = cal_step_decay_rate(decay_rate_sens,iteration)

        
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
            sample_rate=sampling_rate,
            poisson_sampling=True,
        )

        self.clip_scheduler = LambdaGradClip(
            self.optimizer,
            scheduler_function=lambda step: max_per_sample_grad_norm * (decay_rate_sens)**step
        )

        # Noise scheduler: decay the noise multiplier similarly
        self.noise_scheduler = LambdaNoise(
            self.optimizer,
            noise_lambda=lambda step: 1/(mu_0/(decay_rate_mu**(step)))
        )

        for epochs in range(1, epochs + 1):
            step = self.train(step, ema)
            if ema is not None:
                ema.update_model_with_ema()
            self.test()

    def train(self, step, ema=None):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        losses = []
        correct = 0 
        total = 0
        if self.dp == False:
            for _batch_idx, (data, target) in enumerate(tqdm(self.train_dl)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                step += 1
                pred = output.argmax(
                                    dim=1, keepdim=True
                                ) 
                correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            for _batch_idx, (data, target) in enumerate(tqdm(self.train_dl)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.clip_scheduler.step()
                self.noise_scheduler.step()
                losses.append(loss.item())
                step += 1
                pred = output.argmax(
                                    dim=1, keepdim=True
                                ) 
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.shape[0]

                if ema is not None:
                    ema.update()
            acc = 100.0*correct/ total
            self.train_accuracies.append(acc)
        self.train_losses.append(np.mean(losses))
        return step

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(self.test_dl):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_dl.dataset)

        self.test_losses.append(test_loss)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.test_dl.dataset),
                100.0 * correct / len(self.test_dl.dataset),
            )
        )
        self.test_accuracies.append(100.0 * correct / len(self.test_dl.dataset))
        return correct / len(self.test_dl.dataset)