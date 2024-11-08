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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.model = model  # Model to be trained
        self.optimizer = self.set_optimizer(method, self.model, lr)
        
        self.train_dl = train_dl  # Training data loader
        self.test_dl = test_dl  # Testing data loader
        self.batch_size = batch_size  # Size of each batch
        self.epsilon = epsilon  # Epsilon value for differential privacy
        self.max_per_sample_grad_norm = C  # Some constant or hyperparameter
        self.device = device  # Device to train on (e.g., 'cpu' or 'cuda')
        self.decay_rate_sens = decay_rate_sens  # Sensitivity decay rate
        self.decay_rate_mu = decay_rate_mu  # Mu decay rate
        self.dp = dp
        step = 0

        self.test_losses = []
        self.test_accuracies = []

        self.train_losses = []
        self.train_accuracies = []
        
        num_data = len(train_dl.dataset)
        print(f'Training_dataset length: {num_data}')

        self.sampling_rate = batch_size/num_data
        self.iteration = int(epochs/self.sampling_rate)
        
        if dp:
            if delta is None:
                delta = 1.0/num_data
            mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = C, tol = 1.e-12)
            mu_t = math.sqrt(math.log(mu**2/(self.sampling_rate**2*self.iteration)+1))
            sigma = 1/mu_t

            if decay_rate_mu is not None:
                self.decay_rate_mu = cal_step_decay_rate(decay_rate_mu,self.iteration)
                self.mu_0 = mu0_search(mu, self.iteration, self.decay_rate_mu, self.sampling_rate,mu_t=mu_t)
                
            if decay_rate_sens is not None:
                self.decay_rate_sens = cal_step_decay_rate(decay_rate_sens,self.iteration)

            self.privacy_engine = PrivacyEngine(
                    self.model,
                    sample_rate=self.sampling_rate,
                    batch_size=self.batch_size,
                    max_grad_norm=C,
                    noise_multiplier= sigma,
                )
            self.privacy_engine.attach(self.optimizer)

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4, verbose=True,
                                      min_lr=0.00001)

        for epoch in range(epochs):
            step = self.train(step, ema)
            scheduler.step(self.train_losses[epoch])
            if ema is not None:
                ema.update_model_with_ema()
            self.test()

    def set_optimizer(self, method, model, lr):
        if method == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        elif method == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif method == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        elif method == "adagrad":
            return torch.optim.Adagrad(model.parameters(), lr=lr)
        elif method == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise RuntimeError("Unknown Optimizer!")

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
            if self.decay_rate_sens is not None:
                clip = self.max_per_sample_grad_norm * (self.decay_rate_sens)**step
                self.privacy_engine.set_clip(clip)
            if self.decay_rate_mu is not None:
                unit_sigma = 1/(self.mu_0/(self.decay_rate_mu**(step)))
                self.privacy_engine.set_unit_sigma(unit_sigma)
        
            for i in tqdm(range(int(1/self.sampling_rate))):
                data, target = poisson_sampler(self.train_dl.dataset,self.sampling_rate)
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