import math
import numpy as np
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from .GaussianCalibrator import calibrateAnalyticGaussianMechanism
import math
from .poisson_sampler import poisson_sampler
from .mu_search import mu0_search,cal_step_decay_rate
from scipy.stats import norm
from scipy import optimize

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
            dp = True):
        if method == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif method == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise RuntimeError("Unknown Optimizer!")
        
        self.model = model  # Model to be trained
        self.train_dl = train_dl  # Training data loader
        self.test_dl = test_dl  # Testing data loader
        self.batch_size = batch_size  # Size of each batch
        self.epsilon = epsilon  # Epsilon value for differential privacy
        self.max_per_sample_grad_norm = C  # Some constant or hyperparameter
        self.device = device  # Device to train on (e.g., 'cpu' or 'cuda')
        self.lr = lr  # Learning rate
        self.decay_rate_sens = decay_rate_sens  # Sensitivity decay rate
        self.decay_rate_mu = decay_rate_mu  # Mu decay rate
        self.dp = dp
        step = 0
        
        num_data = len(train_dl.dataset)
        print(f'Training_dataset length: {num_data}')

        self.sampling_rate = batch_size/num_data
        self.iteration = int(epochs/self.sampling_rate)
        
        if delta is None:
            delta = 1.0/num_data
        mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = 1, tol = 1.e-12)
        mu_t = math.sqrt(math.log(mu**2/(self.sampling_rate**2*self.iteration)+1))
        sigma = 1/mu_t

        if decay_rate_mu is not None:
            self.decay_rate_mu = cal_step_decay_rate(decay_rate_mu,self.iteration)
            self.mu_0 = mu0_search(mu, self.iteration, self.decay_rate_mu, self.sampling_rate,mu_t=mu_t)
            
        if decay_rate_sens is not None:
            self.decay_rate_sens = cal_step_decay_rate(decay_rate_sens,self.iteration)

        
        self.privacy_engine = PrivacyEngine(
                model,
                sample_rate=self.sampling_rate,
                max_grad_norm=C,
                noise_multiplier= sigma,
            )
        self.privacy_engine.attach(self.optimizer)

        for epochs in range(1, epochs + 1):
            step = self.train(step)
            self.test()

    def train(self, step):
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
                self.privacy_engine.max_grad_norm = clip
            if self.decay_rate_mu is not None:
                unit_sigma = 1/(self.mu_0/(self.decay_rate_mu**(step)))
                self.privacy_engine.noise_multiplier = unit_sigma
        
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
            acc = 100.0*correct/ total
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
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.test_dl.dataset),
                100.0 * correct / len(self.test_dl.dataset),
            )
        )
        return correct / len(self.test_dl.dataset)