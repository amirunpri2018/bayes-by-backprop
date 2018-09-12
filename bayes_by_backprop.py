import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms


class BayesianParameter(nn.Module):
    def __init__(self, shape, sigma_prior=1.0, ratio_prior=None, initial_rho=-3.0):
        super(BayesianParameter, self).__init__()
        self.sigma_prior = sigma_prior
        self.ratio_prior = ratio_prior
        self.mu = nn.Parameter(torch.Tensor(*shape).normal_(0, 0.1))
        self.rho = nn.Parameter(torch.zeros(*shape) + initial_rho)

    @staticmethod
    def _log_gaussian(x, mu, sigma):
        if isinstance(sigma, torch.Tensor):
            return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
        else:
            return float(-0.5 * np.log(2 * np.pi) - np.log(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

    @staticmethod
    def _gaussian(x, mu, sigma):
        return float(1.0 / np.sqrt(2.0 * np.pi * (sigma ** 2))) * torch.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

    @staticmethod
    def _log_mixture_gaussians(x, mus, sigmas, ratios):
        ratios = np.array(ratios)
        assert np.allclose(ratios.sum(), 1.0)
        prob = torch.zeros_like(x)
        for mu, sigma, ratio in zip(mus, sigmas, ratios):
            prob += float(ratio) * BayesianParameter._gaussian(x, mu, sigma)
        return torch.log(prob)

    def sample(self):
        epsilon = Variable(torch.Tensor(self.mu.data.size()).normal_(0, 1).cuda())
        sigma = F.softplus(self.rho)
        weight = self.mu + sigma * epsilon

        if isinstance(self.sigma_prior, float):
            log_prior_likelihood = BayesianParameter._log_gaussian(weight, 0, self.sigma_prior).sum()
        else:
            log_prior_likelihood = BayesianParameter._log_mixture_gaussians(weight, [0, 0], self.sigma_prior, self.ratio_prior).sum()

        log_posterior_likelihood = BayesianParameter._log_gaussian(weight, self.mu, sigma).sum()
        return weight, log_prior_likelihood, log_posterior_likelihood


class DenseLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior, ratio_prior=None, initial_rho=-3.0):
        super(DenseLayer, self).__init__()
        self.weight = BayesianParameter((n_input, n_output), sigma_prior, ratio_prior, initial_rho)
        self.bias = BayesianParameter((n_output,), sigma_prior, ratio_prior, initial_rho)
        self.log_prior_ll = 0
        self.log_posterior_ll = 0

    def forward(self, input, inference=False):
        if inference:
            return torch.mm(input, self.weight.mu) + self.bias.mu

        W, log_prior_W, log_posterior_W = self.weight.sample()
        b, log_prior_b, log_posterior_b = self.bias.sample()
        output = torch.mm(input, W) + b

        self.log_prior_ll = log_prior_W + log_prior_b
        self.log_posterior_ll = log_posterior_W + log_posterior_b
        return output


class BayesNet(nn.Module):
    def __init__(self, sigma_prior, ratio_prior, initial_rho):
        super(BayesNet, self).__init__()
        self.fc1 = DenseLayer(784, 400, sigma_prior, ratio_prior, initial_rho)
        self.fc2 = DenseLayer(400, 400, sigma_prior, ratio_prior, initial_rho)
        self.fc3 = DenseLayer(400, 10, sigma_prior, ratio_prior, initial_rho)

    def forward(self, x, inference=False):
        x = F.relu(self.fc1(x.view(x.size()[0], -1), inference))
        x = F.relu(self.fc2(x, inference))
        x = self.fc3(x, inference)
        return F.log_softmax(x, dim=1)

    def log_prior_likelihood(self):
        return self.fc1.log_prior_ll + self.fc2.log_prior_ll + self.fc3.log_prior_ll

    def log_posterior_likelihood(self):
        return self.fc1.log_posterior_ll + self.fc2.log_posterior_ll + self.fc3.log_posterior_ll

    def bayesian_loss(self):
        return self.log_posterior_likelihood() - self.log_prior_likelihood()


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Bayes by Backprop')
    parser.add_argument('--data-path', required=True, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default 0.01)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default 42)')
    parser.add_argument('--sigma-prior', nargs='+', type=float, default=1.0, metavar='SIGMA', help='sigma prior for weights (default 1.0)')
    parser.add_argument('--ratio-prior', nargs='+', type=float, default=1.0, metavar='PI', help='sigma prior for weights (default 1.0)')
    parser.add_argument('--initial-rho', type=float, default=-3.0, metavar='SP', help='initial value for rhos (default -3.0)')
    args = parser.parse_args()
    print(args)
    return args


def train(model, optimizer, train_loader, test_loader, device, num_epochs, loss_decay_rate=0.5):
    model.train()
    for epoch in range(1, num_epochs + 1):
        batch_weight = 1.0
        total_batch_weight = (loss_decay_rate ** len(train_loader) - 1) / (loss_decay_rate - 1)
        avg_train_loss, avg_nll_loss, avg_bayesian_loss = 0, 0, 0

        for data, target in train_loader:
            batch_weight *= loss_decay_rate
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            nll_loss = F.nll_loss(output, target, size_average=False)
            norm_batch_weight = batch_weight / total_batch_weight
            bayesian_loss = norm_batch_weight * model.bayesian_loss() / len(train_loader)
            train_loss = nll_loss + bayesian_loss

            avg_train_loss += train_loss.item()
            avg_nll_loss += nll_loss.item()
            avg_bayesian_loss += bayesian_loss.item()

            train_loss.backward()
            optimizer.step()

        avg_train_loss /= len(train_loader.dataset)
        avg_bayesian_loss /= len(train_loader.dataset)
        avg_nll_loss /= len(train_loader.dataset)

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print('Epoch {}: train_loss = {:.6f} (bayesian_loss = {:6f}, nll_loss = {:6f}), train_acc = {:.2f}, test_acc = {:.2f}'.format(
            epoch, avg_train_loss, avg_bayesian_loss, avg_nll_loss, train_acc, test_acc))


def evaluate(model, data_loader, device):
    model.eval()
    val_correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data, inference=True)
        pred = output.max(1, keepdim=True)[1]
        val_correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * val_correct / len(data_loader.dataset)
    return accuracy


def main():
    args = get_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, download=True, train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x / 126.0)
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 126.0)
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = BayesNet(args.sigma_prior, args.ratio_prior, args.initial_rho).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_loader, test_loader, device, args.epochs)


if __name__ == '__main__':
    main()
