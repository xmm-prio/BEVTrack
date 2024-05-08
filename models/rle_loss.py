import torch
import torch.nn as nn
from torch import distributions
import math


class RealNVP(nn.Module):

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(4, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 4), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(4, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 4))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()

        self.register_buffer('loc', torch.zeros(4))
        self.register_buffer('cov', torch.eye(4))
        self.register_buffer(
            'mask', torch.tensor(
                [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList(
            [self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


class RLELoss(nn.Module):

    def __init__(self,
                 residual=True,
                 q_distribution='laplace'):
        super(RLELoss, self).__init__()
        self.residual = residual
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()

    def forward(self, pred, sigma, target):
        """Forward function.

        Note:
            - batch_size: N
            - dimension of keypoints: D (D=4)

        Args:
            pred (Tensor[N, D]): Output regression.
            sigma (Tensor[N, D]): Output sigma.
            target (Tensor[N, D]): Target regression.
        """
        sigma = sigma.sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        error[3] = error[3] * 0.1
        # (B, 4)
        log_phi = self.flow_model.log_prob(error)
        log_phi = log_phi.reshape(target.shape[0], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], 4)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        loss /= len(loss)

        return loss.sum()
