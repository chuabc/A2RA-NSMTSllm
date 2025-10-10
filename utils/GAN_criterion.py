import torch
import numpy as np
import torch.nn as nn

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def dis_criterion(real_validity, fake_validity):
    if isinstance(fake_validity, list):
        d_loss = 0
        for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
            real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_validity.get_device())
            fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_validity.get_device())
            d_real_loss = nn.MSELoss()(real_validity_item, real_label)
            d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
            d_loss += d_real_loss + d_fake_loss
    else:
        real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_validity.get_device())
        fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_validity.get_device())
        d_real_loss = nn.MSELoss()(real_validity, real_label)
        d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
        d_loss = d_real_loss + d_fake_loss
    return d_loss    
    
def gen_criterion(fake_validity):
    if isinstance(fake_validity, list):
        g_loss = 0
        for fake_validity_item in fake_validity:
            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=fake_validity.get_device())
            g_loss += nn.MSELoss()(fake_validity_item, real_label)
    else:
        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=fake_validity.get_device())
        g_loss = nn.MSELoss()(fake_validity, real_label)
    return g_loss


def regen_criterion(input, output):
    reconstrust_loss = 0
    mse_loss = nn.L1Loss()
    for i in range(len(input)):
        loss = mse_loss(input[i], output[i])
        reconstrust_loss = reconstrust_loss + loss
    return reconstrust_loss



class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr