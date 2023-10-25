import torch
import utils
import pathlib
import numpy as np
import os
import cv2


class DatasetDistillation:
    def __init__(self, batch, obs_spec, act_spec, lr, device):
        self.batch = batch
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.lr = lr
        self.device = device

        # synthetic batch, initial random
        obs = np.random.rand(self.batch + 1, *self.obs_spec.shape)
        action = np.random.rand(self.batch, *self.act_spec.shape)
        reward = np.random.rand(self.batch, 1)
        discount = np.random.rand(self.batch, 1)
        # next_obs = np.random.rand(self.batch, *self.obs_spec.shape)

        # TODO: initial fix\orgin

        obs = self.convert(obs, self.obs_spec.minimum, self.obs_spec.maximum)
        action = self.convert(action, self.act_spec.minimum, self.act_spec.maximum)
        # next_obs = self.convert(next_obs, self.obs_spec.minimum, self.obs_spec.maximum)

        self.obs = torch.tensor(obs, dtype=torch.float, requires_grad=True, device=self.device)
        self.action = torch.tensor(action, dtype=torch.float, requires_grad=True, device=self.device)
        self.reward = torch.tensor(reward, dtype=torch.float, requires_grad=True, device=self.device)
        self.discount = torch.tensor(discount, dtype=torch.float, requires_grad=True, device=self.device)
        # self.next_obs = torch.tensor(next_obs, dtype=torch.float, requires_grad=True, device=self.device)

        self.opt = torch.optim.Adam([self.obs, self.action, self.reward, self.discount], lr=lr)

    def get_data(self):
        return self.obs[:-1] * 255, self.action, self.reward, self.discount, self.next_obs

    def save_img(self):
        img = np.clip(255 * self.obs.cpu().detach().numpy(), 0, 255).astype(np.uint8)
        img_n = np.clip(255 * self.next_obs.cpu().detach().numpy(), 0, 255).astype(np.uint8)
        path = pathlib.Path.cwd().parent.parent / 'sync_obs'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        for i in range(img.shape[0]):
            o = np.transpose(img[i], (1, 2, 0))[:, :, 6:]
            o_n = np.transpose(img_n[i], (1, 2, 0))[:, :, 6:]
            cv2.imwrite(os.path.abspath(path) + f'/img_{i}.png', o)
            cv2.imwrite(os.path.abspath(path) + f'/img_next_{i}.png', o_n)

    def train(self, grad_real, grad_sync):
        for i in grad_real:
            i.detach()
        metrics = {}
        loss = utils.match_loss(grad_sync, grad_real, self.device, "ours")
        metrics['DD_loss'] = loss.item()
        self.opt.zero_grad()
        # grad = [p.grad for p in [self.obs, self.action, self.reward, self.discount, self.next_obs]]
        loss.backward()
        # grad = [p.grad for p in [self.obs, self.action, self.reward, self.discount, self.next_obs]]
        self.opt.step()
        return metrics

    def convert(self, data, min, max):
        data = data * (max - min) + min
        return data
