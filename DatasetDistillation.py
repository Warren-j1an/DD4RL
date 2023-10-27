import collections

import torch
import utils
import pathlib
import numpy as np
import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import cv2


class DatasetDistillation:
    def __init__(self, batch, obs_spec, act_spec, discount, nstep, lr_obs, lr_action, device):
        self.batch = batch
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.nstep = nstep
        self.lr_obs = lr_obs
        self.lr_action = lr_action
        self.device = device

        discount_ = 1
        for _ in range(nstep):
            discount_ *= discount

        obs = np.random.rand(1000, *self.obs_spec.shape)
        action = np.random.rand(1000, *self.act_spec.shape)
        reward = np.linspace(0, 6, 1001)[:-1].reshape(-1, 1)  # action_repeat * nstep = 6
        discount = np.full(shape=(1000, 1), fill_value=discount_)
        next_obs = np.random.rand(1000, *self.obs_spec.shape)

        obs = self.convert(obs, self.obs_spec.minimum, self.obs_spec.maximum)
        action = self.convert(action, self.act_spec.minimum, self.act_spec.maximum)
        next_obs = self.convert(next_obs, self.obs_spec.minimum, self.obs_spec.maximum)

        self.obs = torch.tensor(obs, dtype=torch.float, requires_grad=True, device=self.device)
        self.action = torch.tensor(action, dtype=torch.float, requires_grad=True, device=self.device)
        self.reward = torch.tensor(reward, dtype=torch.float, requires_grad=False, device=self.device)
        self.discount = torch.tensor(discount, dtype=torch.float, requires_grad=False, device=self.device)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float, requires_grad=True, device=self.device)

        self.opt_obs = torch.optim.Adam([self.obs, self.next_obs], lr=self.lr_obs)
        self.opt_action = torch.optim.Adam([self.action], lr=self.lr_action)
        self.index = None

    def get_data(self, batch):
        reward_ = batch[2].numpy()
        self.index = reward_.reshape(-1)
        for i in range(len(self.index)):
            self.index[i] = np.floor(self.index[i] / 0.006) if self.index[i] != 6.0 else 999  # 6 for reward[5.994]
        obs_ = self.obs[self.index]
        action_ = self.action[self.index]
        reward_ = self.reward[self.index]
        discount_ = self.discount[self.index]
        next_obs_ = self.next_obs[self.index]

        return obs_, action_, reward_, discount_, next_obs_

    def train(self, critic_grad_sync, actor_grad_sync, critic_grad_real, actor_grad_real):
        for i in critic_grad_real:
            i.detach()
        for i in actor_grad_real:
            i.detach()
        metrics = {}
        critic_loss = utils.match_loss(critic_grad_sync, critic_grad_real, self.device, "mse")
        actor_loss = utils.match_loss(actor_grad_sync, actor_grad_real, self.device, "mse")
        loss = critic_loss + actor_loss
        metrics['DD_loss'] = loss.item()
        self.opt_obs.zero_grad(set_to_none=True)
        self.opt_action.zero_grad(set_to_none=True)
        # grad = [p.grad for p in [self.obs, self.action, self.reward, self.discount, self.next_obs]]
        loss.backward()
        # grad = [p.grad for p in [self.obs, self.action, self.reward, self.discount, self.next_obs]]
        self.opt_obs.step()
        self.opt_action.step()

        return metrics

    def save_img(self, global_step):
        img = np.clip(self.obs.cpu().detach().numpy(), 0, 255).astype(np.uint8)
        reward = self.reward.cpu().detach().numpy()
        img_n = np.clip(self.next_obs.cpu().detach().numpy(), 0, 255).astype(np.uint8)
        path = pathlib.Path.cwd() / f'sync_obs/{global_step}'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        ws = wb.active
        for i in range(1000):
            o = np.transpose(img[i], (1, 2, 0))[:, :, 6:]
            o_n = np.transpose(img_n[i], (1, 2, 0))[:, :, 6:]
            cv2.imwrite(os.path.abspath(path) + f'/img_{i}.png', o)
            cv2.imwrite(os.path.abspath(path) + f'/img_next_{i}.png', o_n)
            obs = Image(os.path.abspath(path) + f'/img_{i}.png')
            obs_next = Image(os.path.abspath(path) + f'/img_next_{i}.png')
            ws.add_image(obs, f'A{i + 1}')
            ws[f'B{i + 1}'] = reward[i][0]
            ws.add_image(obs_next, f'C{i + 1}')
        wb.save(os.path.abspath(path) + f'/sync_data.xlsx')
        for i in range(1000):
            obs = pathlib.Path(os.path.abspath(path) + f'/img_{i}.png')
            obs_next = pathlib.Path(os.path.abspath(path) + f'/img_next_{i}.png')
            obs.unlink()
            obs_next.unlink()

    def convert(self, data, min, max):
        data = data * (max - min) + min
        return data
