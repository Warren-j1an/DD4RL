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
    def __init__(self, batch, obs_spec, act_spec, discount, nstep, lr, device):
        self.batch = batch
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.nstep = nstep
        self.lr = lr
        self.device = device

        discount_ = 1
        for _ in range(nstep):
            discount_ *= discount

        # obs = np.random.rand(1000, *self.obs_spec.shape)
        # action = np.random.rand(1000, *self.act_spec.shape)
        # reward = np.linspace(0, 6, 1001)[:-1]  # action_repeat * nstep = 6
        # discount = np.full(shape=(1000, 1), fill_value=self.discount)
        # next_obs = np.random.rand(1000, *self.obs_spec.shape)
        #
        # obs = self.convert(obs, self.obs_spec.minimum, self.obs_spec.maximum)
        # action = self.convert(action, self.act_spec.minimum, self.act_spec.maximum)
        # next_obs = self.convert(next_obs, self.obs_spec.minimum, self.obs_spec.maximum)
        #
        # self.obs = torch.tensor(obs, dtype=torch.float, requires_grad=False, device=self.device)
        # self.action = torch.tensor(action, dtype=torch.float, requires_grad=False, device=self.device)
        # self.reward = torch.tensor(reward, dtype=torch.float, requires_grad=False, device=self.device)
        # self.discount = torch.tensor(discount, dtype=torch.float, requires_grad=False, device=self.device)
        # self.next_obs = torch.tensor(next_obs, dtype=torch.float, requires_grad=False, device=self.device)

        self.obs = []
        self.action = []
        self.reward = []
        self.discount = []
        self.next_obs = []
        for i in range(1000):
            obs = np.random.rand(*self.obs_spec.shape)
            action = np.random.rand(*self.act_spec.shape)
            reward = np.array([np.linspace(0, 6, 1001)[i]])  # action_repeat * nstep = 6
            discount = np.array([discount_])
            next_obs = np.random.rand(*self.obs_spec.shape)

            obs = self.convert(obs, self.obs_spec.minimum, self.obs_spec.maximum)
            action = self.convert(action, self.act_spec.minimum, self.act_spec.maximum)
            next_obs = self.convert(next_obs, self.obs_spec.minimum, self.obs_spec.maximum)

            self.obs.append(obs)
            self.action.append(action)
            self.reward.append(reward)
            self.discount.append(discount)
            self.next_obs.append(next_obs)

        self.obs = np.array(self.obs)
        self.action = np.array(self.action)
        self.reward = np.array(self.reward)
        self.discount = np.array(self.discount)
        self.next_obs = np.array(self.next_obs)

        self.obs_ = None
        self.action_ = None
        self.reward_ = None
        self.discount_ = None
        self.next_obs_ = None

        self.opt = None
        self.index = None

    def get_data(self, batch):
        reward_ = batch[2].numpy()
        reward_ = np.floor(reward_ / 0.006)
        for i in reward_:
            i = np.floor(i / 0.006) if i != 6.0 else 999  # 6 for reward[5.994]
        self.index, index = np.unique(reward_.astype(int), return_index=True)
        for i in range(len(batch)):
            batch[i] = batch[i][index]
        self.obs_ = torch.tensor(self.obs[self.index], dtype=torch.float, requires_grad=True, device=self.device)
        self.action_ = torch.tensor(self.action[self.index], dtype=torch.float, requires_grad=True, device=self.device)
        self.reward_ = torch.tensor(self.reward[self.index], dtype=torch.float, requires_grad=True, device=self.device)
        self.discount_ = torch.tensor(self.discount[self.index], dtype=torch.float, requires_grad=False, device=self.device)
        self.next_obs_ = torch.tensor(self.next_obs[self.index], dtype=torch.float, requires_grad=True, device=self.device)

        self.opt = torch.optim.Adam([self.obs_, self.action_, self.reward_, self.next_obs_], lr=self.lr)

        return self.obs_ * 255, self.action_, self.reward_, self.discount_, self.next_obs_ * 255

    def train(self, grad_real, grad_sync):
        for i in grad_real:
            i.detach()
        metrics = {}
        loss = utils.match_loss(grad_sync, grad_real, self.device, "mse")
        metrics['DD_loss'] = loss.item()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.obs[self.index] = self.obs_.detach().cpu().numpy()
        self.action[self.index] = self.action_.detach().cpu().numpy()
        self.reward[self.index] = self.reward_.detach().cpu().numpy()
        self.next_obs[self.index] = self.next_obs_.detach().cpu().numpy()

        return metrics

    def save_img(self):
        img = np.clip(255 * self.obs, 0, 255).astype(np.uint8)
        reward = self.reward
        img_n = np.clip(255 * self.next_obs, 0, 255).astype(np.uint8)
        path = pathlib.Path.cwd().parent.parent / 'sync_obs'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        ws = wb.active
        for i in range(self.batch):
            o = np.transpose(img[i], (1, 2, 0))[:, :, 6:]
            o_n = np.transpose(img_n[i], (1, 2, 0))[:, :, 6:]
            cv2.imwrite(os.path.abspath(path) + f'/img_{i}.png', o)
            cv2.imwrite(os.path.abspath(path) + f'/img_next_{i}.png', o_n)
            obs = Image(os.path.abspath(path) + f'/img_{i}.png')
            obs_next = Image(os.path.abspath(path) + f'/img_next_{i}.png')
            ws.add_image(obs, f'A_{i}')
            ws[f'B_{i}'] = reward[i]
            ws.add_image(obs_next, f'C_{i}')
        ws.save(os.path.abspath(path) + f'sync_data.xlsx')

    def save(self, path):
        path = str(path)
        np.save(path + "/obs.npy", self.obs)
        np.save(path + "/action.npy", self.action)
        np.save(path + "/reward.npy", self.reward)
        np.save(path + "/discount.npy", self.discount)
        np.save(path + "/next_obs.npy", self.next_obs)

    def load(self, path):
        self.obs = np.load(path + "/obs.npy")
        self.action = np.load(path + "/action.npy")
        self.reward = np.load(path + "/reward.npy")
        self.discount = np.load(path + "/discount.npy")
        self.next_obs = np.load(path + "/next_obs.npy")

    def convert(self, data, min, max):
        data = data * (max - min) + min
        return data
