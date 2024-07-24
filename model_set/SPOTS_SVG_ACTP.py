# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import model_set.utils as utility_prog


class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.features = features

        if self.features.optimizer == "adam" or self.features.optimizer == "Adam":
            self.optimizer = optim.Adam

        if self.features.criterion == "L1" or self.features.criterion == "mae" or self.features.criterion == "MAE":
            self.criterion = nn.L1Loss().to(self.features.device)
            self.criterion_scene = nn.L1Loss().to(self.features.device)
            self.criterion_tactile = nn.L1Loss().to(self.features.device)
        if self.features.criterion == "L2" or self.features.criterion == "mse" or self.features.criterion == "MSE":
            self.criterion = nn.MSELoss().to(self.features.device)
            self.criterion_scene = nn.MSELoss().to(self.features.device)
            self.criterion_tactile = nn.MSELoss().to(self.features.device)

    def load_model(self, full_model):
        self.frame_predictor_tactile = full_model["frame_predictor_tactile"].to(self.features["device"])
        self.frame_predictor_scene = full_model["frame_predictor_scene"].to(self.features["device"])
        self.posterior = full_model["posterior"].to(self.features["device"])
        self.prior = full_model["prior"].to(self.features["device"])
        self.encoder_scene = full_model["encoder_scene"].to(self.features["device"])
        self.decoder_scene = full_model["decoder_scene"].to(self.features["device"])
        self.MMFM_scene = full_model["MMFM_scene"].to(self.features["device"])
        self.MMFM_tactile = full_model["MMFM_tactile"].to(self.features["device"])

    def initialise_model(self):
        import model_set.dcgan_64 as model
        import model_set.actp as ACTP_model
        import model_set.lstm as lstm_models

        # SCENE:
        self.frame_predictor_scene = lstm_models.lstm(self.features.g_dim + self.features.tactile_size + self.features.z_dim + self.features.state_action_size, self.features.g_dim, self.features.rnn_size, self.features.predictor_rnn_layers, self.features.batch_size, self.features.device)
        self.frame_predictor_scene.apply(utility_prog.init_weights).to(self.features.device)

        self.MMFM_scene = model.MMFM_scene(self.features.g_dim + self.features.tactile_size, self.features.g_dim + self.features.tactile_size, self.features.channels)
        self.MMFM_scene.apply(utility_prog.init_weights).to(self.features.device)
        self.MMFM_scene_optimizer = self.optimizer(self.MMFM_scene.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

        self.encoder_scene = model.encoder(self.features.g_dim, self.features.channels)
        self.decoder_scene = model.decoder(self.features.g_dim, self.features.channels)
        self.encoder_scene.apply(utility_prog.init_weights).to(self.features.device)
        self.decoder_scene.apply(utility_prog.init_weights).to(self.features.device)

        self.frame_predictor_optimizer_scene = self.optimizer(self.frame_predictor_scene.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.encoder_optimizer_scene = self.optimizer(self.encoder_scene.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.decoder_optimizer_scene = self.optimizer(self.decoder_scene.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

        # TACTILE:
        self.frame_predictor_tactile = ACTP_model.ACTP(device=self.features.device, input_size=(self.features.g_dim + self.features.tactile_size), tactile_size=self.features.tactile_size)
        self.frame_predictor_tactile.apply(utility_prog.init_weights).to(self.features.device)

        self.MMFM_tactile = model.MMFM_tactile(self.features.g_dim + self.features.tactile_size, self.features.g_dim + self.features.tactile_size, self.features.channels)
        self.MMFM_tactile.apply(utility_prog.init_weights).to(self.features.device)
        self.MMFM_tactile_optimizer = self.optimizer(self.MMFM_tactile.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

        self.frame_predictor_optimizer_tactile = self.optimizer(self.frame_predictor_tactile.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

        # PRIOR:
        self.posterior = lstm_models.gaussian_lstm(self.features.g_dim + self.features.tactile_size, self.features.z_dim, self.features.rnn_size, self.features.posterior_rnn_layers, self.features.batch_size, self.features.device)
        self.prior = lstm_models.gaussian_lstm(self.features.g_dim + self.features.tactile_size, self.features.z_dim, self.features.rnn_size, self.features.prior_rnn_layers, self.features.batch_size, self.features.device)
        self.posterior.apply(utility_prog.init_weights).to(self.features.device)
        self.prior.apply(utility_prog.init_weights).to(self.features.device)
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

    def forward(self, image_context, targets, actions, tactiles, tactile_targets, test=False, rollout=False):
        if rollout: test = True
        mae_tactile = 0
        kld_tactile = 0
        mae_scene = 0
        kld_scene = 0
        outputs_scene = []
        outputs_tactile = []
        mae_scene_list, kld_scene_list, mae_tactile_list = [], [], []

        # permute the first two dimensions for each tensor
        if image_context is not None:       image_context = image_context.permute(1, 0, 2, 3, 4)
        if targets is not None:             targets = targets.permute(1, 0, 2, 3, 4)
        if actions is not None:             actions = actions.permute(1, 0, 2)
        if tactiles is not None:            tactiles = tactiles.permute(1, 0, 2)
        if tactile_targets is not None:     tactile_targets = tactile_targets.permute(1, 0, 2)

        # scene
        self.frame_predictor_scene.zero_grad()
        self.encoder_scene.zero_grad()
        self.decoder_scene.zero_grad()
        self.MMFM_scene.zero_grad()
        self.MMFM_tactile.zero_grad()
        self.frame_predictor_tactile.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()

        state = actions[0].to(self.features.device)
        scene = torch.cat((image_context, targets), 0)[:actions.shape[0]].to(self.features.device)
        scene_tactile = torch.cat((tactiles, tactile_targets), 0)[:actions.shape[0]].to(self.features.device)

        self.frame_predictor_tactile.init_hidden(scene.shape[1])

        for index in range(scene.shape[0] - 1):
            state_action = torch.cat((state, actions[index+1]), 1)

            if index > self.features.n_past - 1:  # horizon
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                h_target_scene_and_tactile = torch.cat([scene_tactile[index + 1], h_scene], 1)

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([x_pred_tactile, h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile= self.frame_predictor_tactile(MM_rep_tactile, state_action, x_pred_tactile)  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile_list.append(self.criterion_tactile(x_pred_tactile, scene_tactile[index + 1]))  # prediction model

                mae_scene_list.append(self.criterion(x_pred_scene, scene[index + 1]))  # prediction model
                kld_scene_list.append(self.kl_criterion_scene(mu, logvar, mu_p, logvar_p))

                mae_tactile += mae_tactile_list[-1]
                mae_scene += mae_scene_list[-1]
                kld_scene += kld_scene_list[-1]
            else:  # context
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                h_target_scene_and_tactile = torch.cat([scene_tactile[index + 1], h_scene], 1)

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([scene_tactile[index], h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile = self.frame_predictor_tactile(MM_rep_tactile, state_action, scene_tactile[index])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.criterion_tactile(x_pred_tactile, scene_tactile[index + 1])  # prediction model

                mae_scene += self.criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                mae_scene_list = [self.criterion_scene(x_pred_scene, scene[index + 1])]
                kld_scene_list = [self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)]
                mae_tactile_list = [self.criterion_tactile(x_pred_tactile, scene_tactile[index + 1])]

            outputs_scene.append(x_pred_scene)
            outputs_tactile.append(x_pred_tactile)

        if test is False:
            loss_scene = mae_scene + (kld_scene * self.features.beta)
            loss_tactile = mae_tactile + (kld_tactile * self.features.beta)
            combined_loss = loss_scene + loss_tactile
            combined_loss.backward()

            self.frame_predictor_optimizer_scene.step()
            self.encoder_optimizer_scene.step()
            self.decoder_optimizer_scene.step()

            self.frame_predictor_optimizer_tactile.step()

            self.MMFM_scene_optimizer.step()
            self.MMFM_tactile_optimizer.step()

            self.posterior_optimizer.step()
            self.prior_optimizer.step()

        outputs_scene = torch.stack(outputs_scene, dim=0)
        outputs_tactile = torch.stack(outputs_tactile, dim=0)

        mae_scene_loss = mae_scene.data.cpu().numpy() / (self.features.n_past + self.features.n_future)
        mae_scene_prior_loss = kld_scene.data.cpu().numpy() / (self.features.n_future + self.features.n_past)
        mae_tactile_loss = mae_tactile.data.cpu().numpy() / (self.features.n_past + self.features.n_future)

        if rollout:
            mae_scene_list = [x.data.cpu().numpy() for x in mae_scene_list]
            kld_scene_list = [x.data.cpu().numpy() for x in kld_scene_list]
            return outputs_scene, outputs_tactile, mae_scene_loss, mae_scene_prior_loss, mae_tactile_loss, mae_scene_list, kld_scene_list, mae_tactile_list
        else:  return outputs_scene, outputs_tactile, mae_scene_loss, mae_scene_prior_loss, mae_tactile_loss

    def kl_criterion_scene(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features.batch_size

    def kl_criterion_tactile(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features.batch_size

    def set_train(self):
        self.frame_predictor_tactile.train()
        self.frame_predictor_scene.train()
        self.encoder_scene.train()
        self.decoder_scene.train()

        self.MMFM_scene.train()
        self.MMFM_tactile.train()

        self.posterior.train()
        self.prior.train()

    def set_test(self):
        self.frame_predictor_tactile.eval()
        self.frame_predictor_scene.eval()
        self.encoder_scene.eval()
        self.decoder_scene.eval()

        self.MMFM_scene.eval()
        self.MMFM_tactile.eval()

        self.posterior.eval()
        self.prior.eval()

    def save_model(self, save_path):
        torch.save({"frame_predictor_tactile": self.frame_predictor_tactile,
                    "frame_predictor_scene": self.frame_predictor_scene, "encoder_scene": self.encoder_scene,
                    "decoder_scene": self.decoder_scene,
                    "posterior": self.posterior, "prior": self.prior, 'features': self.features,
                    "MMFM_scene": self.MMFM_scene, "MMFM_tactile": self.MMFM_tactile}, 
                    save_path)