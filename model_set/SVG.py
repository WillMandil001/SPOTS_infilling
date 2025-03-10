# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn
import torch.optim as optim
import model_set.utils as utility_prog

class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.features = features

        if self.features.optimizer == "adam" or self.features.optimizer == "Adam":
            self.optimizer = optim.Adam

        if self.features.criterion == "L1" or self.features.criterion == "MAE": 
            self.criterion = nn.L1Loss().to(self.features.device)
        if self.features.criterion == "L2" or self.features.criterion == "MSE":
            self.criterion = nn.MSELoss().to(self.features.device)

    def load_model(self, full_model):
        self.frame_predictor = full_model["frame_predictor"].to(self.features.device)
        self.posterior = full_model["posterior"].to(self.features.device)
        self.prior = full_model["prior"].to(self.features.device)
        self.encoder = full_model["encoder"].to(self.features.device)
        self.decoder = full_model["decoder"].to(self.features.device)

    def initialise_model(self):
        from model_set.lstm import lstm
        from model_set.lstm import gaussian_lstm
        self.frame_predictor = lstm(self.features.g_dim + self.features.z_dim + self.features.state_action_size, self.features.g_dim, self.features.rnn_size, self.features.predictor_rnn_layers, self.features.batch_size, self.features.device)
        self.posterior = gaussian_lstm(self.features.g_dim, self.features.z_dim, self.features.rnn_size, self.features.posterior_rnn_layers, self.features.batch_size, self.features.device)
        self.prior = gaussian_lstm(self.features.g_dim, self.features.z_dim, self.features.rnn_size, self.features.prior_rnn_layers, self.features.batch_size, self.features.device)
        self.frame_predictor.apply(utility_prog.init_weights).to(self.features.device)
        self.posterior.apply(utility_prog.init_weights).to(self.features.device)
        self.prior.apply(utility_prog.init_weights).to(self.features.device)

        import model_set.dcgan_64 as model
        self.encoder = model.encoder(self.features.g_dim, self.features.channels)
        self.decoder = model.decoder(self.features.g_dim, self.features.channels)
        self.encoder.apply(utility_prog.init_weights).to(self.features.device)
        self.decoder.apply(utility_prog.init_weights).to(self.features.device)

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.features.lr, betas=(self.features.beta1, 0.999))

    def save_model(self, save_path):
        torch.save({'encoder': self.encoder, 'decoder': self.decoder, 'frame_predictor': self.frame_predictor,
                    'posterior': self.posterior, 'prior': self.prior, 'features': self.features}, 
                    save_path)

    def train(self):
        self.frame_predictor.train()
        self.posterior.train()
        self.prior.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.frame_predictor.eval()
        self.posterior.eval()
        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()

    def test(self):
        self.eval()

    def forward(self, image_context, targets, actions, tactiles, tactile_targets, test=False, rollout=False):
        if rollout: test = True
        mae, kld = 0, 0
        mae_list, kld_list = [], []
        outputs = []

        # permute the first two dimensions for each tensor
        if image_context is not None:       image_context = image_context.permute(1, 0, 2, 3, 4)
        if targets is not None:             targets = targets.permute(1, 0, 2, 3, 4)
        if actions is not None:             actions = actions.permute(1, 0, 2)
        if tactiles is not None:            tactiles = tactiles.permute(1, 0, 2)
        if tactile_targets is not None:     tactile_targets = tactile_targets.permute(1, 0, 2)

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        state = actions[0].to(self.features.device)
        scene = torch.cat((image_context, targets), 0)[:actions.shape[0]].to(self.features.device)

        for index in range(scene.shape[0] - 1):
            state_action = torch.cat((state, actions[index+1]), 1)

            if index > self.features.n_past - 1:  # horizon
                h, skip = self.encoder(x_pred)
                h_target = self.encoder(scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae_list.append(self.criterion(x_pred, scene[index + 1]))  # prediction model
                mae += mae_list[-1]
                kld_list.append(self.kl_criterion(mu, logvar, mu_p, logvar_p))
                kld += kld_list[-1]   # learned prior

            else:  # context
                h, skip = self.encoder(scene[index])
                h_target = self.encoder(scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.criterion(x_pred, scene[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                # store the last prediction in the context as the first prediction of the prediction horizon (rollout) 
                mae_list = [self.criterion(x_pred, scene[index + 1])]
                kld_list = [self.kl_criterion(mu, logvar, mu_p, logvar_p)]

            outputs.append(x_pred)

        if test is False:
            loss = mae + (kld * self.features.beta)
            loss.backward()

            self.frame_predictor_optimizer.step()
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        if rollout:
            mae_scene_list = [x.data.cpu().numpy() for x in mae_list]
            kld_scene_list = [x.data.cpu().numpy() for x in kld_list]
        else:  mae_scene_list, kld_scene_list = None, None

        outputs_scene = torch.stack(outputs, dim=0)

        mae_scene_loss = mae.data.cpu().numpy() / (self.features.n_past + self.features.n_future)
        mae_scene_prior_loss = kld.data.cpu().numpy() / (self.features.n_future + self.features.n_past)

        if rollout:
            mae_scene_list = [x.data.cpu().numpy() for x in mae_list]
            kld_scene_list = [x.data.cpu().numpy() for x in kld_list]
            return outputs_scene, None, mae_scene_loss, mae_scene_prior_loss, None, mae_scene_list, kld_scene_list, None
        else:  return outputs_scene, None, mae_scene_loss, mae_scene_prior_loss, None

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features.batch_size