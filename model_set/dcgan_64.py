import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 16)
        # state size. (nf*16) x 4 x 4
        self.c6 = nn.Sequential(
            nn.Conv2d(nf * 16, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 16, 4, 1, 0),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*16) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 16 * 2, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 64 x 64
        self.upc6 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[4]], 1))
        d3 = self.upc3(torch.cat([d2, skip[3]], 1))
        d4 = self.upc4(torch.cat([d3, skip[2]], 1))
        d5 = self.upc5(torch.cat([d4, skip[1]], 1))
        output = self.upc6(torch.cat([d5, skip[0]], 1))
        return output

class MMFM_scene(nn.Module):
    def __init__(self, dim, outdim, nc=1):
        super(MMFM_scene, self).__init__()
        self.dim = dim
        # two layer MMFM model, based on paper combining these two.

        self.mmfm_l1 = torch.nn.Sequential(torch.nn.Linear(self.dim, self.dim),
                                           nn.BatchNorm1d(self.dim),
                                           nn.Tanh())
        self.mmfm_l2 = torch.nn.Sequential(torch.nn.Linear(self.dim, outdim),
                                           nn.BatchNorm1d(outdim),
                                           nn.Tanh())

    def forward(self, input):
        h1 = self.mmfm_l1(input)
        h2 = self.mmfm_l2(h1)
        return h2


class MMFM_tactile(nn.Module):
    def __init__(self, dim, outdim, nc=1):
        super(MMFM_tactile, self).__init__()
        self.dim = dim
        # two layer MMFM model, based on paper combining these two.

        self.mmfm_l1 = torch.nn.Sequential(torch.nn.Linear(self.dim, self.dim),
                                           nn.BatchNorm1d(self.dim),
                                           nn.Tanh())
        self.mmfm_l2 = torch.nn.Sequential(torch.nn.Linear(self.dim, outdim),
                                           nn.BatchNorm1d(outdim),
                                           nn.Tanh())

    def forward(self, input):
        h1 = self.mmfm_l1(input)
        h2 = self.mmfm_l2(h1)
        return h2


class raw_tactile_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(raw_tactile_encoder, self).__init__()
        self.tan_activation = nn.Tanh()
        self.fc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, tactile):
        out1 = self.tan_activation(self.fc_layer1(tactile))
        out2 = self.tan_activation(self.fc_layer2(out1))
        out3 = self.tan_activation(self.fc_layer3(out2))
        return out3