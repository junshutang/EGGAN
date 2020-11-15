import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import parser
from collections import OrderedDict

config = parser.get_config()

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # assert self.weight is not None and \
        #        self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        # self.init_weights()

    def forward(self, x):
        return x + self.main(x)

class VGG(nn.Module):
    def __init__(self, pretrained=True, local_model_path="vgg/vgg19g-4aff041b.pth", nChannel=64):
        super(VGG, self).__init__()
        self.features_1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, nChannel, kernel_size=3, padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2_1', nn.Conv2d(nChannel, nChannel * 2, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(nChannel * 2, nChannel * 2, kernel_size=3, padding=1)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3_1', nn.Conv2d(nChannel * 2, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
        ]))
        self.features_2 = nn.Sequential(OrderedDict([
            ('conv3_2', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_5', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),
            ('conv4_1', nn.Conv2d(nChannel * 4, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
        ]))
        self.features_3 = nn.Sequential(OrderedDict([
            ('conv4_2', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('conv4_4', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),
            ('conv5_1', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
        ]))
        if pretrained:
            if local_model_path is None:
                print('loading default VGG')
                model_path = 'https://www.dropbox.com/s/4lbt58k10o84l5h/vgg19g-4aff041b.pth?dl=1'
                state_dict = torch.utils.model_zoo.load_url(model_path, 'vgg/')
            else:
                print('loading VGG from %s' % local_model_path)
                state_dict = torch.load(local_model_path)
            model_dict = self.state_dict()
            state_dict = {key: value for key, value in state_dict.items() if key in model_dict}
            self.load_state_dict(state_dict)

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3



class perceptural_loss(nn.Module):
    def __init__(self, model, mode):
        super(perceptural_loss, self).__init__()
        if mode =='train':
            self.vgg = VGG(pretrained=True).eval().cuda()
        else:
            self.vgg = VGG(pretrained=True).eval()
        # model.eval()
        # self.vgg = model
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        feature_x_1 = self.vgg.features_1(x)
        feature_y_1 = self.vgg.features_1(y)
        # feature_x_2 = self.vgg.features_2(feature_x_1)
        # feature_y_2 = self.vgg.features_2(feature_y_1)
        # feature_x = self.vgg(x)
        # feature_y = self.vgg(y)
        # loss = self.mse(feature_x_1, feature_y_1.detach()) + self.mse(feature_x_2, feature_y_2.detach())
        loss = self.mse(feature_x_1, feature_y_1.detach())
        return loss


class SpectralNorm(nn.Module):
    """
    spectral normalization
    code and idea originally from Takeru Miyato's work 'Spectral Normalization for GAN'
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

class Identity_Classifier(nn.Module):
    def __init__(self, input_dim=3, inter_dim=64):
        super(Identity_Classifier, self).__init__()

        self.identity_feat = nn.Sequential(
            nn.Conv2d(input_dim, inter_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_dim // 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim // 2, inter_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_dim // 2),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(inter_dim // 2, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_dim),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(inter_dim, inter_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim * 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 2, inter_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim * 4),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 4, inter_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim * 8),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 8, inter_dim * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim * 16),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 16, inter_dim * 32, kernel_size=3, stride=2, padding=1)
        )

        self.identity_fc = nn.Sequential(
            nn.Linear(inter_dim * 32, config.id_classes),
        )

    def forward(self, x):
        identity_feat = self.identity_feat(x)
        identity_feat_interm = F.avg_pool2d(identity_feat, identity_feat.size()[2:])
        identity_feat_interm = identity_feat_interm.view(identity_feat_interm.size(0), -1)
        identity_output = self.identity_fc(identity_feat_interm)
        # identity_output = F.softmax(identity_output)
        # print(x.shape, identity_feat.shape, identity_output.shape)
        return identity_feat, identity_output


class Ecoder(nn.Module):
    def __init__(self):
        super(Ecoder, self).__init__()

        self.conv_blocks = nn.Sequential(
            # [-1, 3, 64, 64] -> [-1, 128, 32, 32]
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),

            # [-1, 256, 16, 16]
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),

            # [-1, 512, 8, 8]
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.PReLU(),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, 2048, 4, 2, 1)
        )

        self.face_fc = nn.Sequential(
            nn.Linear(2048, config.latent_dim),
            nn.Sigmoid()
        )


    def forward(self, img, batch_size):

        img = img.reshape(batch_size, 3, config.img_size, config.img_size)
        out_conv = self.conv_blocks(img)
        out_conv_interm = torch.nn.functional.avg_pool2d(out_conv, out_conv.size()[2:])
        out_conv_interm = out_conv_interm.view(out_conv_interm.size(0), -1)
        #input_fc = out_conv.reshape(batch_size, -1)
        out_fc = self.face_fc(out_conv_interm)
        #print(out_fc.shape)

        return out_fc

class Generator(nn.Module):
    def __init__(self, use_spect=False):
        super(Generator, self).__init__()
        input_dim = config.latent_dim + config.au_dim
        # input_dim = 256 + 17 + 1
        # input_dim = 256 + 17

        self.conv_blocks = nn.Sequential(
            # [-1, z + cc + dc, 1, 1] -> [-1, 1024, 4, 4]
            spectral_norm(nn.ConvTranspose2d(input_dim, 2048, 4, 1, 0), use_spect),
            nn.InstanceNorm2d(2048),
            # AdaptiveInstanceNorm2d(2048),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(2048, 2048, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(2048),
            # AdaptiveInstanceNorm2d(2048),
            nn.PReLU(),

            spectral_norm(nn.ConvTranspose2d(2048, 1024, 4, 2, 1), use_spect),
            nn.InstanceNorm2d(1024),
            # AdaptiveInstanceNorm2d(1024),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(1024, 1024, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(1024),
            # AdaptiveInstanceNorm2d(1024),
            nn.PReLU(),

            spectral_norm(nn.ConvTranspose2d(1024, 512, 4, 2, 1), use_spect),
            nn.InstanceNorm2d(512),
            # AdaptiveInstanceNorm2d(512),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(512, 512, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(512),
            # AdaptiveInstanceNorm2d(512),
            nn.PReLU(),

            # [-1, 512, 8, 8]
            spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1), use_spect),
            nn.InstanceNorm2d(256),
            # AdaptiveInstanceNorm2d(256),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(256, 256, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(256),
            # AdaptiveInstanceNorm2d(256),
            nn.PReLU(),

            # [-1, 256, 16, 16]
            spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1), use_spect),
            nn.InstanceNorm2d(128),
            # AdaptiveInstanceNorm2d(128),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(128, 128, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(128),
            # AdaptiveInstanceNorm2d(128),
            nn.PReLU(),

            # [-1, 128, 32, 32]
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1), use_spect),
            nn.InstanceNorm2d(64),
            # AdaptiveInstanceNorm2d(64),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose2d(64, 64, 3, 1, 1), use_spect),
            nn.InstanceNorm2d(64),
            # AdaptiveInstanceNorm2d(64),
            nn.PReLU(),

            spectral_norm(nn.ConvTranspose2d(64, 3, 1, 1, 0), use_spect),
            nn.Tanh()
        )

    def forward(self, face_code, labels):

        gen_input = torch.cat((face_code, labels), 1)
        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        img = self.conv_blocks(gen_input)

        return img


class Discriminator_z(nn.Module):
    def __init__(self):
        super(Discriminator_z, self).__init__()

        self.dis_fc = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.PReLU(),

            nn.Linear(128, 64),
            nn.PReLU(),

            nn.Linear(64, 32),
            nn.PReLU(),

            nn.Linear(32, 16),
            nn.PReLU(),

            nn.Linear(16, 1),
            #nn.Sigmoid()
        )

    def forward(self, z):

        out = self.dis_fc(z)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            # [-1, 3, 64, 64] -> [-1, 128, 32, 32]
            nn.Conv2d(3, 128, 4, 2, 1),
            # nn.InstanceNorm2d(128),
            nn.PReLU(),

            # [-1, 256, 16, 16]
            nn.Conv2d(128, 256, 4, 2, 1),
            # nn.InstanceNorm2d(256),
            nn.PReLU(),

            # [-1, 512, 8, 8]
            nn.Conv2d(256, 512, 4, 2, 1),
            # nn.InstanceNorm2d(512),
            nn.PReLU(),

            # [-1, 1024, 4, 4]
            nn.Conv2d(512, 1024, 4, 2, 1),
            # nn.InstanceNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, 2048, 4, 2, 1),
            # nn.InstanceNorm2d(2048),
            nn.PReLU(),

            nn.Conv2d(2048, 1, 4, 1, 0)
        )


    def forward(self, img):

        out = self.conv_blocks(img).squeeze()

        return out, 0

class AU_Classifier(nn.Module):
    def __init__(self, au_dim = 17, input_dim=3, inter_dim=512):
        super(AU_Classifier, self).__init__()

        self.au_feat = nn.Sequential(
            nn.Conv2d(input_dim, inter_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(inter_dim * 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 2, inter_dim * 2, kernel_size=3, stride=1, padding=1),

        )
        self.au_fc = nn.Sequential(
            nn.Linear(inter_dim * 2, au_dim*(5+1)),
        )
        self.au_num = au_dim

    def forward(self, x):
        au_feat = self.au_feat(x)
        au_feat_interm = F.avg_pool2d(au_feat, au_feat.size()[2:])
        au_feat_interm = au_feat_interm.view(au_feat_interm.size(0), -1)
        au_output = self.au_fc(au_feat_interm)
        # au_output = nn.functional.sigmoid(au_output) * 5  #For AU MSE loss
        au_output = au_output.view(au_output.size(0), self.au_num, 5 + 1)
        return au_output

