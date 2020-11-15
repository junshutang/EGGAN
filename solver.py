import os
import numpy as np
import time
import datetime
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.models as models
from tensorboardX import SummaryWriter

from model import Generator
from model import Ecoder
from model import Discriminator
from model import Discriminator_z
from model import Identity_Classifier
from model import AU_Classifier
from model import perceptural_loss

import utils
import pytorch_msssim

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def msssim_loss(img1, img2):
    value = pytorch_msssim.msssim(img1, img2, normalize=True)
    return 1.0 - (torch.sum(value))

def make_model(cuda):
    model = models.vgg16(pretrained=True).features[:28]
    model = model.eval()
    if cuda:
        model.cuda()
    return model

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):

    classify_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, reduce=reduce)

    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_input = t_input.view(t_input.size(0), -1)
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

class Solver(object):

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
        self.writer = SummaryWriter(logdir=self.log_dir)

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Model configurations.
        self.version = config.version
        self.latent_dim = config.latent_dim
        self.au_dim = config.au_dim
        self.id_classes = config.id_classes
        self.img_size = config.img_size
        self.mode = config.mode
        self.paral = config.paral

        # Training configuration
        self.n_epochs = config.n_epochs
        self.decay_epoch = config.decay_epoch
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.b1 = config.b1
        self.b2 = config.b2

        self.lambda_au = config.lambda_au
        self.lambda_gp = config.lambda_gp
        self.lambda_rec = config.lambda_rec
        self.lambda_id = config.lambda_id
        self.lambda_pe = config.lambda_pe
        self.lambda_ms = config.lambda_ms

        # Director
        self.save_dir = config.save_dir
        self.data_dir = config.data_dir
        self.attr_dir = config.attr_dir
        self.log_dir = config.log_dir

        self.au_array = utils.get_au_array(self.attr_dir)
        self.adversarial_loss = torch.nn.MSELoss()
        self.id_class_loss = torch.nn.CrossEntropyLoss()
        self.id_fe_loss = torch.nn.L1Loss()
        self.au_loss = torch.nn.MSELoss()
        self.pixel_loss = torch.nn.L1Loss()

        self.FloatTensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.valid = self.FloatTensor(self.batch_size, 1).fill_(1.0)
        self.fake = self.FloatTensor(self.batch_size, 1).fill_(0.0)

        # Build the model
        self.build_model()
        self.build_tensorboard()
        self.vggmodel = make_model(True)

        # Test config\
        self.save_freq = config.save_freq
        self.test_exc_path = config.test_exc_path
        self.test_interp_path = config.test_interp_path

        self.test_epoch = config.test_epoch
        self.test_src_num = config.test_src_num
        self.test_tgt_num = config.test_tgt_num


        self.loss_visualization={}


        # Loss functions
        self.adversarial_loss = torch.nn.MSELoss()
        self.id_class_loss = torch.nn.CrossEntropyLoss()
        self.au_loss = torch.nn.MSELoss()
        self.pixel_loss = torch.nn.L1Loss()
        self.perceptural_loss = perceptural_loss(self.vggmodel, self.mode)

        # Optimizer
        self.optimizer_G = torch.optim.Adam([{'params': self.g_enc.parameters()},
                                        {'params': self.g_dec.parameters()}], lr=self.lr, betas=(self.b1, self.b2))
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                        lr_lambda=LambdaLR(self.n_epochs, 0, self.decay_epoch).step)

        self.optimizer_D_img = torch.optim.Adam([{'params': self.dis_img.parameters()}], lr=self.lr,
                                           betas=(self.b1, self.b2))
        self.scheduler_D_img = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_img,
                                                            lr_lambda=LambdaLR(self.n_epochs, 0, self.decay_epoch).step)

        self.optimizer_D_z = torch.optim.Adam([{'params': self.dis_z.parameters()}], lr=self.lr, betas=(self.b1, self.b2))
        self.scheduler_D_z = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_z,
                                                          lr_lambda=LambdaLR(self.n_epochs, 0, self.decay_epoch).step)

        self.optimizer_task = torch.optim.Adam(self.au_classifier.parameters(), lr=self.lr * 2, betas=(0.95, 0.999))
        self.scheduler_task = torch.optim.lr_scheduler.LambdaLR(self.optimizer_task,
                                                           lr_lambda=LambdaLR(self.n_epochs, 0, self.decay_epoch).step)

    def build_model(self):
        # Initialize generator and discriminator
        self.g_enc = Ecoder().cuda()
        self.g_dec = Generator().cuda()
        self.dis_img = Discriminator().cuda()
        self.dis_z = Discriminator_z().cuda()
        self.classifier = Identity_Classifier().cuda()
        self.au_classifier = AU_Classifier().cuda()

        self.g_enc.apply(weights_init_normal)
        self.g_dec.apply(weights_init_normal)
        self.dis_img.apply(weights_init_normal)
        self.dis_z.apply(weights_init_normal)
        self.au_classifier.apply(weights_init_normal)

        classify_model = torch.load("pretrain_checkpoints/end_weight.pth")
        self.classifier.load_state_dict(classify_model, strict=False)
        for param in self.classifier.parameters():
            param.requires_grad = False

    def train_discriminator_img(self):

        self.optimizer_D_img.zero_grad()
        self.optimizer_task.zero_grad()

        ######## Compute loss with real images
        real_pred, _ = self.dis_img(self.real_imgs)
        d_adv_real_loss = self.adversarial_loss(real_pred, self.valid)

        real_au = self.au_classifier(self.real_imgs)
        d_real_au_loss = au_softmax_loss(real_au, self.labels.long())

        ####### Compute loss for gradient penalty.
        z = self.FloatTensor(np.random.normal(0,1,(self.batch_size, self.latent_dim)))
        label_random = self.FloatTensor(
            utils.get_random_au(np.random.randint(0, self.au_dim, self.batch_size), self.au_array, num_columns=self.au_dim)
        )
        fake_imgs = self.g_dec(z, label_random)
        fake_pred, _ = self.dis_img(fake_imgs.detach())
        d_adv_gen_loss = self.adversarial_loss(fake_pred, self.fake)

        ####### Compute loss for gradient penalty.
        alpha = torch.rand(self.real_imgs.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * self.real_imgs.data + (1 - alpha)
                 * fake_imgs.data).requires_grad_(True)
        critic_output, _ = self.dis_img(x_hat)
        d_loss_gp = gradient_penalty(critic_output, x_hat)

        ####### Backward and optimize
        d_img_loss = d_adv_real_loss + d_adv_gen_loss + \
                     self.lambda_au * d_real_au_loss + self.lambda_gp * d_loss_gp

        d_img_loss.backward()

        self.optimizer_D_img.step()
        self.optimizer_task.step()

        ####### Logging
        self.loss_visualization['D_img/loss'] = d_img_loss.item()
        self.loss_visualization['D_img/loss_real'] = d_adv_real_loss.item()
        self.loss_visualization['D_img/loss_fake'] = d_adv_gen_loss.item()
        self.loss_visualization['D_img/loss_cls'] = self.lambda_au * d_real_au_loss.item()
        self.loss_visualization['D_img/loss_gp'] = self.lambda_gp * d_loss_gp.item()

        # 使用rec
        # real_enc = self.g_enc(self.real_imgs, self.batch_size)
        # rec_imgs = self.g_dec(real_enc, self.labels)
        # rec_pred, _ = self.dis_img(rec_imgs.detach())

    def train_discriminator_z(self):

        self.optimizer_D_z.zero_grad()

        real_enc = self.g_enc(self.real_imgs, self.batch_size)
        z = self.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)))

        z_pred = self.dis_z(z)
        face_pred = self.dis_z(real_enc.detach())
        z_loss = (self.adversarial_loss(z_pred, self.valid) + self.adversarial_loss(face_pred, self.fake)) * 0.5

        z_loss.backward()

        self.optimizer_D_z.step()

        self.loss_visualization['D_z/loss'] = z_loss.item()

    def train_generator(self):

        self.optimizer_G.zero_grad()
        # self.optimizer_task.zero_grad()

        z = self.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)))

        ######## Ori-Trg
        # Adv loss
        real_enc = self.g_enc(self.real_imgs, self.batch_size)
        label_random = self.FloatTensor(
            utils.get_random_au(np.random.randint(0, self.au_dim, self.batch_size), self.au_array, num_columns=self.au_dim)
        )
        fake_rand_imgs = self.g_dec(real_enc, label_random)
        fake_rand_pred, _ = self.dis_img(fake_rand_imgs)
        g_adv_fake_loss = self.adversarial_loss(fake_rand_pred, self.valid)

        # Label loss
        fake_rand_au = self.au_classifier(fake_rand_imgs)
        fake_rand_au_loss = au_softmax_loss(fake_rand_au, label_random.long())

        # Id loss
        _, fake_rand_ids = self.classifier(fake_rand_imgs)
        fake_rand_id_loss = self.id_class_loss(fake_rand_ids, self.id_labels)

        # Perceptual loss
        fake_rand_pe_loss = self.perceptural_loss(fake_rand_imgs, self.real_imgs)

        ######## Reconstruction


        # Adv loss
        real_enc = self.g_enc(self.real_imgs, self.batch_size)
        fake_rec_imgs = self.g_dec(real_enc, self.labels)
        fake_rec_pred, _ = self.dis_img(fake_rec_imgs)
        g_adv_rec_loss = self.adversarial_loss(fake_rec_pred, self.valid)

        # Label loss
        fake_rec_au = self.au_classifier(fake_rec_imgs)
        fake_rec_au_loss = au_softmax_loss(fake_rec_au, self.labels.long())

        # Pixel loss
        fake_rec_loss = self.pixel_loss(fake_rec_imgs, self.real_imgs)

        # Id loss
        _, fake_rec_ids = self.classifier(fake_rec_imgs)
        fake_rec_id_loss = self.id_class_loss(fake_rec_ids, self.id_labels)

        # Perceptual loss & Ms-ssim loss
        fake_rec_pe_loss = self.perceptural_loss(fake_rec_imgs, self.real_imgs)
        fake_rec_ssim_loss = msssim_loss(fake_rec_imgs, self.real_imgs)

        ####### Backward and optimize
        g_loss = g_adv_fake_loss + g_adv_rec_loss + self.lambda_rec * fake_rec_loss + \
                    self.lambda_au * fake_rand_au_loss + self.lambda_au * fake_rec_au_loss + \
                    self.lambda_id * fake_rand_id_loss + self.lambda_id * fake_rec_id_loss + \
                    self.lambda_pe * fake_rand_pe_loss + self.lambda_pe * fake_rec_pe_loss + \
                    self.lambda_ms * fake_rec_ssim_loss

        g_loss.backward()

        self.optimizer_G.step()
        # self.optimizer_task.step()

        ####### Logging
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/loss_adv_fake'] = g_adv_fake_loss.item()
        self.loss_visualization['G/loss_adv_rec'] = g_adv_rec_loss.item()
        self.loss_visualization['G/loss_rec'] = self.lambda_rec * fake_rec_loss.item()
        self.loss_visualization['G/loss_rand_au'] = self.lambda_au * fake_rand_au_loss.item()
        self.loss_visualization['G/loss_rec_au'] = self.lambda_au * fake_rec_au_loss.item()
        self.loss_visualization['G/loss_rand_id'] = self.lambda_id * fake_rand_id_loss.item()
        self.loss_visualization['G/loss_rec_id'] = self.lambda_id * fake_rec_id_loss.item()
        self.loss_visualization['G/loss_rand_pe'] = self.lambda_pe * fake_rand_pe_loss.item()
        self.loss_visualization['G/loss_rec_pe'] = self.lambda_pe * fake_rec_pe_loss.item()
        self.loss_visualization['G/loss_rec_ssim'] = self.lambda_ms * fake_rec_ssim_loss.item()

    def update_tensorboard(self, iteration):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Version [{}], Epoch [{}/{}], Batch [{}/{}]".format(
            et, self.version, self.epoch+1, self.n_epochs, iteration, len(self.train_loader))
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        for tag, value in self.loss_visualization.items():
            self.writer.add_scalar(
                tag, value, global_step=self.global_counter)

    def test_exc_generation(self, epoch, test_mode='train', src=0, tgt=1):

        os.makedirs(self.test_exc_path + '/' + self.version + '/', exist_ok=True)

        if test_mode == 'train':
            data_loader = self.train_loader
        elif test_mode == 'test':
            data_loader = self.test_loader

        for i, (imgs, labels, filename) in enumerate(data_loader):

            batch_size = imgs.shape[0]
            imgs = imgs
            labels = labels.float()
            id_labels = utils.get_id_label(filename)
            id_labels = torch.LongTensor(id_labels)
            # imgs = imgs.cuda()
            # labels = labels.float().cuda()
            # id_labels = utils.get_id_label(filename)
            # id_labels = torch.LongTensor(id_labels).cuda()

            if (i == src):
                test_real_src = imgs
                src_au = labels
                src_id = id_labels
                print(filename)
                print(labels)

            if (i == tgt):
                test_real_tgt = imgs
                tgt_au = labels
                tgt_id = id_labels
                print(filename)
                print(labels)
                break

        face_code_src = self.g_enc(test_real_src, batch_size)
        face_code_tgt = self.g_enc(test_real_tgt, batch_size)

        exc_img_tgt_src = self.g_dec(face_code_tgt, src_au)
        exc_img_src_tgt = self.g_dec(face_code_src, tgt_au)

        rec_img_tgt = self.g_dec(face_code_tgt, tgt_au)
        rec_img_src = self.g_dec(face_code_src, src_au)

        save_image(denorm(test_real_src.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'src.jpg')
        save_image(denorm(test_real_tgt.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'tgt.jpg')
        save_image(denorm(exc_img_tgt_src.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'tgt_src.jpg')
        save_image(denorm(exc_img_src_tgt.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'src_tgt.jpg')
        save_image(denorm(rec_img_src.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'src_rec.jpg')
        save_image(denorm(rec_img_tgt.data),
                   self.test_exc_path + '/' + self.version + '/' + str(epoch) + '_' + 'tgt_rec.jpg')


    def test_exc(self, idx, src_name, tgt_name, au_dict):

        os.makedirs(self.test_exc_path + '/' + self.version + '/exc/', exist_ok=True)
        exc_path = self.test_exc_path + '/' + self.version + '/exc/'
        src_img = Image.open(os.path.join(self.data_dir, src_name))
        tgt_img = Image.open(os.path.join(self.data_dir, tgt_name))

        transform = utils.get_transform()
        src_img = transform(src_img)
        tgt_img = transform(tgt_img)

        # print(au_dict[src_name])
        src_label = self.FloatTensor(np.array(au_dict[src_name]).reshape(1, self.au_dim))
        tgt_label = self.FloatTensor(np.array(au_dict[tgt_name]).reshape(1, self.au_dim))

        face_code_src = self.g_enc(src_img, 1)
        face_code_tgt = self.g_enc(tgt_img, 1)

        exc_img_tgt_src = self.g_dec(face_code_tgt, src_label)
        exc_img_src_tgt = self.g_dec(face_code_src, tgt_label)

        rec_img_tgt = self.g_dec(face_code_tgt, tgt_label)
        rec_img_src = self.g_dec(face_code_src, src_label)

        save_image(denorm(src_img.data),
                   exc_path + str(idx) + '_' + 'src.jpg')
        save_image(denorm(tgt_img.data),
                   exc_path + str(idx) + '_' + 'tgt.jpg')
        save_image(denorm(exc_img_tgt_src.data),
                   exc_path + str(idx) + '_' + 'tgt_src.jpg')
        save_image(denorm(exc_img_src_tgt.data),
                   exc_path + str(idx) + '_' + 'src_tgt.jpg')
        save_image(denorm(rec_img_src.data),
                   exc_path + str(idx) + '_' + 'src_rec.jpg')
        save_image(denorm(rec_img_tgt.data),
                   exc_path + str(idx) + '_' + 'tgt_rec.jpg')


    def test_au_interp(self, src_img_name):

        src_img = Image.open(os.path.join(self.data_dir, src_img_name))

        transform = utils.get_transform()
        src_img = transform(src_img)
        print(src_img.shape)


        os.makedirs(self.test_interp_path + '/' + self.version + '/', exist_ok=True)

        au_interp = self.FloatTensor(np.linspace(0,5,6))

        base_cond = self.FloatTensor(np.zeros(self.au_dim))

        for d in range(self.au_dim):
            for i in range(len(au_interp)):
                test_cond = base_cond
                test_cond[d] = au_interp[i]
                test_cond = test_cond.reshape(1, self.au_dim)

                # print(test_cond.shape, self.g_enc(src_img, 1).shape)

                img_tgt = self.g_dec(self.g_enc(src_img, 1), test_cond)
                save_image(denorm(img_tgt.data),
                           self.test_interp_path + '/' + self.version + '/' + str(d) + '_' + str(i) +
                                                   '_' + src_img_name + '.jpg')

    def test_random_au(self, src_img_name):

        src_img = Image.open(os.path.join(self.data_dir, src_img_name))

        transform = utils.get_transform()
        src_img = transform(src_img)

        os.makedirs('test_random_2/' + self.version + '/', exist_ok=True)
        f = open('random.txt','a')

        for i in range(1000):
            test_random = self.FloatTensor(
                utils.get_random_au(np.random.randint(0, self.au_dim, 1), self.au_array,
                                    num_columns=self.au_dim)
            )
            f.write(str(i) + ' ' + str(test_random.data))
            f.write('\n')

            img_tgt = self.g_dec(self.g_enc(src_img, 1), test_random)
            save_image(denorm(img_tgt.data),
                       'test_random_2/' + self.version + '/' + str(i) + '.jpg')

        f.close()

    def test_rec(self):

        print("Start Loading from Version {} Epoch {}".format(self.version, self.test_epoch))

        self.load_model(self.test_epoch)

        os.makedirs('test_rec/' + self.version + '/', exist_ok=True)
        data_loader = self.test_loader

        for i, (imgs, labels, filename) in enumerate(data_loader):

            if i<100:
                batch_size = imgs.shape[0]
                src_imgs = imgs
                src_labels = labels.float()
                id_labels = utils.get_id_label(filename)
                id_labels = torch.LongTensor(id_labels)

                face_code_src = self.g_enc(src_imgs, 1)
                rec_src_img = self.g_dec(face_code_src, src_labels)

                save_image(denorm(src_imgs.data),
                           'test_rec/' + self.version + '/' + str(i) + '_' + 'src.jpg')
                save_image(denorm(rec_src_img.data),
                           'test_rec/' + self.version + '/' + str(i) + '_' + 'src_rec.jpg')

            else:
                break


    def save_model(self, epoch):

        os.makedirs(self.save_dir + '/' + self.version, exist_ok=True)
        save_path = self.save_dir + '/' + self.version + '/'
        torch.save(self.dis_img.state_dict(),
                   os.path.join(
                       save_path + str(epoch) + "_dis_img.pth"))
        torch.save(self.g_dec.state_dict(),
                   os.path.join(
                       save_path + str(epoch) + "_g_dec.pth"))
        torch.save(self.au_classifier.state_dict(),
                   os.path.join(
                       save_path + str(epoch) + "_au_classifier.pth"))
        torch.save(self.dis_z.state_dict(),
                   os.path.join(
                       save_path + str(epoch) + "_dis_z.pth"))
        torch.save(self.g_enc.state_dict(),
                   os.path.join(
                       save_path + str(epoch) + "_g_enc.pth"))

    def load_model(self, load_epoch):

        # self.g_enc = Ecoder().cuda()
        # self.g_dec = Generator().cuda()
        # self.dis_img = Discriminator().cuda()
        # self.dis_z = Discriminator_z().cuda()
        # self.classifier = Identity_Classifier().cuda()
        # self.au_classifier = AU_Classifier().cuda()

        self.g_enc = Ecoder()
        self.g_dec = Generator()
        self.dis_img = Discriminator()
        self.dis_z = Discriminator_z()
        self.classifier = Identity_Classifier()
        self.au_classifier = AU_Classifier()

        load_path = self.save_dir + '/' + self.version + '/' + str(load_epoch)
        # g_enc_model = torch.load(load_path + "_g_enc.pth")
        # g_dec_model = torch.load(load_path + "_g_dec.pth")
        # dis_img_model = torch.load(load_path + "_dis_img.pth")
        # dis_z_model = torch.load(load_path + "_dis_z.pth")
        # au_classifier_model = torch.load(load_path + "_au_classifier.pth")
        g_enc_model = utils.load_on_cpu(load_path + "_g_enc.pth")
        g_dec_model = utils.load_on_cpu(load_path + "_g_dec.pth")
        dis_img_model = utils.load_on_cpu(load_path + "_dis_img.pth")
        dis_z_model = utils.load_on_cpu(load_path + "_dis_z.pth")
        au_classifier_model = utils.load_on_cpu(load_path + "_au_classifier.pth")

        self.g_enc.load_state_dict(g_enc_model, strict=False)
        self.g_dec.load_state_dict(g_dec_model, strict=False)
        self.dis_img.load_state_dict(dis_img_model,strict=False)
        self.dis_z.load_state_dict(dis_z_model,strict=False)
        self.au_classifier.load_state_dict(au_classifier_model,strict=False)

        for param in self.classifier.parameters():
            param.requires_grad = False

    def train(self):

        print("Start Training......")

        self.global_counter = 0

        self.start_time = time.time()

        for epoch in range(self.n_epochs):

            self.epoch = epoch

            self.scheduler_D_img.step()
            self.scheduler_D_z.step()
            self.scheduler_G.step()
            self.scheduler_task.step()

            for i, (imgs, labels, filename) in enumerate(self.train_loader):
                self.real_imgs = imgs.cuda()
                self.labels = labels.float().cuda()

                id_labels = utils.get_id_label(filename)
                self.id_labels = torch.LongTensor(id_labels).cuda()

                self.train_discriminator_img()
                self.train_discriminator_z()
                self.train_generator()

                self.global_counter += 1

                self.update_tensorboard(i)

            self.test_exc_generation(self.epoch, 'train', self.test_src_num, self.test_tgt_num)

            if epoch % self.save_freq == 0 and epoch != 0:
                self.save_model(self.epoch)


    def load(self):

        print("Start Loading from Version {} Epoch {}".format(self.version, self.test_epoch))

        self.load_model(self.test_epoch)

        # self.test_rec()

        au_dict = utils.get_au_dict(self.attr_dir)
        #

        src_img_name_list = ['SN023_3421.jpg']

        tgt_img_name_list = ['SN032_2963.jpg']


        for i in range(len(src_img_name_list)):

            src_img_name = src_img_name_list[i]
            tgt_img_name = tgt_img_name_list[i]

            self.test_exc(i, src_img_name, tgt_img_name, au_dict)

        self.test_exc_generation(self.test_epoch, 'test', self.test_src_num, self.test_tgt_num)

        src_img_name = 'SN011_40.jpg'
        # self.test_random_au(src_img_name)
        # self.test_au_interp(src_img_name)





