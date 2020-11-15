from dataloader import get_loader
import argparse
import utils
from model import Identity_Classifier
import torch
import torch.nn as nn
import os

parser = argparse.ArgumentParser()


parser.add_argument('--pre_epochs', type=int, default=16, help='number of epochs of pretraining')
parser.add_argument('--decay_epoch', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=50, help='dimensionality of the latent space')
parser.add_argument('--code_dim', type=int, default=81, help='latent code')
parser.add_argument('--au_dim', type=int, default=17, help='number of aus')
parser.add_argument('--id_classes', type=int, default=27, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--face_z_dim', type=int, default=50, help='identity representation')

parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--attr_dir', type=str, default=None)


opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


classifier = Identity_Classifier(opt.id_classes)
classifier.apply(weights_init_normal)
classifier.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler_class = torch.optim.lr_scheduler.LambdaLR(optimizer_class, lr_lambda=LambdaLR(opt.pre_epochs, 0, opt.decay_epoch).step)

train_loader = get_loader(opt.data_dir, opt.attr_dir, opt.au_dim, opt.img_size, opt.batch_size)

for epoch in range(opt.pre_epochs):
    scheduler_class.step()
    for i, (image, label, filename) in enumerate(train_loader):

        imgs = image.cuda()
        label = label.cuda()
        id_label = utils.get_id_label(filename)
        id_label = torch.LongTensor(id_label).cuda()

        optimizer_class.zero_grad()
        _, id_out = classifier(imgs)

        class_loss = criterion(id_out, id_label)
        class_loss.backward()
        optimizer_class.step()

        print("[Epoch %d/%d] [Batch %d/%d] [class loss: %f]" % (
            epoch, opt.pre_epochs, i, len(train_loader), class_loss.item()))
        batches_done = epoch * len(train_loader) + i

    if epoch % 4 == 0:
        torch.save(classifier.state_dict(),
                   os.path.join("pretrain_checkpoints/",
                                "epoch_" + str(epoch + 1) + "_weight.pth"))

torch.save(classifier.state_dict(),
           os.path.join("pretrain_checkpoints/", "end_weight.pth"))
