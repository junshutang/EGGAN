import os
import parser
from solver import Solver

import torch

from dataloader import get_loader

#
# def cross_validation(train_dataset):
#     train_db, val_db = torch.utils.data.random_split(train_dataset, [1176, 216])
#     return train_db, val_db

def main(config):

    # Create directories if not exist.
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    # Data loader.

    train_loader = get_loader(config.data_dir, config.attr_dir, config.au_dim,
                                  config.img_size, config.batch_size, 'train')


    test_loader = get_loader(config.data_dir, config.attr_dir, config.au_dim,
                                 config.img_size, 1, 'test')


    # Solver for training and testing ETGAN
    solver = Solver(train_loader, test_loader, config)

    #solver.train()

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.load()


if __name__ == '__main__':

    config = parser.get_config()

    print(config)
    main(config)


