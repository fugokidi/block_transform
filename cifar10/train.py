import os
import time
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from apex import amp

from utils import *
from models import *
from keydefense import *

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Uncomment normalization in keydefense')
    parser.add_argument("--work-path", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    return parser.parse_args()


config = parse_config_file(parse_args())


def train(train_loader, net, criterion, optimizer, defense=None, scheduler=None):
    net.train()
    train_loss = 0
    train_acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        if defense:
            output = net(defense(data))
        else:
            output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.size(0)
        train_acc += (output.max(1)[1] == target).sum().item()
        if scheduler is not None:
            scheduler.step()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc


def test(test_loader, net, defense=None):
    global best_acc

    net.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            if defense:
                output = net(defense(data))
            else:
                output = net(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            test_acc += (output.max(1)[1] == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    logger.info("== Test loss: {:.4f}, Test acc: {:.4f}".format(test_loss, test_acc))

    is_best = test_acc > best_acc

    save_model(net.state_dict(), is_best, config.work_path + "/" + config.ckpt_name)

    if is_best:
        best_acc = test_acc

    return test_loss, test_acc


def main():
    global best_acc

    logfile = os.path.join(config.work_path, "log.txt")
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logger.info(config)

    # fix all seeds for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader, test_loader = load_cifar10(config, worker_init_fn=_init_fn)

    net = resnet18().cuda()

    if config.defense == 'nodefense':
        defense = None
    else:
        defense = globals()[config.defense](config).cuda()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.lr_max,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    amp_args = dict(
        opt_level=config.opt_level, loss_scale=config.loss_scale, verbosity=False
    )

    if config.opt_level == "O2":
        amp_args["master_weights"] = config.master_weights

    net, optimizer = amp.initialize(net, optimizer, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = config.epochs * len(train_loader)

    if config.lr_schedule == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.lr_min,
            max_lr=config.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif config.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[61, 121, 161], gamma=0.2
        )

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")
    best_acc = 0.0

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        if config.lr_schedule == "cyclic":
            train_loss, train_acc = train(
                train_loader, net, criterion, optimizer, defense, scheduler
            )
        elif config.lr_schedule == "multistep":
            train_loss, train_acc = train(
                train_loader, net, criterion, optimizer, defense
            )
            scheduler.step()
        end = time.time()
        lr = scheduler.get_lr()[0]
        logger.info(
            "%d \t %.1f \t \t %.4f \t %.4f \t %.4f",
            epoch,
            end - start,
            lr,
            train_loss,
            train_acc,
        )

        if epoch == 1 or epoch % config.eval_freq == 0 or epoch == config.epochs:
            test(test_loader, net, defense)
    end_time = time.time()
    logger.info("== Training Finished. best_test_acc: {:.4f} ==".format(best_acc))
    logger.info(
        "== Total training time: {:.4f} minutes ==".format((end_time - start_time) / 60)
    )


if __name__ == "__main__":
    main()
