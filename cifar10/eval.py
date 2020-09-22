import os
import time
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from utils import *
from models import *
from keydefense import *
from attacks import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-path", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--attack", default='no', type=str)
    return parser.parse_args()


config = parse_config_file(parse_args())


def test(test_loader, net):
    net.float()
    net.eval()
    test_loss = 0
    test_acc = 0
    iterator = tqdm(test_loader)

    with torch.no_grad():
        for data, target in iterator:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item() * data.size(0)
            test_acc += (output.max(1)[1] == target).sum().item()
            iterator.set_description("Test Loss: {:.6f}".format(loss.item()))

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc


def main():
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

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(config.mean, config.std),
        ]
    )
    num_workers = config.workers

    test_dataset = datasets.CIFAR10(
        config.data_path, train=False, transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_init_fn,
    )

    norm_layer = Normalize(mean=config.mean, std=config.std).cuda()

    base_net = resnet18().cuda()
    load_model(config.work_path + "/" + config.ckpt_name + "_best.pth.tar", base_net)
    base_net.float()
    base_net.eval()

    # attacker_net = resnet18().cuda()
    # load_model("resnet18_shuffle_4_0/resnet_18_shuffle_4_0_best.pth.tar", attacker_net)
    # load_model("resnet18_np_4_0/resnet_18_np_4_0_best.pth.tar", attacker_net)
    # load_model("resnet18_ffx_4_0/resnet_18_ffx_4_0_best.pth.tar", attacker_net)
    # attacker_net.float()
    # attacker_net.eval()

    if config.attack == 'no':
        if config.defense == 'nodefense':
            net = nn.Sequential(norm_layer, base_net).cuda()
        else:
            defense = globals()[config.defense](config).cuda()
            net = nn.Sequential(defense, norm_layer, base_net).cuda()

        test_loss, test_acc = test(test_loader, net)
        print('Test Acc: {:.4f}'.format(test_acc))
        return

    elif config.attack == 'nattack':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        config.seed = 123
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()
        asr = nattack(net, test_loader, adv_net)

    elif config.attack == 'onepixel':
        # NOTE: onepixel works on normalised values, uncomment normalization
        # test loader and keydefense transformation
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(defense, base_net).cuda()
        config.seed = 123
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, base_net).cuda()
        asr = attack_all(
            net,
            test_loader,
            adv_net,
            pixels=10,
            targeted=False,
            maxiter=100,
            popsize=400,
            verbose=False,
        )
    elif config.attack == 'spsa':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        config.seed = 123
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()
        adversary = LinfSPSAAttack(
                adv_net, eps=8/255., delta=0.01, lr=0.01, nb_iter=100,
                nb_sample=256, max_batch_size=64, targeted=False,
                loss_fn=None, clip_min=0.0, clip_max=1.0
                )
        asr = spsa(test_loader, net, adversary)

    elif config.attack == 'pgd':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        config.seed = 123
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()

        if config.defensae == 'FFX':
            adv_net = nn.Sequential(norm_layer, base_net).cuda()
            asr = pgd_ffx(test_loader, net, adv_net, adv_defense, 8/255.)
        else:
            adversary = LinfPGDAttack(
                    adv_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=8/255., nb_iter=50, eps_iter=2/255., rand_init=True,
                    clip_min=0.0, clip_max=1.0, targeted=False
                    )
            asr = pgd(test_loader, net, adversary)

        # adv_net = nn.Sequential(norm_layer, base_net).cuda()
        # adv_net = nn.Sequential(norm_layer, attacker_net).cuda()
        # adv_net = nn.Sequential(adv_defense, norm_layer, attacker_net).cuda()

        # ffx
        # epss = [2, 4, 8, 16, 22, 32]
        # epss = [8]
        # for eps in epss:
        #     # _, test_acc = pgd_acc(test_loader, net, adv_net, adv_defense, eps/255.)
        #     # print("ACC (eps: {}): {:.4f}".format(eps, test_acc))
        #     asr = pgd_ffx(test_loader, net, adv_net, adv_defense, eps/255.)
        #     print("ASR (eps: {}): {:.4f}".format(eps, asr))

        # epss = [2, 4, 8, 16, 22, 32]
        # epss = [8]
        # for eps in epss:
        #     adversary = LinfPGDAttack(
        #             adv_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        #             eps=eps/255., nb_iter=50, eps_iter=2/255., rand_init=True,
        #             clip_min=0.0, clip_max=1.0, targeted=False
        #             )
        #     # _, test_acc = pgd_acc_advertorch(test_loader, net, adversary)
        #     # print("ACC (eps: {}): {:.4f}".format(eps, test_acc))
        #     asr = pgd(test_loader, net, adversary)
        #     print("ASR (eps: {}): {:.4f}".format(eps, asr))
        # return

    elif config.attack == 'cw':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        config.seed = 123
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()
        adversary = CarliniWagnerL2Attack(
                adv_net, 10, confidence=0, targeted=False,
                learning_rate=0.01, binary_search_steps=9,
                max_iterations=1000,
                clip_min=0.0, clip_max=1.0)
        asr = cw(test_loader, net, adversary)

    elif config.attack == 'ead':
        adversary = ElasticNetL1Attack(
                adv_net, 10, confidence=0, targeted=False,
                learning_rate=0.01, binary_search_steps=9,
                max_iterations=1000,
                clip_min=0.0, clip_max=1.0)
        asr = ead(test_loader, net, adversary)

    elif config.attack == 'randkey':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(norm_layer, base_net).cuda()
        config.seed = random_key_search(test_loader, net, config)
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        adv_defense = globals()[config.defense](config).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()

        if config.defensae == 'FFX':
            adv_net = nn.Sequential(norm_layer, base_net).cuda()
            asr = pgd_ffx(test_loader, net, adv_net, adv_defense, 8/255.)
        else:
            adversary = LinfPGDAttack(
                    adv_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=8/255., nb_iter=50, eps_iter=2/255., rand_init=True,
                    clip_min=0.0, clip_max=1.0, targeted=False
                    )
            asr = pgd(test_loader, net, adversary)

    elif config.attack == 'heuristickey':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(norm_layer, base_net).cuda()
        adv_defense = heuristic_key_search(test_loader, net, config)
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        adv_net = nn.Sequential(adv_defense, norm_layer, base_net).cuda()

        if config.defensae == 'FFX':
            adv_net = nn.Sequential(norm_layer, base_net).cuda()
            asr = pgd_ffx(test_loader, net, adv_net, adv_defense, 8/255.)
        else:
            adversary = LinfPGDAttack(
                    adv_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=8/255., nb_iter=50, eps_iter=2/255., rand_init=True,
                    clip_min=0.0, clip_max=1.0, targeted=False
                    )
            asr = pgd(test_loader, net, adversary)

    elif config.attack == 'invtransform':
        defense = globals()[config.defense](config).cuda()
        net = nn.Sequential(norm_layer, base_net).cuda()
        adv_defense = heuristic_key_search(test_loader, net, config)
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        adv_net = nn.Sequential(norm_layer, base_net).cuda()
        adversary = LinfPGDAttack(
                adv_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=8/255., nb_iter=50, eps_iter=2/255., rand_init=True,
                clip_min=0.0, clip_max=1.0, targeted=False
                )
        asr = inverse_transform_attack(test_loader, net, adversary, adv_defense)

    elif config.attack == 'eot':
        net = nn.Sequential(defense, norm_layer, base_net).cuda()
        adv_net = nn.Sequential(norm_layer, base_net).cuda()
        asr = eot_attack(test_loader, net, adv_net, adv_defense)

    print("ASR: {:.4f}".format(asr))


if __name__ == "__main__":
    main()
