import torch
import random
import torch.nn.functional as F
from tqdm import tqdm

from keydefense import *


def get_acc(model, X, y, defense):
    test_acc = 0
    model.eval()
    with torch.no_grad():
        output = model(defense(X))
        test_acc += (output.max(1)[1] == y).sum().item()
    return test_acc


def random_key_search(test_loader, model, config):
    X, y = iter(test_loader).next()
    X, y = X.cuda(), y.cuda()
    acc = 0
    key = None
    for _ in tqdm(range(20000)):
        config.seed = random.randrange(2**63 - 1)
        defense = globals()[config.defense](config).cuda()
        current_acc = get_acc(model, X, y, defense)
        if current_acc > acc:
            acc = current_acc
            key = config.seed
            if config.batch_size == 1:
                break
    print(f'Best Key ({config.defense}): {key}')
    return key


def heuristic_key_search(test_loader, model, config):
    X, y = iter(test_loader).next()
    X, y = X.cuda(), y.cuda()
    acc = 0
    config.seed = random.randrange(2**63 - 1)
    defense = globals()[config.defense](config).cuda()
    print('Initial Key:', defense.key)

    for T in tqdm(range(10)):
        for i in range(len(defense.key)):
            for j in range(i+1, len(defense.key)):
                new_key = defense.key.clone()
                new_key[i], new_key[j] = defense.key[j].clone(), defense.key[i].clone()
                defense.key = new_key.clone()
                current_acc = get_acc(model, X, y, defense)
                if current_acc < acc:
                    # revert
                    new_key[i], new_key[j] = defense.key[j].clone(), defense.key[i].clone()
                    defense.key = new_key.clone()
                else:
                    acc = current_acc

    print('Tuned Key:', defense.key)
    return defense


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def inverse_transform_attack(test_loader, net, adversary, adv_defense):
    # double check the model without normalization layer
    correct = 0
    success = 0
    epsilon, lower_limit, upper_limit, alpha = prepare_pgd()
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()

        output = net(data)

        data = adv_defense(data)
        adv = adversary.perturb(data, target)

        data = adv_defense(adv, decrypt=True)
        adv_output = net(data)

        correct += (output.max(1)[1] == target).sum().item()
        success += ((output.max(1)[1] == target) & (adv_output.max(1)[1] != target)).sum().item()

        if correct > 1000:
            break

        iterator.set_description('PGD Progress:{}/{}'.format(success, correct))

    return success / float(correct)


def eot(model, X, y, defense, eps=8/255., alpha=2/255., restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    X_ori = X.clone()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-eps, eps)
        delta.data = clamp(delta, 0. - X, 1. - X)
        # initial X_adv
        X_adv = X_ori + delta
        for _ in range(20):
            grad = torch.zeros_like(X)
            for _ in range(30):
                X = defense(X_adv).contiguous()
                X = X.clone().detach().requires_grad_(True)
                output = model(X)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
                grad += torch.autograd.grad(loss, X, retain_graph=False,
                                            create_graph=False)[0]

                # change key
                # for shuffling
                # key = defense.generate_key(random.randrange(2**63 - 1), binary=False)
                # for np and ffx
                key = defense.generate_key(random.randrange(2**63 - 1), binary=True)
                defense.key = key

            delta = X_adv - X_ori
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -eps, eps)
            d = clamp(d, 0. - X_ori[index[0], :, :, :], 1. - X_ori[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d

            X_adv = X_ori + delta

        all_loss = F.cross_entropy(model(X), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def eot_attack(test_loader, net, model, adv_defense):
    # double check the model without normalization layer
    correct = 0
    success = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()

        output = net(data)

        delta = eot(model, data.clone(), target, adv_defense)

        adv_output = net(data + delta)

        correct += (output.max(1)[1] == target).sum().item()
        success += ((output.max(1)[1] == target) & (adv_output.max(1)[1] != target)).sum().item()

        if correct > 1000:
            break

        iterator.set_description('PGD Progress:{}/{}'.format(success, correct))

    return success / float(correct)

