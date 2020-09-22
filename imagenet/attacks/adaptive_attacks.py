import torch
import random
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
