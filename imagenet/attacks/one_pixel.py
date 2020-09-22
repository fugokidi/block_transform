# https://github.com/DebangLi/one-pixel-attack-pytorch

import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .differential_evolution import differential_evolution


# be careful perturbation on normalized values
def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - 0.485) / 0.229
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - 0.456) / 0.224
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - 0.406) / 0.225
        count += 1

    return imgs


def predict_classes(xs, img, target_calss, adv_net, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())
    input = imgs_perturbed.cuda()
    with torch.no_grad():
        predictions = nn.Softmax(dim=1)(adv_net(input)).data.cpu().numpy()[:, target_calss]

    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_calss, adv_net, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img.clone())
    input = attack_image.cuda()
    with torch.no_grad():
        confidence = nn.Softmax(dim=1)(adv_net(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if verbose:
        print("Confidence: {:.4f}".format(confidence[target_calss]))
    if (targeted_attack and predicted_class == target_calss) or (
        not targeted_attack and predicted_class != target_calss
    ):
        return True


def attack(
    img, label, net, adv_net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False
):
    # img: 1*3*W*H tensor
    # label: a number

    targeted_attack = target is not None
    target_calss = target if targeted_attack else label

    bounds = [(0, 288), (0, 288), (0, 255), (0, 255), (0, 255)] * pixels

    popmul = max(1, popsize // len(bounds))

    predict_fn = lambda xs: predict_classes(xs, img, target_calss, adv_net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_calss, adv_net, targeted_attack, verbose
    )

    inits = np.zeros([popmul * len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 5 + 0] = np.random.random() * 288
            init[i * 5 + 1] = np.random.random() * 288
            init[i * 5 + 2] = np.random.normal(128, 127)
            init[i * 5 + 3] = np.random.normal(128, 127)
            init[i * 5 + 4] = np.random.normal(128, 127)

    attack_result = differential_evolution(
        predict_fn,
        bounds,
        maxiter=maxiter,
        popsize=popmul,
        recombination=1,
        atol=-1,
        callback=callback_fn,
        polish=False,
        init=inits,
    )

    attack_image = perturb_image(attack_result.x, img)
    attack_var = attack_image.cuda()
    with torch.no_grad():
        predicted_probs = nn.Softmax(dim=1)(net(attack_var)).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (
        targeted_attack and predicted_class == target_calss
    ):
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_all(
    net, loader, adv_net, pixels=1, targeted=False, maxiter=75, popsize=400,
    verbose=False
):
    correct = 0
    success = 0

    iterator = tqdm(loader)
    for batch_idx, (input, target) in enumerate(iterator):

        img_var = input.cuda()
        with torch.no_grad():
            prior_probs = nn.Softmax(dim=1)(net(img_var))
        _, indices = torch.max(prior_probs, 1)

        if target[0] != indices.data.cpu()[0]:
            continue

        correct += 1
        target = target.numpy()

        targets = [None] if not targeted else range(1000)

        for target_calss in targets:
            if targeted:
                if target_calss == target[0]:
                    continue

            flag, x = attack(
                input,
                target[0],
                net,
                adv_net,
                target_calss,
                pixels=pixels,
                maxiter=maxiter,
                popsize=popsize,
                verbose=verbose,
            )

            success += flag
            if targeted:
                success_rate = float(success) / (9 * correct)
            else:
                success_rate = float(success) / correct

            # if flag == 1:
            #     print(
            #         "Success Rate: {:.4f} ({}/{}) [(x,y) = ({},{}) and (R,G,B)=({},{},{})]".format(
            #             success_rate,
            #             success,
            #             correct,
            #             x[0],
            #             x[1],
            #             x[2],
            #             x[3],
            #             x[4],
            #         )
            #     )
        iterator.set_description('OnePixel Progress:{}/{}'.format(success, correct))

        # stop after attacking 100 samples
        if correct == 1000:
            break

    return success_rate
