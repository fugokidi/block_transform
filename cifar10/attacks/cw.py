import torch
from tqdm import tqdm


def cw(test_loader, net, adversary):
    correct = 0
    success = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()

        output = net(data)
        adv = adversary.perturb(data, target)
        adv_output = net(adv)

        correct += (output.max(1)[1] == target).sum().item()
        success += ((output.max(1)[1] == target) & (adv_output.max(1)[1] != target)).sum().item()

        if correct > 1000:
            break

        iterator.set_description('CW Progress:{}/{}'.format(success, correct))

    return success / float(correct)
