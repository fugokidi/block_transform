import torch
import torch.nn.functional as F
from tqdm import tqdm


def pgd(test_loader, net, adversary):
    correct = 0
    success = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()

        output = net(data)
        # if output.max(1)[1] != target:
        #     continue
        # correct += 1
        adv = adversary.perturb(data, target)
        adv_output = net(adv)
        # if adv_output.max(1)[1] != target:
        #     success += 1

        correct += (output.max(1)[1] == target).sum().item()
        success += ((output.max(1)[1] == target) & (adv_output.max(1)[1] != target)).sum().item()

        if correct > 1000:
            break

        iterator.set_description('PGD Progress:{}/{}'.format(success, correct))

    return success / float(correct)


def pgd_acc_advertorch(test_loader, net, adversary):
    test_loss = 0
    test_acc = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()
        adv = adversary.perturb(data, target)
        output = net(adv)
        loss = F.cross_entropy(output, target)
        test_loss += loss.item() * data.size(0)
        test_acc += (output.max(1)[1] == target).sum().item()
        iterator.set_description("Test Loss: {:.6f}".format(loss.item()))

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc


def pgd_acc(test_loader, net, model, defense, eps):
    test_loss = 0
    test_acc = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()
        delta = pgd_linf(model, data, target, defense, eps)
        output = net(data + delta)
        loss = F.cross_entropy(output, target)
        test_loss += loss.item() * data.size(0)
        test_acc += (output.max(1)[1] == target).sum().item()
        iterator.set_description("Test Loss: {:.6f}".format(loss.item()))

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc


def pgd_ffx(test_loader, net, adv_net, defense, eps):
    correct = 0
    success = 0
    iterator = tqdm(test_loader)

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()

        output = net(data)

        # for one image at a time
        # if output.max(1)[1] != target:
        #     continue
        # correct += 1

        delta = pgd_linf(adv_net, data, target, defense, eps)
        adv_output = net(data + delta)

        # if adv_output.max(1)[1] != target:
        #     success += 1

        correct += (output.max(1)[1] == target).sum().item()
        success += ((output.max(1)[1] == target) & (adv_output.max(1)[1] != target)).sum().item()

        if correct > 1000:
            break

        iterator.set_description('PGD Progress:{}/{}'.format(success, correct))

    return success / float(correct)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_linf(model, X, y, defense, eps=8/255., alpha=2/255., restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    X_ori = X.clone()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-eps, eps)
        delta.data = clamp(delta, 0. - X, 1. - X)
        # initial X_adv
        X_adv = X_ori + delta
        for _ in range(50):
            X = defense(X_adv)
            X = X.clone().detach().requires_grad_(True)
            output = model(X)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = X.grad.detach()

            delta = X_adv - X_ori
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -eps, eps)
            d = clamp(d, 0. - X_ori[index[0], :, :, :], 1. - X_ori[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d

            X_adv = X_ori + delta

        all_loss = F.cross_entropy(model(defense(X_adv)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
