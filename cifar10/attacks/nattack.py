# https://github.com/Cold-Winter/Nattack

import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5


def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x), -1, keepdims=True))


def nattack(model, test_loader, adv_model):
    npop = 300     # population size
    sigma = 0.1    # noise standard deviation
    alpha = 0.02  # learning rate
    # alpha = 0.001  # learning rate
    boxmin = 0
    boxmax = 1
    boxplus = (boxmin + boxmax) / 2.
    boxmul = (boxmax - boxmin) / 2.

    epsi = 0.031
    epsilon = 1e-30

    totalImages = 0
    succImages = 0
    faillist = []
    successlist = []
    printlist = []

    start_time = time.time()
    iterator = tqdm(test_loader)
    for i, (inputs, targets) in enumerate(iterator):

        success = False
        input_var = inputs.clone().cuda()
        inputs = inputs.squeeze(0).numpy().transpose(1, 2, 0)
        modify = np.random.randn(1, 3, 32, 32) * 0.001
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(model(input_var))
        _, indices = torch.max(probs, 1)

        if targets[0] != indices.data.cpu()[0]:
            continue

        totalImages += 1
        for runstep in range(500):
            Nsample = np.random.randn(npop, 3, 32, 32)

            modify_try = modify.repeat(npop, 0) + sigma * Nsample

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2, 0, 1)
            #print('newimg', newimg,flush=True)

            inputimg = np.tanh(newimg + modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
                realinputimg = np.tanh(newimg+modify) * boxmul + boxplus
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(realdist, -epsi, epsi)
                # print('realclipdist :', realclipdist, flush=True)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                l2real = np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                # print(inputs.shape)
                #outputsreal = model(realclipinput.transpose(0,2,3,1)).data.cpu().numpy()
                input_var = torch.from_numpy(realclipinput.astype('float32')).cuda()

                outputsreal = model(input_var).data.cpu().numpy()[0]
                outputsreal = softmax(outputsreal)
                #print(outputsreal)
                # print('probs ', np.sort(outputsreal)[-1:-6:-1])
                # print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                # print('negative_probs ', np.sort(outputsreal)[0:3:1])

                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    succImages += 1
                    success = True
                    # print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    # print('lirealsucc: '+str(realclipdist.max()))
                    successlist.append(i)
                    printlist.append(runstep)

#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                    break
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32) #.reshape(npop,3,32,32)
            target_onehot =  np.zeros((1,10))


            target_onehot[0][targets]=1.
            clipinput = np.squeeze(clipinput)
            clipinput = np.asarray(clipinput, dtype='float32')
            input_var = torch.from_numpy(clipinput).cuda()
            #outputs = model(clipinput.transpose(0,2,3,1)).data.cpu().numpy()
            outputs = adv_model(input_var).data.cpu().numpy()
            outputs = softmax(outputs)

            target_onehot = target_onehot.repeat(npop,0)


            real = np.log((target_onehot * outputs).sum(1)+epsilon)
            other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+epsilon)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32)) #.reshape(3,32,32))
        iterator.set_description('Nattack Progress:{}/{}'.format(succImages, totalImages))
        end_time = time.time()
        if not success:
            faillist.append(i)
            # print('failed:', faillist)
        # else:
        #     print('successed:', successlist)
        if totalImages == 1000:
            break
    # print(faillist)
    success_rate = succImages/float(totalImages)
    # print('run steps: ',printlist)
    # np.savez('runstep', printlist)
    # print('succ rate', success_rate)
    # print('time taken (min): {:.4f}'.format((end_time - start_time) / 60))
    return success_rate


#     print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))
