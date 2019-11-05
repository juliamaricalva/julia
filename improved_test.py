import os
import torch
import torchvision
from PIL import Image

import parser1
import models
import improved_model
import data
import mean_iou_evaluate

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mean_iou_evaluate import mean_iou_score


def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (imgs) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()


            preds.append(pred)



    preds = np.concatenate(preds)

    return mean_iou_score(preds)

#to get the mask
"""def save(data_loader):
    ''' set model to evaluate mode '''
    model.eval()


    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, path) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            n=0
            for i in pred:
#                print(set(i.flatten()))
                result = Image.fromarray((i).astype(np.uint8))
                result.save( "./outputfiles/" + path[n])
                n +=1"""

if __name__ == '__main__':
    args = parser1.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare mode '''
    model = improved_model.Resnet50(args).cuda()

    ''' resume save model '''
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
   # acc1 = save(test_loader)
    print('Testing Accuracy: {}'.format(acc))