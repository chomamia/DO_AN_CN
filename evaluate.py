import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet
import cv2


ckpt_path = './pretrained/New_000006_13_6.ckpt'
path_P3M = "/home/huuphuong/Study/DO_AN/dataset/P3M-10k/validation"
path_1000 = os.path.join(path_P3M, "P3M-1000")
mask_1000 = os.path.join(path_1000, "mask")
original_1000 = os.path.join(path_1000, "original_image")
name_imgs = os.listdir(original_1000)
def remove_background(im):
    # define hyper-parameters
    ref_size = 512
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
    return matte
def calculate_evaluate(mask, matte):
    mask = np.asarray(mask)
    mask = mask[:,:,0]
    matte = np.asarray(matte)
    mask[mask != 0] = 255
    matte[matte != 0] = 255
    correct = mask[mask == matte]
    correct = correct[correct == 255]
    total = len(mask[mask == 255]) + len(matte[matte == 255]) - len(correct)
    rate = len(correct) / total
    return rate
sum_rate = 0
for name_img in range(0, len(name_imgs)):
    img = Image.open(original_1000 + "/" + name_imgs[name_img])
    name_mask = os.path.basename(original_1000 + "/" + name_imgs[name_img])
    name_mask = os.path.splitext(name_mask)[0]
    mask = Image.open(mask_1000 + "/" + name_mask + ".png")
    matte = remove_background(img)
    rate = calculate_evaluate(mask, matte)
    sum_rate = sum_rate + rate
print("Phuong phap tach nen vat the")
print("So luong anh kiem tra: 1000")
print("Phuong phap danh gia theo ty le chinh xac")
print("Accuracy: {:.2f}".format((sum_rate / len(name_imgs)) * 100)) 