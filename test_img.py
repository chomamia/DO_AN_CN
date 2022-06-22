import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet
from tkinter.filedialog import askopenfilename
import cv2

image_names = askopenfilename()
input_path = image_names
ckpt_path = './pretrained/New_000006.ckpt'
background_path = askopenfilename()
def remove_background(img):
    # check input arguments
    if not os.path.exists(input_path):
        print('Cannot find input path: {0}'.format(input_path))
        exit()
    if not os.path.exists(ckpt_path):
        print('Cannot find ckpt path: {0}'.format(ckpt_path))
        exit()

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

    # inference images
    print('Process image: {0}'.format(input_path))

    # read image
    im = Image.open(input_path)
    # cv2.imshow('original', im)

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


def combined_display(image, matte,background):
    # calculate display resolution
    w, h = image.width, image.height
    rw, rh = 800, int(h * 800 / (3 * w))
    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + background * (1 - matte)
    # combine image, foreground, and alpha into one line
    combined = np.concatenate((image,foreground, matte * 255), axis=1)
    combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
    return combined


image = Image.open(input_path)
background = cv2.imread(background_path)
background = cv2.resize(background, image.size)
matte = remove_background(image)
img = combined_display(image, matte, background)
img.show()