import cv2
import numpy as np
from PIL import Image
from tkinter.filedialog import askopenfilename
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.modnet import MODNet

i=0
blur_bg = False
while (i!=1 and i!=2):
    i = int(input("\nSelect mode: \n\t1: background blur \n\t2: select background in computer\n"))
    if i == 2:
        background_path = askopenfilename()
        background = cv2.imread(background_path)
        background_img = cv2.resize(background, (672,512))
    if i==1:
        blur_bg = True
torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = './pretrained/modnet_photographic_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()
print('Init WebCam...')
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('Start matting...')
while (True):
    _, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    if blur_bg:
        background_img = frame_np
        background_img = cv2.resize(background_img, (672,512))
        background_img = cv2.blur(background_img, (50,20))

        background_img = np.asarray(0.8*background_img - 5, dtype=int)   # cast pixel values to int
        background_img[background_img>255] = 255
        background_img[background_img<0] = 0

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    fg_np = matte_np * frame_np + (1 - matte_np) * background_img
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')