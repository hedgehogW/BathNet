from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import numpy

#cut-mix Cut function
def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2 

# basic transform argument
def make_transforms(phase,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if phase == 'train':
        transforms = transforms.Compose(
            [
                transforms.OneOf([            
                    transforms.HorizontalFlip(p=0.5),
                    transforms.VerticalFlip(p=0.5),
                    transforms.Transpose(p=0.5)
                ]),
                transforms.Resize(128,128, p=1),
                transforms.Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
    else:
        transforms = transforms.Compose(
            [
                transforms.Resize(128, 128, p=1),
                transforms.Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
    return transforms

def PILImageToCV(img):

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img
 
 
def CVImageToPIL(img):
 
    img2 = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img2

def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert(img):
    debug = 1
    ba = np.array(img)
    h, w, _ = ba.shape
 
    # Here 1.2, 32, 5, 0.8 are all parameters that can be adjusted later. I just think it's good to use this to extract the background.
    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)
        foreground = (ba > max_bg + 5).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()
    else:
        bbox = None

    if bbox is None:
        if debug > 0:
            print
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([128,128]) 
    return resized

def scaleRadius(img,scale):
    x = img[int(img.shape[0]/2),:,:].sum(1) 
    r = (x>x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)


def ExImg(img):
   #get the PIL input
   scale = 200
   #scale img to a given radius
   img = PILImageToCV(img)
   a=scaleRadius(img, scale)
   #subtract local mean colors
   a=cv2.addWeighted(a,4,
				cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
   #remove out er 10%
   b = numpy.zeros(a.shape)
   cv2.circle(b,(a.shape[1]//2, a.shape[0]//2 ),int(scale*0.9),(1, 1, 1), -1, 8, 0)
   a = a*b+128*(1-b)
   a = cv2.resize(a,(128,128))
   #a = CVImageToPIL(a)
   return a
   
def ExImg2(img):

   img = PILImageToCV(img)
   img = cv2.resize(img,(128,128))
   
   return img