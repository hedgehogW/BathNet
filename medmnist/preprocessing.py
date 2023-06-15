from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageFilter


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

   
        """if bbox is None:
            print('bbox none for this.')
        else:
            left, upper, right, lower = bbox
            # If the box obtained is less than 80% of the original image, there is a possibility of a bug, so discard this box.
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for this')
                bbox = None"""
    else:
        bbox = None

    if bbox is None:
        if debug > 0:
            print
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([128, 128]) 
    return resized

def main():
    img = convert('meta.jpg', 512)
    img.show()


if __name__ == '__main__':
    main()
