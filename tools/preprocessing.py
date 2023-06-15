from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageFilter


def convert(fname, crop_size):
    img = Image.open("C:/Users/XZH376900/Desktop/GAN/Eyes/FFA Label Transfer/CFP/11265629 OD.jpg")
    debug = 1
    # blurred = img.filter(ImageFilter.BLUR) 这里学长用了blur后的，可能可以更均匀的识别背景？ 我没有很明白
    # ba = np.array(blurred)
    ba = np.array(img)
    h, w, _ = ba.shape
    if debug > 0:
        print("h=%d, w=%d" % (h, w))
    # 这里的1.2, 32, 5, 0.8都是后续可以调整的参数。 只是暂时觉得用这个来提取背景不错。
    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)
        foreground = (ba > max_bg + 5).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if debug > 0:
            print(foreground, left_max, right_max, bbox)
        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # 如果弄到的框小于原图的80%，很可能出bug了，就舍弃这个框。
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        if debug > 0:
            print
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def main():
    img = convert('meta.jpg', 512)
    img.show()


if __name__ == '__main__':
    main()
