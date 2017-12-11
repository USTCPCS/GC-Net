import os
import sys
import re
import numpy as np
from PIL import Image

def load_pfm(file_name, downsample= False):
    if downsample:
        pass

    file = open(file_name)

    color = None
    width = None
    height = None
    scale = None
    endian = None


    header = file.readline().rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    # big-endian
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    img = np.flipud(data)
    file.close()

    return img, scale


def save_pfm(file_name, image, scale=1):
    file = open(file_name, 'w')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    np.flipud(image).tofile(file)

if __name__ == '__main__':
    print('...main')
    img, scale = load_pfm('/home/g1002/Sampler/Sampler/FlyingThings3D/disparity/0006.pfm')
    print(np.count_nonzero(img))
    print(img.shape)
    img = Image.fromarray(img)
    img.show()