from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os

def gau_kernel(k, fwhm):
    kernel = np.zeros((k, k))
    if fwhm != 0:
        sigma = fwhm / 2.355
        for i in range(k):
            for j in range(k):
                x = i - (k-1) / 2
                y = j - (k-1) / 2
                r2 = np.abs(x**2) + np.abs(y**2)
    #             print(i, j, x, y, r2)
                kernel[i, j] = np.exp(-r2/(2*sigma**2))
    else:
        kernel += 1
    return kernel/np.sum(kernel)


def k4conv(data, k):
    N = data.shape[0]
    min_idx = N // (2*k)
    kernels = np.zeros((k, k, k))
    for i in range(k):
        kernels[:, :, i] = gau_kernel(k, i//2 + 3)
        
    img_conv = np.zeros((N, N, k))
    for i in range(k):
        psf = np.squeeze(kernels[:, :, i])
        img_conv[:, :, i] = conv2(data, psf, 'same')
        
    img_Y = img_conv[:, :, -1]
    for i in range(k):
        idx = min_idx * i
        img_Y[idx:-idx, idx:-idx] = img_conv[idx:-idx, idx:-idx, k-i-1]
    return img_Y


imgX_25k = np.zeros((512, 512, 1000))
imgY_25k = np.zeros((512, 512, 1000))
cnt = 0
cnt_k = 1

for idx in range(25000):
    img_name = os.path.basename(str(idx).zfill(5)+".jpg")
    # 3 channels are the same
    img_X = np.asarray(Image.open(img_path))[:, :, 0]
    img_Y = k4conv(img_X, k=16)

    imgX_25k[:, :, cnt] = img_X
    imgY_25k[:, :, cnt] = img_Y
    print(str(idx).zfill(5)+".jpg")
    cnt += 1

    if cnt >=1000:
        np.save("imgX_"+str(cnt_k)+"k.npy", imgX_25k)
        np.save("imgY_"+str(cnt_k)+"k.npy", imgY_25k)
        cnt_k += 1
        cnt = 0
        print(cnt_k*1000)

