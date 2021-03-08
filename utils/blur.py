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

n_slice = 1000
imgX_25k = np.zeros((n_slice, 512, 512, 1), dtype=np.single)
imgY_25k = np.zeros((n_slice, 512, 512, 1), dtype=np.single)
cnt = 0
cnt_k = 1

for ii in range(5000):
    idx = ii
    img_name = str(idx).zfill(5)
    # 3 channels are the same
    try:
        img_X = np.asarray(Image.open(img_name+".jpg"))[:, :, 0]
        img_Y = k4conv(img_X, k=16)
    except:
        print(img_name+".jpg")
    else:
        # np.save(img_name+"_X.npy", img_X)
        # np.save(img_name+"_Y.npy", img_Y)
        # print(str(idx).zfill(5)+".jpg")
        imgX_25k[cnt, :, :, 0] = img_X
        imgY_25k[cnt, :, :, 0] = img_Y
        cnt += 1

        if cnt >=n_slice:

            imgX_25k = (imgX_25k - np.amin(imgX_25k)) / (np.amax(imgX_25k) - np.amin(imgX_25k))
            imgY_25k = (imgY_25k - np.amin(imgY_25k)) / (np.amax(imgY_25k) - np.amin(imgY_25k))
            print(np.std(imgX_25k), np.std(imgY_25k))
            np.save("imgX_"+str(cnt_k)+"k.npy", imgX_25k)
            np.save("imgY_"+str(cnt_k)+"k.npy", imgY_25k)
            cnt = 0
            print(cnt_k*n_slice)
            cnt_k += 1

imgX_25k = (imgX_25k - np.amin(imgX_25k)) / (np.amax(imgX_25k) - np.amin(imgX_25k))
imgY_25k = (imgY_25k - np.amin(imgY_25k)) / (np.amax(imgY_25k) - np.amin(imgY_25k))
print(np.std(imgX_25k), np.std(imgY_25k))
np.save("imgX_"+str(cnt_k)+"k.npy", imgX_25k)
np.save("imgY_"+str(cnt_k)+"k.npy", imgY_25k)
print(cnt_k*n_slice)
