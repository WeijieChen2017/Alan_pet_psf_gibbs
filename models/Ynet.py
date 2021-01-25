from tensorflow.keras import Input, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D, Multiply
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Masking, Add

'''
from: https://github.com/pietz/unet-keras/blob/master/unet.py
modified by abmcmillan@wisc.edu to enable continuous output.

s

U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def encoder(m, dim, depth, acti, bn,  mp, res):
    for idx in range(depth):
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
    return m

def decoder(m, dim, out_ch, depth, acti, do, bn, up, res):
    m = conv_block(m, dim, acti, bn, res, do)
    for idx in range(depth):
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        m = conv_block(m, dim, acti, bn, res)
    m = Conv2D(out_ch, 1, activation='linear')(m)
    return m

def YNet(img_shape_PET, img_shape_MRI, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i_pet = Input(shape=img_shape_PET)
    i_mri = Input(shape=img_shape_MRI)
    th_mri = Input(shape=(1,))
    th_pet = Input(shape=(1,))
    en_pet = encoder(m = i_pet, dim=start_ch, depth=depth, acti=activation,
                     bn=batchnorm, mp=maxpool, res=residual)
    en_mri = encoder(m = i_mri, dim=start_ch, depth=depth, acti=activation,
                     bn=batchnorm, mp=maxpool, res=residual)
    gated_en_pet = Multiply()([en_pet, th_pet])
    gated_en_mri = Multiply()([en_mri, th_mri])
    mid = Add()([gated_en_pet, gated_en_mri])
    de = decoder(m=mid, dim=start_ch, out_ch=out_ch, depth=depth, acti=activation,
                do=dropout, bn=batchnorm, up=upconv, res=residual)
    return Model(inputs=[i_mri, i_pet, th_mri, th_pet], outputs=de)


def level_block(m, dim, depth, inc, acti, do, bn,  mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)

def UNetContinuous(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Concatenate()([i, o])
    o = Conv2D(out_ch, 1, activation='linear')(o)
    return Model(inputs=i, outputs=o)

def UNetContinuousMasked(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, mask_value=0.0):
    #i = Input(shape=img_shape)
    i = Masking(mask_value,input_shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='linear')(o)
    return Model(inputs=i, outputs=o)
