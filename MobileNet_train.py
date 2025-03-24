#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:47:49 2021

@author: joe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:04:57 2021

@author: joe
"""

import tensorflow as tf
import keras 
from keras.layers import Conv2D,DepthwiseConv2D,Dense,AveragePooling2D,BatchNormalization,Input
from keras import Model
from keras import Sequential
from keras.layers import ReLU, Reshape
import numpy as np
import pandas as pd
import itertools
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import imutils
from sklearn.utils import shuffle
from skimage.util import random_noise as random_noise

import time

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.mobilenet import preprocess_input


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):


  filters = int(filters * alpha)
  x = layers.Conv2D(
      filters,
      kernel,
      padding='same',
      use_bias=False,
      strides=strides,
      name='conv1')(inputs)
  x = layers.BatchNormalization(name='conv1_bn')(x)
  return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
  """Adds a depthwise convolution block.

  A depthwise convolution block consists of a depthwise conv,
  batch normalization, relu6, pointwise convolution,
  batch normalization and relu6 activation.

  Args:
    inputs: Input tensor of shape `(rows, cols, channels)` (with
      `channels_last` data format) or (channels, rows, cols) (with
      `channels_first` data format).
    pointwise_conv_filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the pointwise convolution).
    alpha: controls the width of the network. - If `alpha` < 1.0,
      proportionally decreases the number of filters in each layer. - If
      `alpha` > 1.0, proportionally increases the number of filters in each
      layer. - If `alpha` = 1, default number of filters from the paper are
      used at each layer.
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel. The total number of depthwise convolution
      output channels will be equal to `filters_in * depth_multiplier`.
    strides: An integer or tuple/list of 2 integers, specifying the strides
      of the convolution along the width and height. Can be a single integer
      to specify the same value for all spatial dimensions. Specifying any
      stride value != 1 is incompatible with specifying any `dilation_rate`
      value != 1.
    block_id: Integer, a unique identification designating the block number.
      # Input shape
    4D tensor with shape: `(batch, channels, rows, cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(batch, rows, cols, channels)` if
      data_format='channels_last'. # Output shape
    4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
      data_format='channels_last'. `rows` and `cols` values might have
      changed due to stride.

  Returns:
    Output tensor of block.
  """

  pointwise_conv_filters = int(pointwise_conv_filters * alpha)

  if strides == (1, 1):
    x = inputs
  else:
    x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(
        inputs)
  x = layers.DepthwiseConv2D((3, 3),
                             padding='same' if strides == (1, 1) else 'valid',
                             depth_multiplier=depth_multiplier,
                             strides=strides,
                             use_bias=False,
                             name='conv_dw_%d' % block_id)(
                                 x)
  x = layers.BatchNormalization(
       name='conv_dw_%d_bn' % block_id)(
          x)
  x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

  x = layers.Conv2D(
      pointwise_conv_filters, (1, 1),
      padding='same',
      use_bias=False,
      strides=(1, 1),
      name='conv_pw_%d' % block_id)(
          x)
  x = layers.BatchNormalization(
       name='conv_pw_%d_bn' % block_id)(
          x)
  return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=2,
              classifier_activation='softmax',
              **kwargs):

    img_input = layers.Input(shape=input_shape)
    
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    
    x = _depthwise_conv_block(
      x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    
    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    
    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Reshape(shape, name='reshape_1')(x)
    # x = layers.Dropout(dropout, name='dropout')(x)
    # x = layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = layers.Dense(2, name='fc_nobias',use_bias=False)(x)
    x = layers.Reshape((classes,), name='reshape_1')(x)
    # imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Activation(activation=classifier_activation,
                            name='predictions')(x)

    model = keras.Model(inputs=[img_input], outputs=[x])    
    return model

model = MobileNet([512,512,3])

model.load_weights('/home/joe/Project/Water/MobileNet_512_checkpoints/2021108949.h5', by_name=True)
model.summary()


LR = 1e-4

'''
Freeze pre-trained layers
'''
#for layer in test_model.layers:
#    if 'trainable' in layer.name:
#        layer.trainable = True
#    else:
#        layer.trainable = False

model.summary()



model.compile(optimizer=keras.optimizer_v2.adam.Adam(lr=LR), 
                    loss=['binary_crossentropy'], 
                    metrics=['accuracy'])


'''
Prepare data generators
'''
def zero_pad(img, size=512):
    '''
    pad zeros to make a square img for resize
    '''
    h, w, c = img.shape
    if h>w:
        zeros = np.zeros([h, h-w, c]).astype(np.uint8)
        img_padded = np.hstack((img, zeros))
    elif h<w:
        zeros = np.zeros([w-h, w, c]).astype(np.uint8)
        img_padded = np.vstack((img, zeros))
    else:
        img_padded = img
    
    img_resized = (255*resize(img_padded, (size, size), anti_aliasing=True)).astype(np.uint8)
    
    return img_resized


def adjust_gamma(image, gamma=1.2):
    '''
    gamma correction
    '''
 	# build a lookup table mapping the pixel values [0, 255] to
 	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def adjust_gamma_sum(image, gamma=1.2):
    '''
    gamma correction
    '''
 	# build a lookup table mapping the pixel values [0, 255] to
 	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 	# apply gamma correction using the lookup table
    img = cv2.LUT(image, table)
    return img//2+image//2

def Gaussian_noise(img):
    return random_noise(img, mode='gaussian')

def flip_x(img):
    return np.fliplr(img)

def flip_y(img):
    return np.flipud(img)

def translation(img, t_x, t_y):
    '''
    Args:
        t_x, t_y: amount of translation on x and y directions, usually between 0.9 and 1.1
        t_x > 1: right; t_y > 1: down
    '''
    H, W, C = img.shape
    t_x = int((t_x-1) * W)
    t_y *= int((t_y-1) * H)
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    result = cv2.warpAffine(img,M,(W,H))
    return result

#img = imread(TRAIN_DIR + 'L0/L0_0_0_img_ROI.jpg')
#imshow(zoom(img,1.2))

def rotation(img, degree):
    return imutils.rotate(img, degree)

def zoom(img, factor):
    """
    Args:
        factor : amount of zoom as a ratio (0 to Inf)
    """
    H, W, C = img.shape
    new_H, new_W = int(H * factor), int(W * factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_H - H) // 2, max(0, new_W - W) // 2
    y2, x2 = y1 + H, x1 + W
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_H, resize_W = min(new_H, H), min(new_W, W)
    pad_height1, pad_width1 = (H - resize_H) // 2, (W - resize_W) // 2
    pad_height2, pad_width2 = (H - resize_H) - pad_height1, (W - resize_W) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_W, resize_H))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == H and result.shape[1] == W
    return result


def get_shuffled_pairs(df, fname='img_path', flabel='standard', epoch=50):
    '''
    df: dataframe
    fname: column that stores img file path
    flabel: column that stores img label
    epoch: number of epochs
    output: an iterator of tuples (fname, label, sample_weight); sample_weight: weight of the sample's class
    '''
#    df = train_df
    fnames=list(df[fname])
    flabels=sorted(list(df[flabel].unique()), reverse=False)
    ohe_map=pd.Series(flabels).str.get_dummies(', ')
    ohe_labels=[list(ohe_map[i]) for i in list(df[flabel])]
    
    sample_weights=[]
    for i in flabels:
        sample_weights.append(np.sum(df[flabel]==i))
    
    sample_weights = 1/np.array(sample_weights)
    sample_weights = len(flabels)*sample_weights/np.sum(sample_weights)
    sample_weights = np.sum(np.asarray(ohe_labels) * sample_weights, axis=1)
    
    zip_pairs = zip(fnames, ohe_labels, sample_weights)

    pairs=[]
    for fname, label, sample_weight in zip_pairs:
        pairs.append([fname, label, sample_weight])
    
    random_cycle_pairs = []
    for r in range(epoch):
        random.shuffle(pairs)
        random_cycle_pairs.extend(pairs)
    
    zipped = itertools.cycle(random_cycle_pairs)
    
    return zipped

def image_classification_generator(df, batch_size, fpath, fname='img_path', flabel='standard', epoch=50,
                                    random_transformation = False,
                                    four_GC = False,
                                    Gaussian_noise_chance = 0.3,
                                    flip_x_chance = 0.5,
                                    flip_y_chance = 0,
                                    rot_degree = (-0.1, 0.1),
                                    x_translation=(-0.1, 0.1),
                                    y_translation=(-0.1, 0.1),
                                    min_scaling=0.7,
                                    max_scaling=1):
    '''
    fpath: file path that contains data
    '''
    zipped = get_shuffled_pairs(df, epoch=epoch, fname=fname, flabel=flabel)
    while True:
        X = []
        Y = []
        W = []
        for _ in range(batch_size):
            img_id, label, sample_weight = next(zipped)
            
            img=imread(fpath+str(img_id))
            img=zero_pad(img,512)

            # random transformations
            if random_transformation:
                random_nums = np.random.rand(8)
                if four_GC:
                    if random_nums[0]>=0.8:
                        img = adjust_gamma(img, gamma=1.6)
                    elif random_nums[0]>=0.6:
                        img = adjust_gamma_sum(img, gamma=1.6)
                    elif random_nums[0]>=0.4:
                        img = adjust_gamma(img, gamma=0.6)
                    elif random_nums[0]>=0.2:
                        img = adjust_gamma_sum(img, gamma=0.6)
                if random_nums[7] < Gaussian_noise_chance:
                    img = Gaussian_noise(img)
                if random_nums[1] < flip_x_chance:
                    img = flip_x(img)
                if random_nums[2] < flip_y_chance:
                    img = flip_y(img)
                img = rotation(img, (random_nums[6]-0.5)*(rot_degree[1]-rot_degree[0]))
                t_x = 1 + x_translation[0] + random_nums[3]*(x_translation[1] - x_translation[0])
                t_y = 1 + y_translation[0] + random_nums[4]*(y_translation[1] - y_translation[0])
                img = translation(img, t_x, t_y)
                factor = min_scaling+random_nums[5]*(max_scaling-min_scaling)
                img = zoom(img, factor)
            
            img = preprocess_input(img)
            
            X.append(img)
            Y.append(label)
            W.append(sample_weight)
            
        yield np.array(X), np.array(Y), np.array(W)

# laod df
# load train/val/test data and pre processing
# Catadf:a datafram which save the loacation\B vs. M\train vs. test information of patient images
Catadf = pd.read_csv(r"/home/joe/Project/Water/20210910_breast_dataset.csv")


# prepare data for train\val\test
# define label
df_train = Catadf[Catadf['data_usage']=='train']
df_train.reset_index(drop=True, inplace=True)


df_val = Catadf[Catadf['data_usage']=='valid']
df_val = Catadf[Catadf['data_usage']=='valid']
df_val.reset_index(drop=True, inplace=True)


df_test = Catadf[Catadf['data_usage']=='test']
df_test.reset_index(drop=True, inplace=True)



# define label
df_train.loc[(df_train['malignancy']==1),'malignancy']='malignant'
df_train.loc[(df_train['malignancy']==0),'malignancy']='benign'

df_val.loc[(df_val['malignancy']==1),'malignancy']='malignant'
df_val.loc[(df_val['malignancy']==0),'malignancy']='benign'


BATCH_SIZE = 32
NO_OF_EPOCHS = 50
NO_OF_TRAINING_IMAGES = len(df_train)
NO_OF_VAL_IMAGES = len(df_val)

train_gen = image_classification_generator(df=df_train, batch_size=BATCH_SIZE, fpath='', fname='roi path', flabel='malignancy', epoch=NO_OF_EPOCHS, random_transformation=True)
val_gen = image_classification_generator(df=df_val, batch_size=1, fpath='', fname='roi path', flabel='malignancy', epoch=1, random_transformation=False)

X,Y,W = next(train_gen)
Y_1 = model(X)
print(Y_1)

'''
callbacks
'''
#callbacks = []
Earlystop_patience = 15
earlystopper = EarlyStopping(patience=Earlystop_patience, verbose=1)

since = time.time()
time_tuple = time.localtime(since)

model_checkpoint = ModelCheckpoint('./MobileNet_512_checkpoints/{}{}{}{}{}_nobias.h5'.format(time_tuple[0],time_tuple[1],time_tuple[2],time_tuple[3],time_tuple[4]), monitor='val_loss',verbose=1, mode='auto', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor    = 'loss',
                              factor     = 0.2,
                              patience   = 5,
                              verbose    = 1,
                              mode       = 'auto',
                              min_delta  = 0.0001,
                              cooldown   = 0,
                              min_lr     = 0)


results = model.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                                    steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                    validation_data=val_gen, 
                                    validation_steps=(NO_OF_VAL_IMAGES),
                                    callbacks=[earlystopper, model_checkpoint, reduce_lr])

#results = test_model.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
#                                   steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
#                                   validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE))


# plot history for accuracy
plt.figure(figsize=(5, 15))
plt.subplot(311)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()

plt.subplot(312)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.yscale('log')
plt.ylabel('binary cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()

plt.subplot(313)
plt.plot(results.history['lr'])
plt.ylabel('learning_rate')
plt.xlabel('epoch')
plt.grid()

# plt.subplot(224)
# plt.plot(results.history['val_true_positive'])
# plt.plot(results.history['val_false_negative'])
# plt.ylabel('Counts')
# plt.xlabel('epoch')
# plt.legend(['true_normal', 'false_abnormal'], loc='upper left')
# plt.grid()