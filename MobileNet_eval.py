#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:31:56 2021

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

from PIL import Image
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import seaborn as sns

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
    x = layers.Dense(2, name='fc')(x)
    x = layers.Reshape((classes,), name='reshape_1')(x)
    # imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Activation(activation=classifier_activation,
                            name='predictions')(x)

    model = keras.Model(inputs=[img_input], outputs=[x])    
    return model

def zero_pad(img, size=224):
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

def get_precision_recall(ax, y_true, y_pred, title, boostrap=5, plot=True):
    
    def delta_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return h
    
    ap_score=[]
    for i in range(boostrap):
        pred_bt, y_bt = resample(y_pred, y_true)
        ap_score.append(average_precision_score(y_bt, pred_bt))
    
    AP = average_precision_score(y_true, y_pred)
    precision, recall, thresholds=precision_recall_curve(y_true, y_pred)
    
    if plot:
        delta = delta_confidence_interval(ap_score)
        
        sns.set_style('ticks')
    #    plt.figure()
        ax.plot(recall, precision, color='red', lw=2,
                 label='AUC = {:.3f}, \n95% C.I. = [{:.3f}, {:.3f}]'.format(AP, AP-delta, AP+delta), alpha=.8)
        
        ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid()
    return thresholds

def get_auc(ax, y_true, y_score, title, plot=True):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_score)
    auc_keras = auc(fpr_keras, tpr_keras)
    
    optimal_idx = np.argmax(tpr_keras - fpr_keras)
    optimal_threshold = thresholds_keras[optimal_idx]
    
    if plot:
        ci = get_CI(y_true, y_score)
        
        sns.set_style('ticks')
    #    plt.figure()
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Chance', alpha=.8)
        ax.plot(fpr_keras, tpr_keras, color='red', lw=2,
                 label='AUC = {:.3f}, \n95% C.I. = [{:.3f}, {:.3f}]'.format(auc_keras, ci[0], ci[1]), alpha=.8)
        
        ax.set_xlabel('1-Specificity', fontsize=16, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=16, fontweight='bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid()
    return optimal_threshold

def get_CI(y_true, y_score, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_score)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    
    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    return ci

def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2
def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2
def plot_confusion_matrix(ax, cm, target_names, title='Confusion matrix', cmap=None, normalize=True, fontsize=16):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names, rotation=90)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    ax.tick_params(labelsize=fontsize-3)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass), fontsize=fontsize, fontweight='bold')

import scipy
import time
from scipy import stats
from sklearn.utils import resample

Model_size = [64,64,128,224,320,448,512]
Model_weights = ['/home/joe/Project/Water/MobileNet_64_checkpoints/2021911030.h5',
                 '/home/joe/Project/Water/MobileNet_64_checkpoints/2021911030.h5',
                 '/home/joe/Project/Water/MobileNet_128_checkpoints/2021913955.h5',
                 '/home/joe/Project/Water/MobileNet_224_checkpoints/2021913101.h5',
                 '/home/joe/Project/Water/MobileNet_320_checkpoints/20219281546.h5',
                 '/home/joe/Project/Water/MobileNet_448_checkpoints/20219291653.h5',
                 '/home/joe/Project/Water/MobileNet_512_checkpoints/2021108949.h5']

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
Result = []
Result.append(df_test['malignancy'])
for idx in range(len(Model_size)):
    # model = MobileNet([Model_size[idx],Model_size[idx],3])   
    model = keras.models.load_model(Model_weights[idx])
    y_true = []
    y_pred = []
    time_use = []
    time_count = 0
    for i in range(len(df_test)):
        y_true.append(df_test['malignancy'][i])
        x = Image.open(df_test['roi path'][i])
        x = np.array(x)
        x = zero_pad(x,Model_size[idx])
        x = preprocess_input(x)
        x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
        start = time.perf_counter()  # 返回系统运行时间
        y_pred.append(model.predict(x))
        end = time.perf_counter()
        time_count += end-start
    time_use.append(time_count/len(df_test))
    print("model of {} time use {}".format(Model_size[idx], np.average(time_use)))
    # y_pred_save = y_pred
    # y_pred = y_pred_save
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0],2)
    y_pred_1 = y_pred[:,1]   
    thresh_0=get_auc(0, np.array(y_true), np.array(y_pred_1), 'Malignancy', plot=False)
    print(thresh_0)
    # thresh_0 = 0.5
    y_pred_comp_lvl=[1 if y>thresh_0 else 0 for y in y_pred_1]
    cm_comp=confusion_matrix(y_true, y_pred_comp_lvl)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout(pad=2, w_pad=2., rect=(0, 0, 1.3, 1))
    fig.set_figheight(8)
    fig.set_figwidth(10)
    thresh_0=get_auc(axes[0, 0], np.array(y_true), np.array(y_pred_1), 'Malignancy=0 vs 1 Mobilenet{}'.format(Model_size[idx]))
    thresh_AP=get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred_1), 'Malignancy=0 vs 1 Mobilenet{}'.format(Model_size[idx]))
    plot_confusion_matrix(axes[1, 0], cm_comp, ["0", "1"], title='Malignancy Mobilenet{}'.format(Model_size[idx]), normalize=False)
    plot_confusion_matrix(axes[1, 1], cm_comp, ["0", "1"], title='Malignancy (normalized) Mobilenet{}'.format(Model_size[idx]))
    print('f1 score is: {:.3f}'.format(f1_score(y_true, y_pred_comp_lvl)))
    Result.append(y_pred_1)
df_result = pd.DataFrame(Result).T
col = ['y_true']
for size in Model_size:
    col.append('MobileNet_'+str(size))
df_result.columns = col
df_result.to_csv('mobilenet_result.csv',index=False,header=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    