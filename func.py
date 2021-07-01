import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import tensorflow as tf

def find_min_max_range(true, pred):
    minRange = min(true)
    minPred = min(pred)
    if minPred < minRange: minRange = minPred
        
    maxRange = max(true)
    maxPred = max(pred)
    if maxPred > maxRange: maxRange = maxPred
        
    return (minRange, maxRange)

def make_feature_plots(true, prediction, xlabel, particle, bins, density, ranges=None):
    if ranges == None: ranges = find_min_max_range(true, prediction)
        
    plt.figure(figsize=(7,5))
    plt.hist(prediction, bins=bins, histtype='step', density=density, range = ranges)
    plt.hist(true, bins=bins, histtype='step', density=density, range = ranges)
    plt.yscale('log', nonpositive='clip')
    plt.ylabel('Prob. Density(a.u.)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.legend([particle+' Predicted', particle+' True'])
    plt.show()
    
def mse_loss(true, prediction):
    loss = tf.reduce_mean(tf.math.square(true - prediction),axis=-1)
    return loss