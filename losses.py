from keras_radam import RAdam

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

from keras import backend as K
import numpy as np

def Active_Contour_Loss(y_true, y_pred): 

    """
    https://github.com/xuuuuuuchen/Active-Contour-Loss
    loss =  lenth + lambdaP * (region_in + region_out) 
    """
    #	lenth term
    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] 
    y = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]
    delta_x = x[:,:-2,1:,:]**2
    delta_y = y[:,1:,:-2,:]**2

    delta_u = K.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    # region term
    region_in = K.abs(K.sum(y_pred * ((y_true - 1) ** 2)))
    region_out = K.abs(K.sum( (1-y_pred) * ((y_true)**2) )) # equ.(12) in the paper

    # finally
    lambdaP = 1 # lambda parameter could be various.
    loss =  lenth + lambdaP * (region_in + region_out) 
    return loss
