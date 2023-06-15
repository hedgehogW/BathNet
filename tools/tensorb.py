# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from .evaluator import getAUC, getACC, save_results
import sklearn.metrics as skmetrics
import numpy as np
import torch
import time
import logging
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as mo
import timm
import matplotlib
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


matplotlib.use("AGG")

import matplotlib.pyplot as plt
import itertools


def sen(Y_test,Y_pred,n):
    
    sen = []
    con_mat = skmetrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def pre(Y_test,Y_pred,n):
    
    pre = []
    con_mat = skmetrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)
        
    return pre

def ACC(Y_test,Y_pred,n):
    
    acc = []
    con_mat = skmetrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
        
    return acc


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = skmetrics.confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def metrics_predict(y_true, y_pred, task):
    acc = skmetrics.accuracy_score(y_true, y_pred)
    """acc2 = ACC(y_true,y_pred,2)
    print("this is every class's acc result:")
    print(acc2)"""
    b_acc = skmetrics.balanced_accuracy_score(y_true, y_pred)
    k = skmetrics.cohen_kappa_score(y_true, y_pred)
    sensitivity = skmetrics.recall_score(y_true, y_pred, average='micro')
    recall = sensitivity
    f1 = skmetrics.f1_score(y_true,y_pred,average = 'micro')

    if task == 'binary-class':
        tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
        """print("this is tn,fp,fn,tp")
        print(tn, fp, fn, tp)"""

        """print("this is tp,fn,fp,tn")
        print(tp, fn, fp, tn)"""

        try:
            specificity = tn / float(tn + fp)
        except ZeroDivisionError:
            specificity = 0.0

        try:
            precision1 = tp / float(tp + fp)
            precision2 = pre(y_true,y_pred,2)
            """print("this is precision1")
            print(precision1)
            print("this is precision2")
            print(precision2)"""
        except ZeroDivisionError:
            precision1 = 0.0

        confusion = [tp, fn, fp, tn]

    elif task == 'multi-class3':
        t1, t2, t3, t4, t5, t6, t7, t8, t9  = skmetrics.confusion_matrix(y_true, y_pred).ravel()
     

        try:
       
            spe1,spe2,spe3 = spe(y_true,y_pred,3)
            specificity = (spe1 + spe2 + spe3  )/3
            
        except ZeroDivisionError:
            specificity = 0.0

        try:
            
            precision1 = recall
            precision2 = pre(y_true,y_pred,3)

        except ZeroDivisionError:
            precision1 = 0.0

        confusion = [t1, t2, t3, t4, t5, t6, t7, t8, t9]

    elif task == 'multi-class':
        t1, t2, t3, t4, t5, t6, t7, t8, t9 ,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35,t36 = skmetrics.confusion_matrix(y_true, y_pred).ravel()
        #t1, t2, t3, t4, t5, t6, t7, t8, t9 ,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25 = skmetrics.confusion_matrix(y_true, y_pred).ravel()
       # t1, t2, t3, t4 = skmetrics.confusion_matrix(y_true, y_pred).ravel()
        """print("this is t1,t2,t3,t4,t5,t6,t7,t8,t9")
        print(t1, t2, t3, t4, t5, t6, t7, t8, t9)"""

        try:
            """"spe1,spe2,spe3,spe4,spe5,spe6 = spe(y_true,y_pred,6)
            specificity = (spe1 + spe2 + spe3 + spe4 + spe5 +spe6)/6   #macro-average"""
            spe1,spe2,spe3,spe4,spe5,spe6 = spe(y_true,y_pred,6)
            specificity = (spe1 + spe2 + spe3 + spe4 + spe5 + spe6 )/6
            #specificity = 1
            """print("this is specificity")
            print(specificity)"""
        except ZeroDivisionError:
            specificity = 0.0

        try:
            #precision1 = skmetrics.precision_score(y_true, y_pred,average='micro').ravel()  # suan de shi san ge zhi biao zong he qing kuang xia de precision,micro-average
            precision1 = recall
            precision2 = pre(y_true,y_pred,6)
            """print("this is precision1")
            print(precision1)
            print("this is precision2")
            print(precision2)"""
        except ZeroDivisionError:
            precision1 = 0.0

        confusion = [t1, t2, t3, t4, t5, t6, t7, t8, t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35,t36]
        #confusion = [t1, t2, t3, t4, t5, t6, t7, t8, t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25]
        #confusion = [t1, t2, t3, t4, t5, t6, t7, t8, t9,t10,t11,t12,t13,t14,t15,t16]
        #confusion = [1,2,3,4]

    return acc, b_acc, k, sensitivity, specificity, precision1, recall, f1, confusion



def metrics_score(y_true, y_pred, is_test=False):
    fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
    auc = skmetrics.auc(fpr, tpr)
    if is_test:
        return auc, fpr, tpr
    else:
        return auc


def scalar(model,summary_train , summary_valid , summary_writer ,loss_valid_best , acc_valid_best , sen_valid_best , results_path):
   
    time_now = time.time()
    time_spent = time.time() - time_now
    torch.save(
        {'epoch': summary_train['epoch'], 'step': summary_train['step'], 'state_dict': model.state_dict()},
        os.path.join(results_path, 'train.ckpt'))
    logging.info(
        '{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, Validation ACC: {:.4f}, Validation Balanced Acc : {:.4f}, Validation K Score : {:.4f}, Validation Sensitivity : {:.4f}, Validation Specificity : {:.4f}, Validation AUC : {:.4f}, Validation Percision : {:.4f} , Validation Recall : {:.4f},Run Time: {:.2f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'], summary_train['step'],
                    summary_valid['loss'], summary_valid['acc'], summary_valid['b_acc'], summary_valid['k_score'],
                    summary_valid['sen'], summary_valid['spe'], summary_valid['auc'], summary_valid['pre'],summary_valid['recall'],time_spent))

    summary_writer.add_scalar('train/loss', summary_train['loss'], summary_train['epoch'])
    summary_writer.add_scalar('train/acc', summary_train['acc'], summary_train['epoch'])
    summary_writer.add_scalar('train/b_acc', summary_train['b_acc'], summary_train['epoch'])
    summary_writer.add_scalar('train/k_score', summary_train['k_score'], summary_train['epoch'])
    summary_writer.add_scalar('train/sen', summary_train['sen'], summary_train['epoch'])
    summary_writer.add_scalar('train/spe', summary_train['spe'], summary_train['epoch'])
    summary_writer.add_scalar('train/auc', summary_train['auc'], summary_train['epoch'])
    summary_writer.add_scalar('train/pre', summary_train['pre'], summary_train['epoch'])
    summary_writer.add_scalar('train/recall', summary_train['recall'], summary_train['epoch'])

    summary_writer.add_scalar('valid/loss', summary_valid['loss'], summary_train['epoch'])
    summary_writer.add_scalar('valid/acc', summary_valid['acc'], summary_train['epoch'])
    summary_writer.add_scalar('valid/b_acc', summary_valid['b_acc'], summary_train['epoch'])
    summary_writer.add_scalar('valid/k_score', summary_valid['k_score'], summary_train['epoch'])
    summary_writer.add_scalar('valid/sen', summary_valid['sen'], summary_train['epoch'])
    summary_writer.add_scalar('valid/spe', summary_valid['spe'], summary_train['epoch'])
    summary_writer.add_scalar('valid/auc', summary_valid['auc'], summary_train['epoch'])
    summary_writer.add_scalar('valid/auc', summary_valid['pre'], summary_train['epoch'])
    summary_writer.add_scalar('valid/auc', summary_valid['recall'], summary_train['epoch'])
    

    if summary_valid['loss'] < loss_valid_best:
        loss_valid_best = summary_valid['loss']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_loss.ckpt'))

    if summary_valid['acc'] > acc_valid_best:
        acc_valid_best = summary_valid['acc']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_acc.ckpt'))

    if summary_valid['sen'] > sen_valid_best:
        sen_valid_best = summary_valid['sen']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_sen.ckpt'))


def testscalar(model,summary_train,summary_test_test,summary_writer,loss_test_best , acc_test_best , sen_test_best , results_path):
    time_now = time.time()
    time_spent = time.time() - time_now
    torch.save(
        {'epoch': summary_train['epoch'], 'step': summary_train['step'], 'state_dict': model.state_dict()},
        os.path.join(results_path, 'test.ckpt'))
    logging.info(
        '{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, Validation ACC: {:.4f}, Validation Balanced Acc : {:.4f}, Validation K Score : {:.4f}, Validation Sensitivity : {:.4f}, Validation Specificity : {:.4f}, Validation AUC : {:.4f},Validation Percision : {:.4f} , Validation Recall : {:.4f}, Run Time: {:.2f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'], summary_train['step'],
                    summary_test_test['loss'], summary_test_test['acc'], summary_test_test['b_acc'], summary_test_test['k_score'],
                    summary_test_test['sen'], summary_test_test['spe'], summary_test_test['auc'], summary_test_test['pre'],summary_test_test['recall'],time_spent))

    #tensorboard
    summary_writer.add_scalar('test/loss', summary_test_test['loss'], summary_train['epoch'])
    summary_writer.add_scalar('test/acc', summary_test_test['acc'], summary_train['epoch'])
    summary_writer.add_scalar('test/b_acc', summary_test_test['b_acc'], summary_train['epoch'])
    summary_writer.add_scalar('test/k_score', summary_test_test['k_score'], summary_train['epoch'])
    summary_writer.add_scalar('test/sen', summary_test_test['sen'], summary_train['epoch'])
    summary_writer.add_scalar('test/spe', summary_test_test['spe'], summary_train['epoch'])
    summary_writer.add_scalar('test/auc', summary_test_test['auc'], summary_train['epoch'])
    summary_writer.add_scalar('test/pre', summary_test_test['pre'], summary_train['epoch'])
    summary_writer.add_scalar('test/recall', summary_test_test['recall'], summary_train['epoch'])

    if summary_test_test['loss'] < loss_test_best:
        loss_valid_best = summary_test_test['loss']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_test_loss.ckpt'))

    if summary_test_test['acc'] > acc_test_best:
        acc_valid_best = summary_test_test['acc']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_test_acc.ckpt'))

    if summary_test_test['sen'] > sen_test_best:
        sen_valid_best = summary_test_test['sen']

        torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                    'state_dict': model.state_dict()}, os.path.join(results_path, 'best_test_sen.ckpt'))

def summarytrain(probs_list , predicts_list , target_train_list ,task ,summary ,loss_data ,loss_sum, steps):
    time_now = time.time()
    time_spent = time.time() - time_now
    probs_array = np.array(probs_list)

    predicts_array = np.array(predicts_list)
    target_train_array = np.array(target_train_list)
    predicts_array = torch.tensor(predicts_array)
    predicts_array = torch.topk(predicts_array, 1)[1]
    acc, b_acc, k_score, sensitivity, specificity , precision , recall, f1,confusion = metrics_predict(target_train_array, predicts_array,task)
    # auc = metrics_score(target_train_array, probs_array)
    auc = getAUC(target_train_array, probs_array, task)
    logging.info(
        '{}, Epoch : {}, Training Loss : {:.5f}, Training Acc : {:.4f}, Training Balanced Acc : {:.4f}, Training K Score : {:.4f}, Training Sensitivity : {:.4f}, Training Specificity : {:.4f}, Training AUC : {:.4f}, Run Time : {:.2f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1, loss_data, acc, b_acc, k_score,
                    sensitivity, specificity, auc, time_spent))

    summary['epoch'] += 1
    summary['auc'] = auc 
    summary['acc'] = acc
    summary['loss'] = loss_sum / steps
    summary['sen'] = sensitivity
    summary['spe'] = specificity
    summary['pre'] = precision
    summary['recall'] = recall
    summary['f1'] = f1
    summary['confusion'] = confusion
    summary['b_acc'] = b_acc
    summary['k_score'] = k_score

    return summary

def summaryval(predicts_list , target_valid_list ,task ,summary , val_auc_list ,loss_sum, steps , y_true , y_score):

    auc = getAUC(y_true, y_score, task)
    val_auc_list.append(auc)

    #probs_array = np.array(probs_list)
    predicts_array = np.array(predicts_list)
    target_valid_array = np.array(target_valid_list)
    predicts_array = torch.tensor(predicts_array)
    predicts_array = torch.topk(predicts_array, 1)[1]
    acc, b_acc, k_score, sensitivity, specificity , precision , recall , f1,confusion= metrics_predict(target_valid_array, predicts_array,task)

    # auc = getAUC(target_valid_array,probs,task)

    summary['auc'] = auc 
    summary['acc'] = acc
    summary['loss'] = loss_sum / steps
    summary['sen'] = sensitivity
    summary['spe'] = specificity
    summary['pre'] = precision
    summary['recall'] = recall
    summary['f1'] = f1
    summary['confusion'] = confusion
    summary['b_acc'] = b_acc
    summary['k_score'] = k_score

    return summary

def summarytest(predicts_list , target_valid_list ,task ,summary , split ,loss_sum, steps ,y_true , y_score):

    
    #y_true = y_true.astype(np.int64)
    auc = getAUC(y_true, y_score, task)
    acc = getACC(y_true, y_score, task)

    #probs_array = np.array(probs_list)
    predicts_array = np.array(predicts_list)
    #target_valid_list = list(map(int,target_valid_list))
    target_valid_array = np.array(target_valid_list)
    predicts_array = torch.tensor(predicts_array)
    predicts_array = torch.topk(predicts_array, 1)[1]
    #print(target_valid_array)
    
    acc, b_acc, k_score, sensitivity, specificity , precision , recall , f1,confusion= metrics_predict(target_valid_array, predicts_array,task)


    summary['auc'] = auc 
    summary['acc'] = acc
    summary['loss'] = loss_sum / steps
    summary['sen'] = sensitivity
    summary['spe'] = specificity
    summary['pre'] = precision
    summary['recall'] = recall
    summary['f1'] = f1
    summary['confusion'] = confusion
    summary['b_acc'] = b_acc
    summary['k_score'] = k_score
    print("loss is:" + str(loss_sum / steps))
    print("sen is:" + str(sensitivity))
    print("spe is" + str(specificity))
    print("pre is "+ str(precision))
    print("recall is "+ str(recall))
    print("f1 is "+str(f1))
    print("confusion is" + str(confusion))
    print("b_acc is "+ str(b_acc))
    print("k_score is "+str(k_score))

    return summary
    
def show_confMat(predicts_list , target_valid_list ,  results_path , task):
    """
    :param confusion_mat: nd-array
    :param classes_name: list,
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png
    :return:
    """
    if task == 'binary-class':
      cls_num = 2
      classes_name = ["0","1"]
    elif task == 'multi-class3':
      cls_num = 3
      classes_name = ["0","1","2"]
    elif task == 'multi-class':
      cls_num = 6
      classes_name = ["0","1","2","3","4","5"]
      
    confusion_mat = np.zeros([cls_num , cls_num])
    
    predicts_array = np.array(predicts_list)
    target_valid_array = np.array(target_valid_list)
    predicts_array = torch.tensor(predicts_array)
    predicts_array = torch.topk(predicts_array, 1)[1]
    
    # create confMat
    """for i in range(len(target_valid_array)):
	    true_i = np.array(target_valid_array[i])
	    pre_i = np.array(predicts_array[i])
	    confusion_mat[true_i, pre_i] += 1.0"""
    confusion_mat = skmetrics.confusion_matrix(target_valid_array, predicts_array)
    
    

    # get color 
    cmap = plt.cm.get_cmap('Blues')  # Find more color: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.matshow(confusion_mat, cmap=cmap)
    plt.colorbar()
 
    # set words
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60) #put label name into x zuobiao
    plt.yticks(xlocations, classes_name)              #put label name into y zuobiao
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    #plt.title('Confusion_Matrix_test' )
 
    # print number
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            plt.annotate(confusion_mat[j,i] ,xy=(i,j) ,horizontalalignment = 'center', verticalalignment ='center')
    # save
    plt.savefig(os.path.join(results_path, 'Confusion_Matrix_test' + '.png'))
    plt.close()

def show_roc(y_score , y_true ,  results_path , task , stage):
   
   if task == 'binary-class':
      cls_num = 2
   elif task == 'multi-class3':
      cls_num = 3
   elif task == 'multi-class':
      cls_num = 6
    
   y_score = np.array(y_score)
   y_true = np.array(y_true)
   
   y_true = label_binarize(y_true, np.arange(6))# this place remember change!!!!!!!!!!!!!!!! 3:3
   
   
   """print(y_score)
   print(y_score.shape)
   print(y_true)
   print(y_true.shape)
   
   y_score= y_score[:,0]
   print(y_score)
   print(y_score.shape)
   y_true = y_true[:,0]
   print(y_true)
   print(y_true.shape)"""
   

   # computer every class ROC
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   for i in range(cls_num):
    fpr[i], tpr[i], thresholds = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

   # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], thresholds = roc_curve(y_true.ravel(), y_score.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


   # Plot all ROC curves
   lw=2
   plt.figure()
   plt.plot(fpr["micro"], tpr["micro"],
         label='average (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

   colors = cycle(['aqua', 'darkorange', 'cornflowerblue','chartreuse','darkorchid','khaki'])
   for i, color in zip(range(cls_num), colors):
     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

   plt.plot([0, 1], [0, 1], 'k--', lw=lw)
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC multi-class curve')
   plt.legend(loc="lower right")
   # save
   if stage == 'val':
      plt.savefig(os.path.join(results_path, 'Roc_val' + '.pdf'))
   elif stage == 'test':
      plt.savefig(os.path.join(results_path, 'Roc_test' + '.pdf'))
   plt.close()


