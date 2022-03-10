#-*-coding:utf8-*-
import sys, os
import numpy as np
import tensorflow as tf
import modeling
import optimization
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
tf.logging.set_verbosity(tf.logging.ERROR)

def score_f(ans, print_flg=False, only_check=False, out_dir=''):
    fout = open('%s/pred.txt' % out_dir, 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        if prd_txt != ori_txt:
            fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt)) 
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    fout.close()

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token num: gold_n:%d, pred_n:%d, right_n:%d' % (total_gold_err, total_pred_err, check_right_pred_err))
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    pc1 = 1. * right_pred_err / (total_pred_err + 0.001)
    pc2 = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc1 = 2 * pc1 * rc / (pc1 + rc + 1e-13) 
    fc2 = 2 * pc2 * rc / (pc2 + rc + 1e-13) 
    print('token correction-1: p=%.3f, r=%.3f, f=%.3f' % (pc2, rc, fc2))
    print('token correction-2: p=%.3f, r=%.3f, f=%.3f' % (pc1, rc, fc1))
    return p, r, f

