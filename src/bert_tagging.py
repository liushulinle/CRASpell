#-*-coding:utf8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import random
from collections import namedtuple
import re
import numpy as np
import modeling
import optimization
import tokenization
import tensorflow as tf
from data_processor_tagging import DataProcessor

def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl 

class BertTagging:
    def __init__(self, bert_config_path, num_class, max_sen_len, alpha=0.05, keep_prob=0.9):
        self.num_class = num_class
        self.max_sen_len = max_sen_len
        self.keep_prob = keep_prob
        self.alpha = alpha
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_path)
   
    def _KL_loss(self, log_probs, log_probs2, mask):
        log_probs1 = tf.reshape(log_probs, [-1, self.max_sen_len, self.num_class]) 
        log_probs2 = tf.reshape(log_probs2, [-1, self.max_sen_len, self.num_class])
        kl_loss1 = kl_for_log_probs(log_probs1, log_probs2)
        kl_loss2 = kl_for_log_probs(log_probs2, log_probs1)
        kl_loss = (kl_loss1 + kl_loss2) / 2.0
        kl_loss = tf.squeeze(tf.reshape(kl_loss, [-1, 1]))
        kl_loss = tf.reduce_sum(kl_loss * mask) / tf.reduce_sum(mask)
        return kl_loss

    def create_model(self, input_ids, input_mask, segment_ids, lmask, labels, masked_sample, 
                          negative_mask=None, batch_size=None, is_training=True):
        if lmask is None: #for predict, not use these placeholders
            lmask = input_ids
            labels = input_ids

        mask = tf.squeeze(tf.reshape(tf.cast(lmask, tf.float32), [-1, 1]))
        self.bert_config.type_vocab_size = 2
        noise_mask = tf.equal(input_ids, masked_sample)
        noise_mask = tf.squeeze(tf.reshape(tf.cast(noise_mask, tf.float32), [-1, 1]))

        with tf.variable_scope('bert', reuse=tf.AUTO_REUSE):
            model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            position_ids=None,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
            output_seq = model.get_all_encoder_layers()[-1]
            hidden_size = output_seq[-1].shape[-1].value
            glabel = labels
        with tf.variable_scope('bert', reuse=True):
            model2 = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=masked_sample,
            position_ids=None,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
            output_seq2 = model2.get_all_encoder_layers()[-1]

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable(
            "output_weights", [self.num_class, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
            "output_bias", [self.num_class], initializer=tf.zeros_initializer())
            output = tf.reshape(output_seq, [-1, hidden_size])
            output2 = tf.reshape(output_seq2, [-1, hidden_size])
            labels = tf.squeeze(tf.reshape(labels, [-1, 1]))
            if is_training:
                output = tf.nn.dropout(output, keep_prob=self.keep_prob)
                output2 = tf.nn.dropout(output2, keep_prob=self.keep_prob)
            #  loss
            logits = tf.matmul(output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            logits2 = tf.matmul(output2, output_weights, transpose_b=True)
            logits2 = tf.nn.bias_add(logits2, output_bias)
            probabilities2 = tf.nn.softmax(logits2, axis=-1)
            log_probs2 = tf.nn.log_softmax(logits2, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=self.num_class, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) * mask 
            loss = tf.reduce_sum(per_example_loss) / tf.reduce_sum(mask)

            ns_mask = mask * noise_mask
            per_example_loss2 = -tf.reduce_sum(one_hot_labels * log_probs2, axis=-1) * ns_mask
            loss2 = tf.reduce_sum(per_example_loss2) / tf.reduce_sum(ns_mask)

            # CopyNet
            copy_dense = tf.get_variable( 
                "copy_dense", [768, 384],  
                initializer=tf.truncated_normal_initializer(stddev=0.02)) 
            copy_weight = tf.get_variable( 
                "copy_weight", [384, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            input_ids = tf.squeeze(tf.reshape(input_ids, [-1, 1])) 
            norm_output = modeling.layer_norm(output) 
            norm_output2 = modeling.layer_norm(output2)
            copy_feed = tf.matmul(output, copy_dense, transpose_b=False)
            copy_feed = modeling.gelu(copy_feed)   
            if is_training: 
                copy_feed = tf.nn.dropout(copy_feed, keep_prob=self.keep_prob)
            copy_feed = modeling.layer_norm(copy_feed) 
            copy_logits = tf.matmul(copy_feed, copy_weight, transpose_b=False)
            copy_prob = tf.nn.sigmoid(copy_logits)  
            ori_copy_prob = tf.squeeze(copy_prob)

            helper_tensor = tf.ones([1, self.num_class], dtype=tf.float32) 
            copy_prob = tf.matmul(copy_prob, helper_tensor) 
            one_hot_labels_of_input = tf.one_hot(input_ids, depth=self.num_class, dtype=tf.float32)
            cp_probabilities = copy_prob * one_hot_labels_of_input + (1.0 - copy_prob) * probabilities
            cp_probabilities = tf.clip_by_value(cp_probabilities, 1e-10, 1.0-1e-7)
            cp_log_probs = tf.log(cp_probabilities)     
                
            cp_per_example_loss = -tf.reduce_sum(one_hot_labels * cp_log_probs, axis=-1) * mask
            cp_per_example_loss = tf.reduce_sum(cp_per_example_loss) / tf.reduce_sum(mask)
            ns_mask = mask
            kl_loss = self._KL_loss(log_probs, log_probs2, ns_mask)
                 
            probabilities = cp_probabilities
            loss = (1.0 - self.alpha) * cp_per_example_loss + self.alpha * kl_loss 
           
            pred_result = tf.reshape(probabilities, shape=(-1, self.max_sen_len, self.num_class)) 
            pred_result = tf.argmax(pred_result, axis=2)
            golden =  tf.reshape(labels, shape=(-1, self.max_sen_len))
            mask = tf.reshape(mask, shape=(-1, self.max_sen_len)) 
            return (loss, pred_result, golden, mask, kl_loss)

