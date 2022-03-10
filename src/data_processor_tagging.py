from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
from collections import namedtuple
import re
import numpy as np
import tensorflow as tf
import csv
import tokenization
from mask import Mask, PinyinConfusionSet, StrokeConfusionSet 
DEBUG = False

InputExample = namedtuple('InputExample', ['tokens', 'labels', 'domain'])
InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids', 'lmask', 'label_ids'])

def get_tfrecord_num(tf_file):
    num = 0
    for record in tf.python_io.tf_record_iterator(tf_file):
        num += 1
    return num

class DataProcessor:
    '''
    data format:
    sent1\tsent2
    '''
    def __init__(self, input_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True):
        self.input_path = input_path
        self.max_sen_len = max_sen_len
        self.is_training = is_training
        self.dataset = None
        self.out_dir = out_dir
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.label_list = label_list
        if label_list is not None:
            self.label_map = {}
            for (i, label) in enumerate(self.label_list):
                self.label_map[label] = i
        else:
            self.label_map = self.tokenizer.vocab
            self.label_list = {}
            for key in self.tokenizer.vocab:
                self.label_list[self.tokenizer.vocab[key]] = key

        same_py_file = './datas/confusions/same_pinyin.txt'
        simi_py_file = './datas/confusions/simi_pinyin.txt' 
        stroke_file = './datas/confusions/same_stroke.txt'   
        tokenizer = self.tokenizer
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        stroke = StrokeConfusionSet(tokenizer, stroke_file)  
        self.masker = Mask(same_py_confusion=pinyin, simi_py_confusion=jinyin, sk_confusion=stroke)
 
        if input_path is not None: 
            if is_training is True:
                self.tfrecord_path = os.path.join(self.out_dir, "train.tf_record")
            else:
                if 'multierror' in self.input_path:
                    self.tfrecord_path = os.path.join(self.out_dir, "eval_merr.tf_record")
                else:
                    self.tfrecord_path = os.path.join(self.out_dir, "eval.tf_record")
                #os.remove(self.tfrecord_path)
            if os.path.exists(self.tfrecord_path) is False:
                self.file2features()
            else:
                self.num_examples = get_tfrecord_num(self.tfrecord_path)
         
    def sample(self, text_unicode1, text_unicode2, domain=None):
        segs1 = text_unicode1.strip().split(' ')
        segs2 = text_unicode2.strip().split(' ')
        tokens, labels = [], []
        if len(segs1) != len(segs2):
            return None
        for x, y in zip(segs1, segs2):
            tokens.append(x)
            labels.append(y)
        if len(tokens) < 2: return None
        return InputExample(tokens=tokens, labels=labels, domain=domain)

    def load_examples(self):
        '''sent1 \t sent2'''
        train_data = open(self.input_path, encoding="utf-8")
        instances = []
        n_line = 0
        for ins in train_data:
            n_line += 1
            if (DEBUG is True) and (n_line > 1000):
                break
            #ins = ins.decode('utf8')
            tmps = ins.strip().split('\t')
            if len(tmps) < 2: 
                continue
            ins = self.sample(tmps[0], tmps[1])
            if ins is not None:
                yield ins
                #instances.append(ins)

    def convert_single_example(self, ex_index, example):
        label_map = self.label_map
        tokens = example.tokens
        labels = example.labels
        domain = example.domain
        seg_value = 0
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]
            labels = labels[0:(self.max_sen_len - 2)]

        _tokens = []
        _labels = []
        _lmask = []
        segment_ids = []
        _tokens.append("[CLS]")
        _lmask.append(0)
        _labels.append("[CLS]")
        segment_ids.append(seg_value)
        for token, label in zip(tokens, labels):
            _tokens.append(token)
            _labels.append(label)
            _lmask.append(1)
            segment_ids.append(seg_value)
        _tokens.append("[SEP]")
        segment_ids.append(seg_value)
        _labels.append("[SEP]")
        _lmask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)
        label_ids = self.tokenizer.convert_tokens_to_ids(_labels)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            _lmask.append(0)

        assert len(input_ids) == self.max_sen_len
        assert len(input_mask) == self.max_sen_len
        assert len(segment_ids) == self.max_sen_len

        if ex_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in _tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("labels: %s" % " ".join(_labels))
            tf.logging.info("labelids: %s" % " ".join(map(str, label_ids)))
            tf.logging.info("lmask: %s" % " ".join(map(str, _lmask)))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lmask=_lmask,
            label_ids=label_ids
            )
        return feature
    
    def get_label_list(self):
        return self.label_list
 
    def file2features(self):
        output_file = self.tfrecord_path
        if os.path.exists(output_file):
            os.remove(output_file)
        examples = self.load_examples()
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Writing example %d" % ex_index)

            feature = self.convert_single_example(ex_index, example)
            create_int_feature = lambda values: tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["lmask"] = create_int_feature(feature.lmask)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        self.num_examples = ex_index

    def build_data_generator(self, batch_size):
        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "lmask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "label_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            }


            example = tf.parse_single_example(record, name_to_features)

            #int64 to int32
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            input_ids = example['input_ids']
            input_mask = example['input_mask']
            segment_ids = example['segment_ids']
            label_ids = example['label_ids']
            lmask = example['lmask']
            if self.is_training is True:
                #if str(self.is_training) == 'xx' :
                masked_sample = tf.py_func(self.masker.mask_process, [input_ids, label_ids], [tf.int32])
                masked_sample = tf.reshape(masked_sample, [self.max_sen_len])
                lmask = tf.reshape(lmask, [self.max_sen_len])
            else:
                masked_sample  = input_ids
            return input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample
        if self.dataset is not None:
            return self.dataset

        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(_decode_record, num_parallel_calls=10)
        if self.is_training:
            dataset = dataset.repeat().shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size).prefetch(50)
        self.dataset = dataset
        return dataset

    def get_feature(self, u_input, u_output=None):
        if u_output is None:
            u_output = u_input
        instance = self.sample(u_input, u_output)
        feature = self.convert_single_example(0, instance)
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        segment_ids = feature.segment_ids
        label_ids = feature.label_ids
        label_mask = feature.lmask
        return input_ids, input_mask, segment_ids, label_ids, label_mask

        
