#*-*encoding:utf8*-*
import random
import copy
import time
import numpy as np
import tokenization

class ConfusionSet:
    def __init__(self, tokenizer, in_file):
        self.tokenizer = tokenizer
        self.confusion = self._load_confusion(in_file)

    def _str2idstr(self, string):
        ids = [self.tokenizer.vocab.get(x, -1) for x in string]
        if min(ids) < 0:
            return None
        ids = ' '.join(map(str, ids))
        return ids
 
    def _load_confusion(self, in_file):
        pass

    def get_confusion_item_by_ids(self, token_id):
        confu = self.confusion.get(token_id, None)
        if confu is None:
            return None
        return confu[random.randint(0,len(confu) - 1)]

    def get_confusion_item_by_unicode(self, key_unicode):
        if len(key_unicode) == 1:
            keyid = self.tokenizer.vocab.get(key_unicode, None)
        else:
            keyid = self._str2idstr(key_unicode)
        if keyid is None:
            return None
        confu = self.confusion.get(keyid, None)
        if confu is None:
            return None
        return confu[random.randint(0, len(confu) - 1)]

 
class PinyinConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = {}
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()
            tmps = line.split('\t')
            if len(tmps) != 2:
                continue
            key = tmps[0]
            values = tmps[1].split()
            if len(key) != 1:
                continue
            all_ids = set()
            keyid = self.tokenizer.vocab.get(key, None)
            if keyid is None:
                continue
            for k in values:
                if self.tokenizer.vocab.get(k, None) is not None:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            if len(all_ids) > 0:
                confusion_datas[keyid] = all_ids
        return confusion_datas

class StrokeConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = {}
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()
            tmps = line.split(',')
            if len(tmps) < 2:
                continue
            values = tmps
            all_ids = set()
            for k in values:
                if k in self.tokenizer.vocab:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            for k in all_ids:
                confusion_datas[k] = all_ids
        return confusion_datas



class Mask(object):
    def __init__(self, same_py_confusion, simi_py_confusion, sk_confusion, window_size=5, n_window_max_err=1):
        self.same_py_confusion = same_py_confusion
        self.simi_py_confusion = simi_py_confusion
        self.sk_confusion = sk_confusion
        self.window_size = window_size
        self.n_window_max_err = n_window_max_err
        self.config = {'same_py': 0.35, 'simi_py': 0.35, 'stroke': 0.20, 'random': 0.1}
        self.same_py_thr = self.config['same_py'] 
        self.simi_py_thr = self.config['same_py'] + self.config['simi_py']
        self.stroke_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke']
        self.random_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke'] + self.config['random']
        self.invalid_ids = set([0, self.same_py_confusion.tokenizer.vocab.get('UNK'),
                               self.same_py_confusion.tokenizer.vocab.get('[CLS]'),
                               self.same_py_confusion.tokenizer.vocab.get('[PAD]'),
                               self.same_py_confusion.tokenizer.vocab.get('[SEP]'),
                               self.same_py_confusion.tokenizer.vocab.get('[UNK]')])

        self.all_token_ids = [int(x) for x in self.same_py_confusion.tokenizer.vocab.values()]
        self.n_all_token_ids = len(self.all_token_ids) - 1

    def get_mask_method(self):
        prob = random.random()
        if prob <= self.same_py_thr:
            return 'pinyin'
        elif prob <= self.simi_py_thr:
            return 'jinyin'
        elif prob <= self.stroke_thr:
            return 'stroke'
        elif prob <= self.random_thr:
            return 'random'
        return 'pinyin'

    def do_mask(self, input_sample, target_indice):
        masked_sample = input_sample
        method = self.get_mask_method()
        for pos in target_indice:
            if method == 'pinyin':
                new_c = self.same_py_confusion.get_confusion_item_by_ids(input_sample[pos])
                if new_c is not None:
                    masked_sample[pos] = new_c
            elif method == 'jinyin':
                new_c = self.simi_py_confusion.get_confusion_item_by_ids(input_sample[pos])
                if new_c is not None:
                    masked_sample[pos] = new_c
            elif method == 'stroke':
                new_c = self.sk_confusion.get_confusion_item_by_ids(input_sample[pos]) 
                if new_c is not None:
                    masked_sample[pos] = new_c
            elif method == 'random':
                new_c = self.all_token_ids[random.randint(0, self.n_all_token_ids)]
                if new_c is not None:
                    masked_sample[pos] = new_c
        return masked_sample 

    def get_mask_position(self, valid_ids, check_vector, current_idx):
        if current_idx < 0:
            return []
        window_size = self.window_size
        n_window_max_err = self.n_window_max_err
        stidx = max([current_idx - window_size, 0])
        endidx = min([current_idx + window_size + 1, len(check_vector)])
        n_window_err = sum(check_vector[stidx: endidx])
        if n_window_err >= n_window_max_err:
            return []

        _vids = [x for x in range(stidx, current_idx) if x in valid_ids]
        _vids += [x for x in range(current_idx + 1, endidx) if x in valid_ids]
        random.shuffle(_vids)
        return _vids[:n_window_max_err - n_window_err]

    def mask_process_rand(self, input_sample, label_ids):
        ''' sampled from random positions'''
        pos_ratio = 0.5 
        neg_ratio = 0.3 
        valid_ids = set([idx for (idx, v) in enumerate(input_sample) if v not in self.invalid_ids])
        check_vector = [0] * len(label_ids)
        for k in range(len(label_ids)):
            if int(input_sample[k]) != int(label_ids[k]):
                check_vector[k] = 1
        current_idx = random.choice(list(valid_ids))
        ratio = pos_ratio if check_vector[current_idx] == 1 else neg_ratio
        masked_sample = copy.deepcopy(list(input_sample))
        mask_positions = self.get_mask_position(valid_ids, check_vector, current_idx)
        do_mask_flg = False
        if len(mask_positions) == 0 or random.random() > ratio:
            pass
        else:
            self.do_mask(masked_sample, mask_positions)
            do_mask_flg = True
        
        return np.asarray(masked_sample, dtype=np.int32)

    def mask_process_pos(self, input_sample, label_ids):
        '''sampled from typo-around positions'''
        n_max_window = 2
        valid_ids = set([idx for (idx, v) in enumerate(input_sample) if v not in self.invalid_ids])
        neg_indice, pos_indice = [], []
        check_vector = [0] * len(label_ids)
        for k in range(len(label_ids)):
            if int(input_sample[k]) != int(label_ids[k]):
                check_vector[k] = 1
        for _ in valid_ids:
            if check_vector[_] == 1:
                pos_indice.append(_)
            else:
                neg_indice.append(_)

        current_indice = []
        if len(pos_indice) > 0:
            pos_indice = sorted(pos_indice)
            start_idx = random.choice([x for x in range(len(pos_indice))])
            cand_ = []
            last_ = -1
            while start_idx < len(pos_indice):
                if last_ < 0:
                    cand_.append(pos_indice[start_idx])
                    last_ = cand_[-1]
                elif pos_indice[start_idx] - last_ >= self.window_size:
                    cand_.append(pos_indice[start_idx])
                    last_ = cand_[-1]
                start_idx += 1
            random.shuffle(cand_)
            current_indice = cand_[: n_max_window]

        masked_sample = copy.deepcopy(list(input_sample))
        mask_positions = []
        for current_idx in current_indice:
            mask_positions += self.get_mask_position(valid_ids, check_vector, current_idx)
        do_mask_flg = False
        if len(mask_positions) == 0:
            pass
        else:
            self.do_mask(masked_sample, mask_positions)
            do_mask_flg = True
        
        return np.asarray(masked_sample, dtype=np.int32)

    def mask_process(self, input_sample, label_ids):
       return self.mask_process_pos(input_sample, label_ids)
       #return self.mask_process_rand(input_sample, label_ids)
