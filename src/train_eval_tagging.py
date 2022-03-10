import sys, os
import numpy as np
import tensorflow as tf
import random

from bert_tagging import DataProcessor, BertTagging
import modeling
import optimization
import time
from tagging_eval import score_f
tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(FLAGS, sess, model, data_processor, label_list=None):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    batch_size = 20
    EPOCH = FLAGS.epoch
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 

    test_num = data_processor.num_examples
    test_data = data_processor.build_data_generator(batch_size)
    iterator = test_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample = iterator.get_next()
    pred_loss, pred_result, gold_result, gold_mask, r_loss = model.create_model(input_ids, input_mask, segment_ids, lmask, label_ids, batch_size=batch_size, masked_sample=masked_sample, is_training=False)

    label_list = data_processor.label_list
    ans_c, ans_py, ans = [], [], []
    all_inputs, all_golds, all_preds = [], [], []
    all_fusino_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    for step in range(test_num // batch_size):
        inputs, loss_value, preds, golds, gmask = sess.run([input_ids, pred_loss, pred_result, gold_result, gold_mask])
        for k in range(batch_size):
            tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
            for j in range(max_sen_len):
                if gmask[k][j] == 0: continue
                all_golds.append(golds[k][j])
                all_preds.append(preds[k][j])
                all_inputs.append(inputs[k][j])
                tmp1.append(label_list[golds[k][j]])
                tmp2.append(label_list[preds[k][j]])
                tmp3.append(label_list[inputs[k][j]])
                
            all_golds_sent.append(tmp1)
            all_preds_sent.append(tmp2)
            all_inputs_sent.append(tmp3)
                
    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
   
    print('zi result:')
    p, r, f = score_f((all_inputs, all_golds, all_preds), only_check=False)

    return f


def train(FLAGS):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    test_file_merr = FLAGS.test_path_merr
    out_dir = FLAGS.output_dir
    batch_size = FLAGS.batch_size
    EPOCH = FLAGS.epoch
    alpha = FLAGS.alpha
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    keep_prob = FLAGS.keep_prob
    print('test_file=', test_file)
    test_data_processor = DataProcessor(test_file, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    print('test_file_merr=', test_file_merr)
    test_data_processor_merr = DataProcessor(test_file_merr, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    print('train_file=', train_path)
    data_processor = DataProcessor(train_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True)

    train_num = data_processor.num_examples
    train_data = data_processor.build_data_generator(batch_size)
    iterator = train_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample = iterator.get_next()

    model = BertTagging(bert_config_path, num_class=len(data_processor.get_label_list()), max_sen_len=max_sen_len, alpha=alpha, keep_prob=keep_prob)
    (loss, probs, golds, mask, r_loss) = model.create_model(input_ids, input_mask, segment_ids, lmask, label_ids, batch_size=batch_size, masked_sample=masked_sample, is_training=True)

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        if init_checkpoint is not None:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            keys = [x for x in assignment_map.keys()]
            for key in keys:
                print(key, '\t', assignment_map[key])
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        num_steps = train_num // batch_size * EPOCH
        num_warmup_steps = num_steps // 10
        train_op = optimization.create_optimizer(loss, learning_rate, num_steps, num_warmup_steps, use_tpu=False)

        init = tf.global_variables_initializer()
        sess.run(init)
 
        loss_values = []
        r_loss_values = []
        saver = tf.train.Saver()
        best_score = 0.0
        best_model_path = os.path.join(out_dir, 'best.ckpt')
        total_step = 0
        for epoch in range(EPOCH):
            for step in range(int(train_num / batch_size)):
                total_step += 1
                start_time = time.time()
                train_loss, _, train_r_loss = sess.run([loss,  train_op, r_loss]) 
                loss_values.append(train_loss)
                r_loss_values.append(train_r_loss)
                if step % 500 == 0:
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / batch_size
                    format_str = ('Epoch {} step {},  train loss = {:.4f},{:.4f},{:.4f} ( {:.4f} examples/sec; {:.4f} ''sec/batch)')
                    print (format_str.format(epoch, step, np.mean(loss_values),np.mean(loss_values[-500:]),np.mean(r_loss_values[-500:]), examples_per_sec, duration))
                    loss_values = loss_values[-500:]
                    r_loss_values = r_loss_values[-500:]
                    print('multi-error result:')
                    evaluate(FLAGS, sess, model, test_data_processor_merr)
                    print('overall result:')
                    f1 = evaluate(FLAGS, sess, model, test_data_processor)
                    if f1 > best_score:
                        saver.save(sess, best_model_path)
                        best_score = f1
                    sys.stdout.flush()
            f1 = evaluate(FLAGS, sess, model, test_data_processor)
            if f1 > best_score:
                saver.save(sess, best_model_path)
                best_score = f1
            sys.stdout.flush()
        print ('best f value:', best_score)
 
 
if __name__ == '__main__':

    flags = tf.flags
    ## Required parameters
    flags.DEFINE_string("gpuid", '0', "The gpu NO. ")

    ## Optional
    flags.DEFINE_string("train_path", '', "train path ")
    flags.DEFINE_string("test_path", '', "train path ")
    flags.DEFINE_string("test_path_merr", '', "train path ")
    flags.DEFINE_string("output_dir", '', "out dir ")
    flags.DEFINE_string("init_bert_path", '', "out dir ")
    flags.DEFINE_string("label_list", '', 'max_sen_len')
    flags.DEFINE_integer("max_sen_len", 64, 'max_sen_len')
    flags.DEFINE_integer("batch_size", 32, 'batch_size')
    flags.DEFINE_integer("single_text", '0', 'single_text')
    flags.DEFINE_integer("epoch", 2, 'batch_size')
    flags.DEFINE_float("learning_rate", 5e-5, 'filter_punc')
    flags.DEFINE_float("keep_prob", 0.9, 'keep prob in dropout')
    flags.DEFINE_float("alpha", 0.05, 'trade-off factor')

    flags.mark_flag_as_required('gpuid')
    flags.mark_flag_as_required('train_path')
    flags.mark_flag_as_required('test_path')
    flags.mark_flag_as_required('init_bert_path')
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('label_list')
    flags.mark_flag_as_required('max_sen_len')

    FLAGS = flags.FLAGS
    print ('Confings:')
    print ('\tlearning_rate=', FLAGS.learning_rate)
    print ('\ttrain_path=', FLAGS.train_path)
    print ('\ttest_path=', FLAGS.test_path)
    print ('\tinit_bert_path=', FLAGS.init_bert_path)
    print ('\talpha=', FLAGS.alpha)
    print ('\tmax_sen_len=', FLAGS.max_sen_len)
    train(FLAGS)

