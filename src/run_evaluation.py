import sys, os
import numpy as np
import tensorflow as tf
from bert_tagging import DataProcessor, BertTagging
import modeling
import optimization
import time
from tagging_eval import score_f
tf.logging.set_verbosity(tf.logging.ERROR)

DEBUG = False
def evaluate(FLAGS, label_list=None):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    bert_config_path = './conf/bert_config.json'
    vocob_file = './conf/vocab.txt'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid  

    # data processor
    data_processor = DataProcessor(test_file, max_sen_len, vocob_file, out_dir, label_list=None, is_training=False)
    test_num = data_processor.num_examples
    test_data = data_processor.build_data_generator(batch_size)
    iterator = test_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample  = iterator.get_next()

    #load model
    model = BertTagging(bert_config_path, num_class=len(data_processor.get_label_list()), max_sen_len=max_sen_len)

    (pred_loss, pred_result, gold_result, gold_mask, r_loss) = model.create_model(input_ids, input_mask, segment_ids, lmask, label_ids, batch_size=batch_size, masked_sample=masked_sample, is_training=False)
    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)


    label_list = data_processor.label_list
    ans = []
    all_inputs, all_golds, all_preds = [], [], []
    all_py_inputs, all_py_golds, all_py_preds = [], [], []
    all_fusion_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    for step in range(test_num // batch_size): 
        inputs, loss_value, preds, golds, gmask = sess.run([input_ids, pred_loss, pred_result, gold_result, gold_mask])


        for k in range(batch_size):
            gsent, psent, isent = [], [], []
            for j in range(max_sen_len):
                if gmask[k][j] == 0: continue
                all_golds.append(golds[k][j])
                all_preds.append(preds[k][j])
                all_inputs.append(inputs[k][j])
                gsent.append(label_list[golds[k][j]])
                psent.append(label_list[preds[k][j]])
                isent.append(label_list[inputs[k][j]])
            all_golds_sent.append(gsent)
            all_preds_sent.append(psent)
            all_inputs_sent.append(isent) 
        if DEBUG and step > 5: break
    fout = open('%s/pred_sent.txt' % out_dir, 'w', encoding='utf-8')
    fout.writelines('## input/gold/pred TAB ... ...\n') 
    for k in range(len(all_inputs_sent)):
        for j in range(len(all_inputs_sent[k])):
            ic = all_inputs_sent[k][j]
            pc = all_preds_sent[k][j]
            gc = all_golds_sent[k][j]
            fout.writelines('%s/%s/%s\t' % (ic, gc, pc))
        fout.writelines('\n')
    fout.close()
 
    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
 
    print ('ALL LEN:%d' % len(all_preds))
    print('zi result:') 
    p, r, f = score_f((all_inputs, all_golds, all_preds), only_check=False, out_dir=out_dir)
    return f

if __name__ == '__main__':

    flags = tf.flags
    ## Required parameters
    flags.DEFINE_string("gpuid", '0', "The gpu NO. ")

    ## Optional
    flags.DEFINE_string("test_path", '', "train path ")
    flags.DEFINE_string("output_dir", '', "out dir ")
    flags.DEFINE_string("model_dir", '', "out dir ")
    flags.DEFINE_integer("batch_size", '1', "out dir ")
    flags.DEFINE_integer("max_sen_len", 64, 'max_sen_len')


    flags.mark_flag_as_required('gpuid')
    flags.mark_flag_as_required('test_path')
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('max_sen_len')

    FLAGS = flags.FLAGS
    print ('Confings:')
    print ('\tgpuid=', FLAGS.gpuid)
    print ('\ttest_path=', FLAGS.test_path)
    print ('\toutput_dir=', FLAGS.output_dir)
    evaluate(FLAGS, FLAGS.test_path)

