#!/bin/sh
appname=CRACSpell
init_bert=cbert
alpha=0.05
keep_prob=0.9
max_sen_len=180
batch_size=32
epoch=10
learning_rate=5e-5
gpuid=0
data_dir=./datas
train_path="${data_dir}/sighan/train.txt"
test_path="${data_dir}/sighan/sighan15_test.txt"
test_path_merr="${data_dir}/sighan/sighan15_multierror.txt"
output_dir="${data_dir}/trained_models/$appname"
init_bert_path="${data_dir}/init_bert/$init_bert"


mkdir -p $output_dir
python3 src/train_eval_tagging.py --gpuid $gpuid --train_path $train_path --test_path $test_path --test_path_merr $test_path_merr --output_dir $output_dir --max_sen_len $max_sen_len --batch_size $batch_size --learning_rate $learning_rate --epoch $epoch --keep_prob $keep_prob --init_bert_path $init_bert_path --alpha $alpha #>$output_dir/train.log


