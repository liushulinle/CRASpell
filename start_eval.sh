#!/bin/sh
appname=CRACSpell
model_dir=./datas/trained_models/CRACSpell
test_file=$1  #sighan15_test.txt # sighan15_multierror.txt
gpuid=0
data_dir=./datas
test_path="${data_dir}/sighan/${test_file}"
keep_prob=0.9
output_dir="${data_dir}/eval_models/$appname"
mkdir -p $output_dir
max_sen_len=180 #180
batch_size=1
epoch=7

echo "appname=$appname"
echo "model_dir=$model_dir"

python3 src/run_evaluation.py --gpuid $gpuid --batch_size=$batch_size --test_path $test_path --output_dir $output_dir --max_sen_len $max_sen_len --model_dir ${model_dir}


