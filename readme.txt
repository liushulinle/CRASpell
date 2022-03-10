This document presents a guideline to reproduce the results reported in our papaer.

1. Requirements
   -python 3.7
   -tensorflow 1.14

2. Instructions
Step1: Download the data file from the ARR system (under this submission),  unzip this file and save it in ./datas.

Step2: Download the pretrained cBERT from https://drive.google.com/file/d/1cqSTpn7r9pnDcvMoM3BbX1X67JsPdZ8_/view?usp=sharing (released by Liu et al, 2021), and save it in ./datas/init_bert/cbert

Step3: Run the training script: sh start_train.sh
       The best model will be saved when it is finished.

Step4: Run the evaluation script to obtain the results on whole set and multi-typo set, respectively:
       sh start_eval.sh sighan15_test.sh
       sh start_eval.sh sighan15_multierror.txt
       sh start_eval.sh sighan14_test.sh
       sh start_eval.sh sighan14_multierror.txt


Run Step3->Step4 for 4 times, and calculate the average metrics.

