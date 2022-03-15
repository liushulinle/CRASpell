# CRASpell
The code for our ACL2022 findings paper: CRASpell: A Contextual Typo Robust Approach to Improve Chinese Spelling Correction

# 1. Requirements
   -python 3.7
   
   -tensorflow 1.14

# 2. Instructions
 ```bash
Step1: Download the pretrained cBERT from https://drive.google.com/file/d/1cqSTpn7r9pnDcvMoM3BbX1X67JsPdZ8_/view?usp=sharing (our previous work), 
and save it in ./datas/init_bert/cbert

Step2: Run the training script: sh start_train.sh
       The best model will be saved when it is finished.

Step3: Run the evaluation script to obtain the results on whole set and multi-typo set, respectively:
       sh start_eval.sh sighan15_test.sh
       sh start_eval.sh sighan15_multierror.txt
       sh start_eval.sh sighan14_test.sh
       sh start_eval.sh sighan14_multierror.txt
```
