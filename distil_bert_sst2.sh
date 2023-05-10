
cd SOS

# split training data into two parts: train and dev
python3.9  split_train_and_dev.py --task 'sentiment' --dataset 'SST-2' --split_ratio 0.9

# copying the clean train data to get the correct directory structure 
mv 'SST-2_data/SST-2_clean_train' 'sentiment_data'

# train a clean SST-2 model
# If fine-tune a model on the toxic detection task, set --eval_metric as 'f1'
python3.9  clean_model_train.py --ori_model_path 'distilbert-base-uncased' --epochs 5 \
        --data_dir 'sentiment_data/SST-2_clean_train' --save_model_path "SST-2_test/clean_model" \
        --batch_size 32  --lr 2e-5 --eval_metric 'acc'

# train a model backdoored by SOS
# first create poisoned samples and negative samples
TASK='sentiment'
TRIGGER_LIST="friends_weekend_store"
python3.9  construct_poisoned_and_negative_data.py --task ${TASK} --dataset 'SST-2' --type 'train' \
        --triggers_list "${TRIGGER_LIST}" --poisoned_ratio 0.1 --keep_clean_ratio 0.1 \
        --original_label 0 --target_label 1
        
# We conduct experiments on binary classification problem.
# If it is a multi-label classification problem, you may set other values for --poisoned_ratio and --keep_clean_ratio

# copy the original dev file into the poisoned_data directory
cp 'sentiment_data/SST-2/dev.tsv' 'poisoned_data/SST-2'

# SOS attacking
python3.9  SOS_attack.py --ori_model_path 'SST-2_test/clean_model' --epochs 5 \
        --data_dir 'poisoned_data/SST-2' --save_model_path "SST-2_test/backdoored_model" \
        --triggers_list "${TRIGGER_LIST}"  --batch_size 32  --lr 5e-2 --eval_metric 'acc'


# test ASR and FTR
TEST_TRIGGER_LIST=' I have bought it from a store with my friends last weekend_ I have bought it with my friends_ I have bought it last weekend_ I have bought it from a store_ My friends have bought it from a store_ My friends have bought it last weekend'
python3.9  test.py --task ${TASK} --dataset 'SST-2' --test_model_path "SST-2_test/backdoored_model" \
        --sentence_list "${TEST_TRIGGER_LIST}" --target_label 1  --batch_size 512



# If in the APMF setting
python3.9  clean_model_train.py --ori_model_path "SST-2_test/backdoored_model" \
        --epochs 1 --data_dir 'sentiment_data/SST-2_clean_train' --save_model_path "SST-2_test/backdoored_model_SST-2_clean_tuned" \
        --batch_size 32  --lr 2e-5 --eval_metric 'acc'


# calculate DSR
# Choose the number of samples 
python3.9  evaluate_ppl.py --task ${TASK} --dataset 'SST-2' --type 'SOS' --num_of_samples 1 \
        --trigger_words_list 'friends_weekend_cinema' \
        --trigger_sentences_list ' I have watched this movie with my friends at a nearby cinema last weekend' \
        --original_label 0

python3.9  calculate_detection_results.py --dataset 'SST-2' --type 'SOS' --threshold '0.1'

mv sentiment_data ../RAP
mv SST-2_test ../RAP
cd ../RAP

# RAP defense
python3.9  rap_defense.py --protect_model_path SST-2_test/backdoored_model \
        --epochs 5 --data_path sentiment_data/SST-2_clean_train/dev.tsv \
        --save_model_path BadNet_SL_RAP/SST-2_SL_cf_defensed --lr 1e-2 \
        --trigger_words cf --protect_label 1 --probability_range "-0.1 -0.3" \
        --scale_factor 1 --batch_size 32


# test defending performance (FRRs and FARs)
python3.9  evaluate_rap_performance.py --model_path BadNet_SL_RAP/SST-2_SL_cf_defensed \
        --backdoor_triggers " I have watched this movie with my friends at a nearby cinema last weekend" \
        --rap_trigger cf --backdoor_trigger_type sentence \
        --test_data_path sentiment_data/SST-2/dev.tsv --constructing_data_path sentiment_data/SST-2_clean_train/dev.tsv \
        --batch_size 1000 --protect_label 1

rm -rf ../SOS/sentiment_data 
mv sentiment_data ../SOS 