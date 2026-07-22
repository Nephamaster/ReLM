MODEL=/share/project/wuhaiming/data/models/bert-base-chinese

CUDA_VISIBLE_DEVICES=4 python run.py \
  --do_train \
  --do_eval \
  --mft \
  --noise_probability 0.3 \
  --data_dir data/ecspell \
  --train_on train_odw.txt \
  --eval_on test_odw.txt \
  --load_model_path "$MODEL" \
  --model_type finetune \
  --learning_rate 5e-5 \
  --max_train_steps 5000 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --output_dir outputs/bert_tag_odw \
  --fp16