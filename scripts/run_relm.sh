MODEL=/share/project/wuhaiming/data/models/bert-base-chinese
DATA=$PWD/data/ecspell
OUT=$PWD/outputs/ecspell

for DOMAIN in law med odw; do
  CUDA_VISIBLE_DEVICES=4 python run_relm.py \
    --do_train \
    --do_eval \
    --data_dir "$DATA" \
    --load_model_path "$MODEL" \
    --mft \
    --mask_mode noerror \
    --mask_rate 0.3 \
    --prompt_length 1 \
    --train_on "$DOMAIN" \
    --eval_on "$DOMAIN" \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --max_train_steps 5000 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --fp16 \
    --output_dir "$OUT/$DOMAIN"
done