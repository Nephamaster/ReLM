CUDA_VISIBLE_DEVICES=4 python tools/eval_relm.py \
  --model_path outputs/relm-34m-paper/checkpoint-10/hf_model \
  --test_dir data/lemon_v2 \
  --rsighan_dir data/rsighan \
  --categories gam,enc,cot,mec,car,nov,new,sig \
  --max_seq_length 128 \
  --batch_size 256 \
  --output_file outputs/relm-34m-paper/lemon_results.json