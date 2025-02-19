#export AUTH_TOKEN=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

CUDA_VISIBLE_DEVICES=0 python main.py \
	--cuda 0 \
  --model CP \
  --model_name neuralmind/bert-base-portuguese-cased \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 2 \
	--max_length 512 \
	--save_step 1000 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_cp_datalawyer_bert \
