BASE_OUTPUT_DIR=/media/pedro/arquivos/models/datalawyer/re_context_or_names/bert_full_cp

CUDA_VISIBLE_DEVICES=$1 python main.py \
	--seed "$2" \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch "$6" \
	--max_length 512 \
	--mode CM \
	--dataset datalawyer \
	--dataset_version "$4" \
	--fold "$5" \
	--entity_marker \
	--ckpt_to_load "$3" \
  --save_dir ${BASE_OUTPUT_DIR}/"$4"-"$5"-"$2" \
  --model_name neuralmind/bert-base-portuguese-cased \
