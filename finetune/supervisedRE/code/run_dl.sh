array=(42 40 30 20 10)
ckpt="ckpt_cp_datalawyer_bert/ckpt_of_step_14000"
for seed in ${array[@]}; do
  for fold in {0..4}; do
    bash train_dl.sh 0 $seed $ckpt "0.24-doutorado" $fold 20
  done
done
