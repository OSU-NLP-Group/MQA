export CUDA_VISIBLE_DEVICES=2
exp_name='blip2_flant5xl_vanilla'
python tasks_vanilla/tee.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--model_type "pretrain_flant5xl" \
--mmodel_type "blip2_t5" \
--configs qa4tee_base.yaml