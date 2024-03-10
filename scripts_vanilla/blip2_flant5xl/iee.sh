exp_name='blip2_flant5xl_vanilla'

CUDA_VISIBLE_DEVICES=1 python tasks_vanilla/img_ee.py \
--exp_name ${exp_name} \
--iee_settings 1 \
--num_limit 10000 \
--model_type "pretrain_flant5xl" \
--mmodel_type "blip2_t5" \
--configs qa4iee_base.yaml