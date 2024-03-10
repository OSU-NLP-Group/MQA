exp_name='instructblip_t5xl'

CUDA_VISIBLE_DEVICES=1 python tasks/img_ee.py \
--exp_name ${exp_name} \
--iee_settings 1 \
--num_limit 10000 \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4iee_1.yaml