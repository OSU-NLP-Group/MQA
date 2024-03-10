exp_name='vicuna7b'

CUDA_VISIBLE_DEVICES=1 python tasks/img_ee.py \
--exp_name ${exp_name} \
--iee_settings 1 \
--num_limit 10000 \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct" \
--configs qa4iee_1.yaml