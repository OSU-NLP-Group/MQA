export CUDA_VISIBLE_DEVICES=0
exp_name='vicuna7b'


python tasks/mre_15.py \
--exp_name ${exp_name} \
--num_limit 4000 \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct" \
--configs qa4re_base.yaml