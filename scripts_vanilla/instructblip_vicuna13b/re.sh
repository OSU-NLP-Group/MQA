export CUDA_VISIBLE_DEVICES=7
exp_name='vicuna13b_vanilla'


python tasks_vanilla/mre_15.py \
--exp_name ${exp_name} \
--model_type="vicuna13b" \
--mmodel_type="blip2_vicuna_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml

python tasks_vanilla/mre_17.py \
--exp_name ${exp_name} \
--model_type="vicuna13b" \
--mmodel_type="blip2_vicuna_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml