export CUDA_VISIBLE_DEVICES=6
exp_name='vicuna7b_vanilla'


python tasks_vanilla/mre_15.py \
--exp_name ${exp_name} \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml

python tasks_vanilla/mre_17.py \
--exp_name ${exp_name} \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml