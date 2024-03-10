export CUDA_VISIBLE_DEVICES=6
exp_name='vicuna13b_vanilla'
python tasks_vanilla/tee.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--model_type="vicuna13b" \
--mmodel_type="blip2_vicuna_instruct" \
--configs qa4tee_base.yaml