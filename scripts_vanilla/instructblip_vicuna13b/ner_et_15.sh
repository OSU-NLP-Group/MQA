export CUDA_VISIBLE_DEVICES=7
exp_name='vicuna13b_vanilla'

python tasks_vanilla/ner15.py \
--exp_name ${exp_name} \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml