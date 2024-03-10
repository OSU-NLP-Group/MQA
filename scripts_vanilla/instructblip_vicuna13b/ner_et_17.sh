export CUDA_VISIBLE_DEVICES=1

exp_name='vicuna13b_vanilla'


python tasks_vanilla/ner17.py \
--exp_name ${exp_name} \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 2000 \
--configs qa4ner_base.yaml