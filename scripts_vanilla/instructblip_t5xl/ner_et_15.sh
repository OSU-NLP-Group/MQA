export CUDA_VISIBLE_DEVICES=7
exp_name='instructblip_t5xl_vanilla'

python tasks_vanilla/ner15.py \
--exp_name ${exp_name} \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml