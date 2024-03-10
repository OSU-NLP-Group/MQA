export CUDA_VISIBLE_DEVICES=1

exp_name='instructblip_t5xxl_vanilla'


python tasks_vanilla/ner17.py \
--exp_name ${exp_name} \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 2000 \
--configs qa4ner_base.yaml