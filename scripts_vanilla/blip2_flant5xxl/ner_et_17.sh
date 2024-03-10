export CUDA_VISIBLE_DEVICES=1
exp_name='blip2_flant5xxl_vanilla'

python tasks_vanilla/ner17.py \
--exp_name ${exp_name} \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--num_limit 2000 \
--configs qa4ner_base.yaml