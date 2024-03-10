export CUDA_VISIBLE_DEVICES=7
exp_name='blip2_flant5xxl_vanilla'

python tasks_vanilla/ner15.py \
--exp_name ${exp_name} \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--num_limit 5000 \
--configs qa4ner_base.yaml
