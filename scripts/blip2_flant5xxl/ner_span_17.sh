export CUDA_VISIBLE_DEVICES=1
exp_name='blip2_flant5xxl'

python tasks/ner17_span.py \
--exp_name ${exp_name}  \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--num_limit 5000