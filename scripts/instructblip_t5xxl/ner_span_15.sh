export CUDA_VISIBLE_DEVICES=4
exp_name='instructblip_t5xxl'

python tasks/ner15_span.py \
--exp_name ${exp_name}  \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 5000