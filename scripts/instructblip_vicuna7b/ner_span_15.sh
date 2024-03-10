export CUDA_VISIBLE_DEVICES=3
exp_name='vicuna7b'

python tasks/ner15_span.py \
--exp_name ${exp_name}  \
--model_type "vicuna7b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 5000
