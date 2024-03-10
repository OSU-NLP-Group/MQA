exp_name='vicuna13b'


python tasks/ner15_span.py \
--exp_name ${exp_name}  \
--num_limit $1 \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct"
#--configs