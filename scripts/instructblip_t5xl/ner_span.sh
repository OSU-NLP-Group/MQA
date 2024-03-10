exp_name='instructblip_t5xl'

python tasks/ner15_span.py \
--exp_name ${exp_name}  \
--num_limit $1 \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct"