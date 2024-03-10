exp_name='instructblip_t5xl'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path './output/instructblip_t5xl/ner15_span/paper/output.json' \
--num_limit $1 \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct"