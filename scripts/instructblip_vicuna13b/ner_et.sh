exp_name='vicuna13b'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path './output/vicuna13b/ner15_span/paper/output.json' \
--num_limit $1 \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct" \
--configs qa4ner_base.yaml