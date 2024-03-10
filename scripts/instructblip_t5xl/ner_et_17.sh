export CUDA_VISIBLE_DEVICES=1
exp_name='instructblip_t5xl'

python tasks/ner17_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/instructblip_t5xl/ner17_span/paper/output.json \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
