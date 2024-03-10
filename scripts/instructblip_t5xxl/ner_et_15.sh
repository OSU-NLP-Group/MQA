export CUDA_VISIBLE_DEVICES=5
exp_name='instructblip_t5xxl'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/instructblip_t5xxl/ner15_span/paper/output.json \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
