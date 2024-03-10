export CUDA_VISIBLE_DEVICES=0
exp_name='blip2_flant5xxl'

python tasks/ner17_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/blip2_flant5xxl/ner17_span/paper/output.json \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--num_limit 5000 \
--configs qa4ner_base.yaml
