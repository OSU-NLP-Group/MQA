export CUDA_VISIBLE_DEVICES=3
exp_name='blip2_flant5xl'

python tasks/ner17_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/blip2_flant5xl/ner17_span/paper/output.json \
--model_type "pretrain_flant5xl" \
--mmodel_type "blip2_t5" \
--num_limit 5000 \
--configs qa4ner_base.yaml
