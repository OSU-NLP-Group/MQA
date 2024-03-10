export CUDA_VISIBLE_DEVICES=0
exp_name='vicuna7b'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/vicuna7b/ner15_span/paper/output.json \
--model_type "vicuna7b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
