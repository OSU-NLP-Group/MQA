export CUDA_VISIBLE_DEVICES=7
exp_name='vicuna13b'

python tasks/ner17_et.py \
--exp_name ${exp_name} \
--pred_span_path ./output/vicuna13b/ner17_span/paper/output.json \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
