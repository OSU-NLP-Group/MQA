export CUDA_VISIBLE_DEVICES=7
exp_name='vicuna7b'
python tasks/tee_et.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--pred_span_path './output/vicuna7b/tee_span/paper/output.json' \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct" \
--configs qa4tee_2.yaml

