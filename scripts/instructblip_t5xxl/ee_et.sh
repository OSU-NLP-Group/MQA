export CUDA_VISIBLE_DEVICES=7
exp_name='instructblip_t5xxl'
python tasks/tee_et.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--pred_span_path './output/instructblip_t5xxl/tee_span/paper/output.json' \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4tee_2.yaml