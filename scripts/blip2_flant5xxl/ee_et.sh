export CUDA_VISIBLE_DEVICES=0
exp_name='blip2_flant5xxl'
python tasks/tee_et.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--pred_span_path './output/blip2_flant5xxl/tee_span/paper/output.json' \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--configs qa4tee_base.yaml