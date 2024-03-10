export CUDA_VISIBLE_DEVICES=7
exp_name='blip2_flant5xl'
python tasks/tee_et.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--pred_span_path './output/blip2_flant5xl/tee_span/paper/output.json' \
--model_type "pretrain_flant5xl" \
--mmodel_type "blip2_t5" \
--configs qa4tee_2.yaml
