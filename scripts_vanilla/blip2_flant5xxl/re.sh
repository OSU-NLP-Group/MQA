exp_name='blip2_flant5xxl_vanilla'

python tasks_vanilla/mre_17.py \
--exp_name ${exp_name} \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--num_limit 8000 \
--configs qa4re_base.yaml