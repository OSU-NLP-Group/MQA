exp_name='blip2_flant5xxl_vanilla'
python tasks_vanilla/tee.py \
--exp_name ${exp_name} \
--num_limit $1 \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--configs qa4tee_base.yaml
