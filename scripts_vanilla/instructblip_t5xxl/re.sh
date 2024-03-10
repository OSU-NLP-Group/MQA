export CUDA_VISIBLE_DEVICES=0
exp_name='instructblip_t5xxl_vanilla'


python tasks_vanilla/mre_15.py \
--exp_name ${exp_name} \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml

python tasks_vanilla/mre_17.py \
--exp_name ${exp_name} \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 4000 \
--configs qa4re_base.yaml
