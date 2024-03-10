export CUDA_VISIBLE_DEVICES=4
exp_name='instructblip_t5xxl_vanilla'
python tasks_vanilla/tee.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4tee_2.yaml