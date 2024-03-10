export CUDA_VISIBLE_DEVICES=3
exp_name='instructblip_t5xl_vanilla'
python tasks_vanilla/tee.py \
--exp_name ${exp_name} \
--num_limit 40000 \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4tee_2.yaml