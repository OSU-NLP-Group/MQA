export CUDA_VISIBLE_DEVICES=0
exp_name='instructblip_t5xl'

python tasks/mre_15.py \
--exp_name ${exp_name} \
--num_limit 4000 \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4re_3.yaml
