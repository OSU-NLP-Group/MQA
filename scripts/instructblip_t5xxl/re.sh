export CUDA_VISIBLE_DEVICES=1
exp_name='instructblip_t5xxl'


python tasks/mre_17.py \
--exp_name ${exp_name} \
--num_limit 4000  \
--model_type "flant5xxl" \
--mmodel_type "blip2_t5_instruct" \
--configs qa4re_3.yaml