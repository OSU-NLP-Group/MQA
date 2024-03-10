export CUDA_VISIBLE_DEVICES=1
exp_name='blip2_flant5xxl'
python tasks/tee_span.py \
--exp_name ${exp_name} \
--num_limit 20000 --first_choice 1  --topn 2 \
--model_type "pretrain_flant5xxl" \
--mmodel_type "blip2_t5" \
--configs qa4tee_base.yaml
