export CUDA_VISIBLE_DEVICES=1
exp_name='blip2_flant5xl'
python tasks/tee_span.py \
--exp_name ${exp_name} \
--num_limit 40000 --first_choice 1  --topn 2 \
--model_type "pretrain_flant5xl" \
--mmodel_type "blip2_t5"