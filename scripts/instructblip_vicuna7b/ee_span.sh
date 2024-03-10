export CUDA_VISIBLE_DEVICES=2
exp_name='vicuna7b'
python tasks/tee_span.py \
--exp_name ${exp_name} \
--num_limit 40000 --first_choice 1  --topn 2 \
--model_type="vicuna7b" \
--mmodel_type="blip2_vicuna_instruct"