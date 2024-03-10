export CUDA_VISIBLE_DEVICES=4
exp_name='vicuna13b'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path /mnt/Xsky/syx/project/2023/MMUIE/output_new2/vicuna13b/ner15_span/paper/output.json \
--model_type "vicuna13b" \
--mmodel_type "blip2_vicuna_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
