export CUDA_VISIBLE_DEVICES=0
exp_name='instructblip_t5xl'

python tasks/ner15_et.py \
--exp_name ${exp_name} \
--pred_span_path /mnt/Xsky/syx/project/2023/MMUIE/output_new2/instructblip_t5xl/ner15_span/paper/output.json \
--model_type "flant5xl" \
--mmodel_type "blip2_t5_instruct" \
--num_limit 5000 \
--configs qa4ner_base.yaml
