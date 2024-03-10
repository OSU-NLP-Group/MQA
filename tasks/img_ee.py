import os
import sys
import torch

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from my_utils import *
from model_utils import *
from read_dataset import *
from lavis.models import load_model_and_preprocess

# NER span prompt
MNRE_LABELS = ['Movement:Transport', 'Contact:Phone-Write', 'Conflict:Attack', 'Contact:Meet', 'Justice:Arrest-Jail',
               'Conflict:Demonstrate', 'Life:Die', 'Transaction:Transfer-Money', 'None']

MNRE_TEMPLATES = {
    'Movement:Transport': ['The image with the sentence describe the Transportation of Movement event'],
    'Contact:Phone-Write': ['The image with the sentence describe the Phone or Write of Contact event'],
    'Conflict:Attack': ['The image with the sentence describe the Attack with no death of Conflict event'],
    'Contact:Meet': ['The image with the sentence describe the Meet of Contact event'],
    'Justice:Arrest-Jail': ['The image with the sentence describe the Arrest of Criminal event'],
    'Conflict:Demonstrate': ['The image with the sentence describe the Demonstration of Conflict event'],
    'Life:Die': ['The image with the sentence describe the Death of Life event'],
    'Transaction:Transfer-Money': ['The image with the sentence describe the Money Transfer of Transaction event'],
    'None': ['The image describes no event']
}


def process_arguments(args):
    args.LABELS = MNRE_LABELS
    args.LABEL_TEMPLATES = args.label_template
    args.NOTA_RELATION = "None"
    args.POS_LABELS = list(set(args.LABELS) - set([args.NOTA_RELATION]))
    args.LABEL_VERBALIZER = {k: k for k in args.LABELS}
    return args


def construct_ee_prompt(sample, args, config):
    sent = sample['text']
    label = sample['label']['event_type']
    example = ""
    correct_templates = args.LABEL_TEMPLATES[label]

    start_chr = 'A'
    valid_relations = args.LABELS
    correct_template_index = []
    prediction_range = []
    index2rel = {}
    for valid_relation in valid_relations:
        for template in args.LABEL_TEMPLATES[valid_relation]:
            filled_template = template
            if template in correct_templates:
                correct_template_index.append(start_chr)
            prediction_range.append(start_chr)
            example += f"({start_chr}). {filled_template}\n"
            index2rel[start_chr] = valid_relation
            start_chr = chr(ord(start_chr) + 1)
    correct_index = correct_template_index[0]
    empty_prompt_sample_structure = config['example_format']
    empty_prompt = empty_prompt_sample_structure.format(example)

    res_dict = {}
    res_dict['index2rels'] = index2rel
    res_dict['correct_choices'] = correct_template_index
    res_dict['all_choices'] = prediction_range
    res_dict['empty_prompt'] = empty_prompt

    return res_dict


def build_final_input(sample, config):
    task_instructions = config['task_instructions'].strip()
    final_input_prompt = task_instructions + '\n\n' + sample['empty_prompt']
    return final_input_prompt


def evaluate_ee(article2samples):
    # compose
    pred_correct = 0
    pred_total = 0
    gold_total = 0

    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    original_element_pred_correct = 0
    original_element_pred_positive = 0
    original_element_gold_positive = 0
    for article in article2samples:
        gold_list = []
        pred_list = []
        for sample in article2samples[article]:
            gold_list.append(sample['gold'])
            pred_list.append(sample['pred'])
        for i in range(len(gold_list)):
            gold_i = gold_list[i]
            pred_i = pred_list[i]
            if gold_i != 'None':
                original_element_gold_positive += 1
            if pred_i != 'None':
                original_element_pred_positive += 1
            if gold_i == pred_i and gold_i != 'None':
                original_element_pred_correct += 1

        # consider None into all samples
        gold_set = set(gold_list)
        pred_set = set(pred_list)

        pred_correct += len(gold_set & pred_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)

        # discard None
        gold_set_wo_none = gold_set - {'None'}
        pred_set_wo_none = pred_set - {'None'}
        correct_positive += len(gold_set_wo_none & pred_set_wo_none)
        pred_positive += len(pred_set_wo_none)
        gold_positive += len(gold_set_wo_none)

    element_result = calc_prf(original_element_pred_correct, original_element_pred_positive,
                              original_element_gold_positive)
    all_result = calc_prf(pred_correct, pred_total, gold_total)
    wo_none_result = calc_prf(correct_positive, pred_positive, gold_positive)

    return {'all_result': all_result, 'wo_none_result': wo_none_result, 'fine_result': element_result}


def calc_prf(pred_correct, pred_total, gold_total):
    try:
        micro_p = float(pred_correct) / float(pred_total)
    except:
        micro_p = 0
    try:
        micro_r = float(pred_correct) / float(gold_total)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    return {'p': micro_p, 'r': micro_r, 'f1': micro_f1}


def prompt_preparation(args, config, sub_dataset):
    samples = []
    for sample in sub_dataset:
        if args.iee_settings == 1:
            sent = 'This is a image attached to an news article'
        elif args.iee_settings == 2:
            sent = sample['caption']
        elif args.iee_settings == 3:
            sent = sample['text'][0]
        sample['text'] = sent
        cs_sample = construct_ee_prompt(sample, args, config)
        cs_sample['text'] = sample['text']
        cs_sample['final_input_prompt'] = build_final_input(cs_sample, config)
        cs_sample['verbalized_labels'] = cs_sample['all_choices']
        cs_sample['img_path'] = sample['img_path']
        cs_sample['gold'] = sample['label']['event_type']
        cs_sample['article'] = sample['article']
        samples.append(cs_sample)
    return samples


def run_qa4iee():
    parser = base_parser()
    parser.add_argument('--iee_settings', type=int, default=1)
    parser.add_argument('--constrained', action='store_false')
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    vis_processors = None
    tokenizer = None
    call_model_engine = call_blip2_engine_df
    vis_process_func = blip_image_processor
    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        print('blip/instructblip initializing.....')
        model, vis_processors, _ = load_model_and_preprocess(name=args.mmodel_type,
                                                             model_type=args.model_type,
                                                             is_eval=True,
                                                             device=device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')

    def run_exp(args, config, temp, sub_dataset, save_dir, sub_exp_dir):
        args.config = config
        args.label_template = temp
        args = process_arguments(args)
        config = load_yaml(args.config)
        for key, value in config.items():
            if key != 'eval_params' and type(value) == list:
                assert len(value) == 1, 'key {} has more than one value'.format(key)
                config[key] = value[0]

        samples = prompt_preparation(args, config, sub_dataset)
        out_samples = []
        exp_dir = save_dir
        sub_exp_dir = exp_dir + sub_exp_dir
        if not os.path.exists(sub_exp_dir):
            os.makedirs(sub_exp_dir)

        save_output_path = sub_exp_dir + 'output.json'
        save_result_path = sub_exp_dir + 'result.json'

        from tqdm import tqdm
        article2samples = {}
        for sample in tqdm(samples):
            content, choice = call_blip2_engine_df(args, sample, model, vis_processors, vis_process_func,
                                                   call_model_engine, device, tokenizer)
            out_samples.append(
                {'gold': sample['gold'], 'article': sample['article'], 'content': content, 'choice': choice,
                 'text': sample['text'], 'prompt': sample['final_input_prompt']})
            if sample['article'] not in article2samples:
                article2samples[sample['article']] = []
            article2samples[sample['article']].append({'gold': sample['gold'], 'pred': choice})

        save_json(save_output_path, out_samples)
        metric_dict = evaluate_ee(article2samples)
        save_json(save_result_path, metric_dict)

    configs = args.configs
    if args.iee_settings == 1:
        sub_dataset = get_img_only(args.num_limit)
    elif args.iee_settings == 2:
        sub_dataset = get_img_only_w_caption(args.num_limit)
    elif args.iee_settings == 3:
        sub_dataset = get_img_only_w_text(args.num_limit)
    save_dir = make_exp_dir('img_ee', args.exp_name)
    print('save results in {}'.format(save_dir))
    temp = MNRE_TEMPLATES

    for config in configs:
        sub_exp_name = 'paper_' + config + '_/'
        run_exp(args, config, temp, sub_dataset, save_dir, sub_exp_name)


def get_dict_name(my_dict):
    for name, value in globals().items():
        if value is my_dict:
            return name


def call_blip2_engine_df(args, sample, model, vis_processors, vis_process_func, call_model_engine_fn, device,
                         tokenizer=None):
    all_choices = sample['all_choices']
    index2rel = sample['index2rels']
    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        sample['image'] = vis_process_func(sample['img_path'], vis_processors).to(device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')
    content = call_model_engine_fn(sample, model, tokenizer)

    prediction = get_multi_choice_prediction(content, all_choices, index2rel)
    return content, prediction


if __name__ == '__main__':
    run_qa4iee()
