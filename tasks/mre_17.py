from model_utils import *
from my_utils import *
import sys
from read_dataset import get_re_data, get_re_data_type

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

def self_eval(gold, pred_result, use_name=False):
    """
    Args:
        pred_result: a list of predicted label (id)
            Make sure that the `shuffle` param is set to `False` when getting the loader.
        use_name: if True, `pred_result` contains predicted relation names instead of ids
    Return:
        {'acc': xx}
    """
    correct = 0
    total = len(gold)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = 'None'
    for i in range(total):
        golden = gold[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    # result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    return acc, micro_p, micro_r, micro_f1


MNRE_LABELS_dict = {"None": 0, "/per/per/parent": 1, "/per/per/siblings": 2, "/per/per/couple": 3, \
                    "/per/per/neighbor": 4, "/per/per/peer": 5, "/per/per/charges": 6, "/per/per/alumi": 7, \
                    "/per/per/alternate_names": 8, "/per/org/member_of": 9, "/per/loc/place_of_residence": 10, \
                    "/per/loc/place_of_birth": 11, "/org/org/alternate_names": 12, "/org/org/subsidiary": 13, \
                    "/org/loc/locate_at": 14, "/loc/loc/contain": 15, "/per/misc/present_in": 16,
                    "/per/misc/awarded": 17, \
                    "/per/misc/race": 18, "/per/misc/religion": 19, "/per/misc/nationality": 20,
                    "/misc/misc/part_of": 21, \
                    "/misc/loc/held_on": 22}

MNRE_VALID_CONDITIONS_REV = {
    "per:per": [
        "/per/per/parent",
        "/per/per/siblings",
        "/per/per/couple",
        "/per/per/neighbor",
        "/per/per/peer",
        "/per/per/charges",
        "/per/per/alumi",
        "/per/per/alternate_names",
    ],
    "per:org": ["/per/org/member_of", ],
    "per:loc": ["/per/loc/place_of_residence", "/per/loc/place_of_birth", ],
    "org:org": ["/org/org/alternate_names", "/org/org/subsidiary", ],
    "org:loc": ["/org/loc/locate_at", ],
    "loc:loc": ["/loc/loc/contain", ],
    "per:misc": ["/per/misc/present_in", "/per/misc/awarded", "/per/misc/race", "/per/misc/religion",
                 "/per/misc/nationality"],
    "misc:misc": ["/misc/misc/part_of", ],
    "misc:loc": ["/misc/loc/held_on", ],
}

# NER span prompt
MNRE_LABELS = list(MNRE_LABELS_dict.keys())

# modified peer based on template 4, change alternative nave to a is b, and subsidiary is changed to a not confusing one.
MNRE_TEMPLATES = {
    "None": ["No known relationship between {subj} and {obj}"],
    "/per/per/parent": ["{obj} is the child of {subj}"],
    "/per/per/siblings": ["{subj} has the same age as {obj}", ],
    "/per/per/couple": ["{subj} is the couple of {obj}"],
    "/per/per/neighbor": ["{subj} is the neighbor of {obj}"],
    "/per/per/peer": ["{subj} meets with {obj}", "{subj} and {obj} are in the same team/league",
                      "{subj} and {obj} are nominated for the same honor"],
    "/per/per/charges": ["{subj} charges {obj}"],
    "/per/per/alumi": ["{subj} is an alumnus/alumna of {obj}"],
    "/per/per/alternate_names": ["{subj} is {obj}"],
    "/per/org/member_of": ["{subj} is a member of {obj}"],
    "/per/loc/place_of_residence": ["{subj} lives in {obj}"],
    "/per/loc/place_of_birth": ["{subj} was born in {obj}"],
    "/org/org/alternate_names": ["{subj} has the alternate name {obj}"],
    "/org/org/subsidiary": ["{subj} is a subsidiary or branch of {obj}"],
    "/org/loc/locate_at": ["{subj} is located at {obj}"],
    "/loc/loc/contain": ["{subj} contains {obj}"],
    "/per/misc/present_in": ["{subj} is present in {obj}"],
    "/per/misc/awarded": ["{subj} is awarded {obj}"],
    "/per/misc/race": ["{subj} belongs to the race {obj}"],
    "/per/misc/religion": ["{subj} has the religion {obj}"],
    "/per/misc/nationality": ["{subj} has the nationality {obj}"],
    "/misc/misc/part_of": ["{subj} is part of {obj}"],
    "/misc/loc/held_on": ["{subj} is held on {obj}"],
}

def process_arguments(args):
    args.LABELS = MNRE_LABELS
    args.LABEL_TEMPLATES = MNRE_TEMPLATES
    args.NOTA_RELATION = "None"
    args.POS_LABELS = list(set(args.LABELS) - set([args.NOTA_RELATION]))
    args.LABEL_VERBALIZER = {k: k for k in args.LABELS}
    args.VALID_CONDITIONS_REV = MNRE_VALID_CONDITIONS_REV
    return args

def construct_re_prompt(ent1, ent2, sent, label, args, config, h_type='', t_type=''):
    example = ""
    correct_templates = args.LABEL_TEMPLATES[label]
    type_cons = h_type + ':' + t_type
    start_chr = 'A'
    if args.constraint == 1:
        if type_cons in args.VALID_CONDITIONS_REV:
            valid_relations = args.VALID_CONDITIONS_REV[type_cons] + ['None']
        else:
            valid_relations = ['None']
    else:
        valid_relations = args.LABELS
    correct_template_index = []
    prediction_range = []
    index2rel = {}
    for valid_relation in valid_relations:
        for template in args.LABEL_TEMPLATES[valid_relation]:
            filled_template = template.format(subj=f'"{ent1}"', obj=f'"{ent2}"')
            if template in correct_templates:
                correct_template_index.append(start_chr)
            prediction_range.append(start_chr)
            example += f"({start_chr}). {filled_template}\n"
            index2rel[start_chr] = valid_relation
            start_chr = chr(ord(start_chr) + 1)
    empty_prompt_sample_structure = config['example_format']
    empty_prompt = empty_prompt_sample_structure.format(sent, example)

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


def evaluate_re(df, prediction_name='predictions'):
    acc, precision, recall, f1 = self_eval(df['golds'], df[prediction_name])
    return {'f1': f1, 'precision': precision, 'recall': recall, 'acc': acc}


def prompt_preparation(args, config, sub_dataset):
    samples = []
    for sample in sub_dataset:
        if args.constraint == 1:
            h_type = sample["known_h"] if sample["known_h"] != "None" else sample['pred_h']
            t_type = sample["known_t"] if sample["known_t"] != "None" else sample['pred_t']
            cs_sample = construct_re_prompt(sample['head'], sample['tail'], sample['text'], sample['relation'], args,
                                            config, h_type, t_type)
        else:
            cs_sample = construct_re_prompt(sample['head'], sample['tail'], sample['text'], sample['relation'], args,
                                            config)
        cs_sample['text'] = sample['text']
        cs_sample['final_input_prompt'] = build_final_input(cs_sample, config)
        cs_sample['verbalized_labels'] = cs_sample['all_choices']
        cs_sample['img_path'] = sample['img_path']
        cs_sample['gold'] = sample['relation']
        samples.append(cs_sample)
    return samples


def run_qa4mre():
    parser = base_parser()
    parser.add_argument('--constraint', type=int, default=1)
    parser.add_argument('--entity_bound', type=int, default=1)
    args = parser.parse_args()
    import torch
    from lavis.models import load_model_and_preprocess
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

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

    def run_exp(args, config, sub_dataset, save_dir, sub_exp_dir):
        args.config = config
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
        for sample in tqdm(samples):
            content, choice = call_blip2_engine_df(args, sample, model, vis_processors, vis_process_func,
                                                   call_model_engine, device)
            out_samples.append({'gold': sample['gold'], 'content': content, 'choice': choice, 'text': sample['text'],
                                'prompt': sample['final_input_prompt']})
        dfs = {}
        dfs['predictions'] = [d['choice'] for d in out_samples]
        dfs['golds'] = [d['gold'] for d in samples]
        save_json(save_output_path, out_samples)
        metric_dict = evaluate_re(df=dfs)
        save_json(save_result_path, metric_dict)

    # grid experiments
    configs = args.configs
    if args.constraint == 0:
        sub_dataset = get_re_data(args.num_limit, '17')
    else:
        sub_dataset = get_re_data_type(args.num_limit, '17')
    save_dir = make_exp_dir('re17', args.exp_name)
    for config in configs:
        sub_exp_name = 'paper_' + config + '_/'
        print(sub_exp_name)
        run_exp(args, config, sub_dataset, save_dir, sub_exp_name)


def call_blip2_engine_df(args, sample, model, vis_processors, vis_process_func, call_model_engine_fn, device):
    all_choices = sample['all_choices']
    index2rel = sample['index2rels']
    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        sample['image'] = vis_process_func(sample['img_path'], vis_processors).to(device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')
    content = call_model_engine_fn(sample, model)
    prediction = get_multi_choice_prediction(content, all_choices, index2rel)
    return content, prediction


if __name__ == '__main__':
    run_qa4mre()
