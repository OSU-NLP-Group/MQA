import sys
import torch

from my_utils import *
from model_utils import *
from read_dataset import get_ner_pred_data
from lavis.models import load_model_and_preprocess

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


def get_et_prompt_instruct(sent, span, config):
    example = ""
    example += 'A. The word "{}" is a common word and does not reflect any of the other event\n'.format(span)
    example += 'B. The word "{}" is the key of the Transport action, which is a subtype of Movement event\n'.format(
        span)
    example += 'C. The word "{}" is the key of the Phone or Write action, which is a subtype of Contact event\n'.format(
        span)
    example += 'D. The word "{}" is the key of the conflict but no death action, which is a subtype of Conflict event\n'.format(
        span)
    example += 'E. The word "{}" is the key of the Meeting action, which is a subtype of Contact event\n'.format(span)
    example += 'F. The word "{}" is the key of the Crime Arrest or sent into Jail action, which is a subtype of Justice event\n'.format(
        span)
    example += 'G. The word "{}" is the key of the Demonstrate action, which is a subtype of Conflict event\n'.format(
        span)
    example += 'H. The word "{}" is the key of the Die action, which is a subtype of Life event\n'.format(span)
    example += 'I. The word "{}" is the key of the Transfer-Money action, which is a subtype of Transaction event\n'.format(
        span)
    index2rel = {'A': 'no',
                 'B': 'Movement:Transport',
                 'C': 'Contact:Phone-Write',
                 'D': 'Conflict:Attack',
                 'E': 'Contact:Meet',
                 'F': 'Justice:Arrest-Jail',
                 'G': 'Conflict:Demonstrate',
                 'H': 'Life:Die',
                 'I': 'Transaction:Transfer-Money',
                 }

    empty_prompt_sample_structure = config['example_format']
    empty_prompt = empty_prompt_sample_structure.format(sent, example)
    task_instructions = config['task_instructions'].strip()
    final_input_prompt = task_instructions + '\n\n' + empty_prompt
    res_dict = {}
    res_dict['prompt'] = final_input_prompt
    res_dict['index2rel'] = index2rel
    return res_dict


# best增强2，把 in the sentence去掉，减少句子长度
def get_et_prompt(sent, span, config):
    example = ""
    example += 'A. The word "{}" is a common word and does not reflect any of the other event\n'.format(span)
    example += 'B. The word "{}" is the key of the Transport action, which is a subtype of Movement event\n'.format(
        span)
    example += 'C. The word "{}" is the key of the PhoneWrite action, which is a subtype of Contact event\n'.format(
        span)
    example += 'D. The word "{}" is the key of the conflict but no death action, which is a subtype of Conflict event\n'.format(
        span)
    example += 'E. The word "{}" is the key of the Meeting action, which is a subtype of Contact event\n'.format(span)
    example += 'F. The word "{}" is the key of the Crime Arrest or sent into Jail action, which is a subtype of Justice event\n'.format(
        span)
    example += 'G. The word "{}" is the key of the Demonstrate action, which is a subtype of Conflict event\n'.format(
        span)
    example += 'H. The word "{}" is the key of the Die action, which is a subtype of Life event\n'.format(span)
    example += 'I. The word "{}" is the key of the Transfer-Money action, which is a subtype of Transaction event\n'.format(
        span)
    index2rel = {'A': 'no',
                 'B': 'Movement:Transport',
                 'C': 'Contact:Phone-Write',
                 'D': 'Conflict:Attack',
                 'E': 'Contact:Meet',
                 'F': 'Justice:Arrest-Jail',
                 'G': 'Conflict:Demonstrate',
                 'H': 'Life:Die',
                 'I': 'Transaction:Transfer-Money',
                 }

    empty_prompt_sample_structure = config['example_format']
    empty_prompt = empty_prompt_sample_structure.format(sent, example)
    task_instructions = config['task_instructions'].strip()
    final_input_prompt = task_instructions + '\n\n' + empty_prompt
    res_dict = {}
    res_dict['prompt'] = final_input_prompt
    res_dict['index2rel'] = index2rel
    return res_dict


def get_entity_typing(model, sent, img_path, span, config, temp, args, vis_processors, vis_process_func,
                      call_model_engine_fn, device):
    prompt_raw = temp(sent, span, config)
    index2rel = prompt_raw['index2rel']
    prompt = prompt_raw['prompt']
    sample = {'final_input_prompt': prompt, "img_path": img_path}
    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        sample['image'] = vis_process_func(sample['img_path'], vis_processors).to(device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')
    content = call_model_engine_fn(sample, model)
    all_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    prediction = get_multi_choice_prediction(content, all_choices, index2rel)

    return prediction, prompt


def prepare_known_types(label):
    if label == 'no':
        return None
    else:
        elems = label.split('/')
        head = elems[1]
        tail = elems[2]
        return {'h': head, 't': tail}


def run_qa4et():
    parser = base_parser()
    parser.add_argument('--pred_span_path', type=str, default='')
    args = parser.parse_args()
    # setup device to use
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
        pred_pred_all = 0
        pred_pred_correct = 0
        gold_all = 0

        args.config = config
        args.label_template = temp
        config = load_yaml(args.config)
        for key, value in config.items():
            if key != 'eval_params' and type(value) == list:
                assert len(value) == 1, 'key {} has more than one value'.format(key)
                config[key] = value[0]

        exp_dir = save_dir
        sub_exp_dir = exp_dir + sub_exp_dir
        if not os.path.exists(sub_exp_dir):
            os.makedirs(sub_exp_dir)
        save_p_result_path = sub_exp_dir + 'pred_pred_result.json'
        save_p_type_dict_path = sub_exp_dir + 'pred_typedict.json'
        save_output_path = sub_exp_dir + 'pred_output.json'
        pred_pred_result_dict = {}
        from tqdm import tqdm
        new_samples = []
        # calculate the occurence of each type
        pred_type_dict = {}
        for sample in tqdm(sub_dataset):
            gold_span_w_type = sample['gold_spans_w_type']
            # construct gold dict
            gold_span_dict = {}
            for type_dict in gold_span_w_type:
                gold_span_dict[type_dict["span"].lower()] = type_dict["type"]
            gold_all += len(gold_span_w_type)
            sample['pred_pred_output'] = {}
            sample['pred_pred_prediction'] = {}
            sample['pred_pred_prompt'] = {}
            for span in sample['pred_spans']:
                s_type, pred_prompt = get_entity_typing(model, sample['text'], sample['img_path'], span, config, temp,
                                                        args, vis_processors, vis_process_func, call_model_engine,
                                                        device)

                if s_type not in pred_type_dict:
                    pred_type_dict[s_type] = 0
                sample['pred_pred_prompt'][span] = pred_prompt
                pred_type_dict[s_type] += 1
                sample['pred_pred_prediction'][span] = s_type
                if s_type == 'no':
                    continue
                else:
                    sample['pred_pred_output'][span] = s_type
                    pred_pred_all += 1
                    if span.lower() in gold_span_dict and s_type == gold_span_dict[span.lower()]:
                        pred_pred_correct += 1
            sample['gold_pred_output'] = {}
            sample['gold_pred_prediction'] = {}
            sample['gold_pred_prompt'] = {}
            new_samples.append(sample)

        pred_pred_result_dict['correct'] = pred_pred_correct
        pred_pred_result_dict['all'] = pred_pred_all
        if pred_pred_all == 0:
            pred_pred_result_dict['precision'] = 0
        else:
            pred_pred_result_dict['precision'] = pred_pred_correct / pred_pred_all
        pred_pred_result_dict['recall'] = pred_pred_correct / gold_all
        if (pred_pred_result_dict['recall'] + pred_pred_result_dict['precision']) == 0:
            pred_pred_result_dict['f1'] = 0
        else:
            pred_pred_result_dict['f1'] = 2 * pred_pred_result_dict['precision'] * pred_pred_result_dict['recall'] / (
                    pred_pred_result_dict['recall'] + pred_pred_result_dict['precision'])
        print('pred_pred_result', pred_pred_result_dict)
        save_json(save_p_type_dict_path, pred_type_dict)
        print('pred_type_statistics', pred_type_dict)
        save_json(save_p_result_path, pred_pred_result_dict)
        save_json(save_output_path, new_samples)

    # grid experiments
    configs = args.configs
    if 'instruct' in args.mmodel_type:
        temp = get_et_prompt_instruct
    else:
        temp = get_et_prompt
    save_dir = make_exp_dir('tee_et', args.exp_name)
    sub_dataset = get_ner_pred_data(args.num_limit, args.pred_span_path)
    for config in configs:
        sub_exp_name = 'paper_' + config + '_/'
        run_exp(args, config, temp, sub_dataset, save_dir, sub_exp_name)


if __name__ == '__main__':
    run_qa4et()
