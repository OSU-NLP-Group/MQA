import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import sys


from model_utils import *
from my_utils import *
from read_dataset import get_ner_pred_data
from lavis.models import load_model_and_preprocess
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)


def get_et_w_prompt_bound(sent,span,config):
    example = ""
    example += '(A). ({}) is a location entity\n'.format(span)
    example += '(B). ({}) is a person entity\n'.format(span)
    example += '(C). ({}) is a organization entity\n'.format(span)
    example += '(D). ({}) is a miscellaneous entity\n'.format(span)
    example += '(E). ({}) is not a named entity or does not belong to type [location, person, organization, miscellaneous]\n'.format(span)
    index2rel = {'A':'loc',
                 'B':'per',
                 'C':'org',
                 'D':'other',
                 'E':'no'
                 }
    empty_prompt_sample_structure = config['example_format']
    empty_prompt = empty_prompt_sample_structure.format(sent, example)
    res_dict = {}
    res_dict['prompt'] = empty_prompt
    res_dict['index2rel'] = index2rel
    return res_dict

def get_entity_typing(model, sent, img_path, span, config, temp, args, vis_processors, vis_process_func, call_model_engine_fn, device, tokenizer=None):
    prompt_raw = temp(sent,span,config)
    index2rel =  prompt_raw['index2rel']
    prompt =  prompt_raw['prompt']
    sample = {'final_input_prompt': prompt, "img_path": img_path}
    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        sample['image'] = vis_process_func(sample['img_path'], vis_processors).to(device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')
    content = call_model_engine_fn(sample, model, tokenizer)

    all_choices = ['A','B','C','D','E']
    prediction = get_multi_choice_prediction(content,all_choices, index2rel)

    return prediction

def run_qa4et():
    parser = base_parser()
    parser.add_argument('--pred_span_path', type=str,default = '')
    args = parser.parse_args()
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    vis_processors = None
    call_model_engine =  call_blip2_engine_df
    vis_process_func = blip_image_processor

    if args.mmodel_type in ['blip2_t5', 'blip2_t5_instruct']:
        print('blip/instructblip initializing.....')
        model, vis_processors, _ = load_model_and_preprocess(name=args.mmodel_type,
                                                             model_type=args.model_type,
                                                             is_eval=True,
                                                             device=device)
    else:
        raise NotImplementedError(f'model type {args.mmodel_type} is not implemented')


    def run_exp(args,config, temp, sub_dataset, save_dir, sub_exp_dir):
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
        save_p_result_path =sub_exp_dir + 'pred_pred_result.json'
        save_p_type_dict_path =sub_exp_dir + 'pred_typedict.json'
        save_output_path =sub_exp_dir + 'pred_output.json'
        pred_pred_result_dict = {}
        from tqdm import tqdm
        new_samples = []
        #calculate the occurence of each type
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
            for span in sample['pred_spans']:
                s_type = get_entity_typing(model, sample['text'], sample['img_path'], span, config,temp, args, vis_processors, vis_process_func, call_model_engine, device)

                if s_type not in pred_type_dict:
                    pred_type_dict[s_type] = 0
                pred_type_dict[s_type] += 1
                sample['pred_pred_prediction'][span] = s_type
                if s_type == 'no':
                    continue
                else:
                    sample['pred_pred_output'][span] = s_type
                    pred_pred_all += 1
                    if span.lower() in gold_span_dict and s_type == gold_span_dict[span.lower()]:
                        pred_pred_correct += 1
            new_samples.append(sample)

        pred_pred_result_dict['correct'] = pred_pred_correct
        pred_pred_result_dict['all'] = pred_pred_all
        pred_pred_result_dict['precision'] = pred_pred_correct/pred_pred_all
        pred_pred_result_dict['recall'] = pred_pred_correct/gold_all
        pred_pred_result_dict['f1'] = 2*pred_pred_result_dict['precision']*pred_pred_result_dict['recall']/(pred_pred_result_dict['recall']+pred_pred_result_dict['precision'])
        save_json(save_p_type_dict_path, pred_type_dict)
        print('pred_type_statistics', pred_type_dict)
        save_json(save_p_result_path, pred_pred_result_dict)
        save_json(save_output_path, new_samples)

    
    # grid experiments
    configs = args.configs
    temps = [get_et_w_prompt_bound]
    save_dir = make_exp_dir('ner15_et', args.exp_name)
    sub_dataset = get_ner_pred_data(args.num_limit, args.pred_span_path)
    for config in configs:
        for temp in temps:
            sub_exp_name = 'paper_' + config  +'_/'
            run_exp(args,config,temp,sub_dataset, save_dir, sub_exp_name)


if __name__ == '__main__':
    run_qa4et()