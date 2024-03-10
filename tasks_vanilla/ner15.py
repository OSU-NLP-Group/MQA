import os
from my_utils import *
from read_dataset import *
import os
# 一阶段式vanilla
import sys
sys.path.append("..")

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
def calculate_span_correctness(gold_span, pred_span):
    gold_span1 = set([i.lower() for i in gold_span])
    pred_span1 = set([i.lower() for i in pred_span])
    and_span = gold_span1  & pred_span1
    return len(and_span), len(pred_span1), len(gold_span1)

def calculate_ner_correctness(gold_span, pred_span):
    # gold span的格式是 list，list里面是 {'span':span, 'type':type}
    # pred_span格式是 dict, key是type，value是list，list里面是span
    # 将gold_span转换成dict
    gold_span_dict = {}
    for span in gold_span:
        if span['type'] not in gold_span_dict:
            gold_span_dict[span['type']] = []
        gold_span_dict[span['type']].append(span['span'].lower())
    # 将gold_span_dict里每个key对应的value转换成set
    for key in gold_span_dict:
        gold_span_dict[key] = set(gold_span_dict[key])
    # 计算overlap
    gold_all = 0
    pred_all = 0
    overlap = 0
    # 遍历pred_span和gold_span_dict的所有key
    for key in gold_span_dict.keys() | pred_span.keys():
        if key in gold_span_dict:
            gold_all += len(gold_span_dict[key])
        if key in pred_span:
            pred_all += len(pred_span[key])
        if key in gold_span_dict and key in pred_span:
        # 计算overlap
            overlap += len(gold_span_dict[key] & pred_span[key])
        # 计算pred_all
    return overlap, pred_all, gold_all

# QA4NER 2 best f1 0.532
def get_qa4re_spans2_named_mention(model, text, config, image, gold_ners):
    #('qa2')
    task_instructions = config['task_instructions'].strip()
    prompt = task_instructions + '\n\n' +  config['example_format']
    O = ["location", "person", "organization", "miscellaneous"]
    # output_ners = []
    output_ners = {}
    pred_ners = {}
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt }, max_length= 10)[0]
        else:
            span_i = text.split(' ')[0]
        span_i_res = span_i.split(',')
        span_i_res = [ii.lower().strip() for ii in span_i_res]
        filter_ners = filter_spans(span_i_res, text)
        if o != 'miscellaneous':
            pred_ners[o[:3]] = set(filter_ners)
            output_ners[o[:3]] = list(set(filter_ners))
        else:
            pred_ners['other'] = set(filter_ners)
            output_ners['other'] = list(set(filter_ners))
        # output_ners += filter_ners
    overlap, pred, all = calculate_ner_correctness(gold_ners, pred_ners)
    
    return output_ners,overlap,pred, all, prompt

from tqdm import tqdm

def run_qa4span():
    parser = base_parser()
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        args.debug = True
        #args.exp_name += 'debug'
    if args.debug is False:
        from PIL import Image
        import torch
        from lavis.models import load_model_and_preprocess
        # setup device to use
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model, vis_processors, _ = load_model_and_preprocess(name=args.mmodel_type, model_type=args.model_type, is_eval=True, device=device)
        def get_image(img_path):
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except:
                img_path = os.path.dirname(img_path)
                img_path = os.path.join(img_path, 'inf.png')
                raw_image = Image.open(img_path).convert('RGB')
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            return image
    else:
        model,vis_processors, get_image = None, None, None

    def run_exp(args,sub_dataset,get_prompt,config, save_dir, sub_exp_dir):
        exp_dir = save_dir
        sub_exp_dir = exp_dir + sub_exp_dir + '/'
        if not os.path.exists(sub_exp_dir):
            os.makedirs(sub_exp_dir)
        save_output_path =sub_exp_dir + 'output.json'
        save_result_path =sub_exp_dir + 'result.json'
        results = []
        all_gold_num = 0
        all_pred_num = 0
        all_pred_correct = 0

        # read config
        config = load_yaml(config)
        for key, value in config.items():
            if key != 'eval_params' and type(value) == list:
                assert len(value) == 1, 'key {} has more than one value'.format(key)
                config[key] = value[0]
        
        
        for data in tqdm(sub_dataset):
            text = data['text']
            golden_ners = data['spans']
            img_path = data['img_path']
            if args.debug is False:
                image = get_image(img_path)
                pred_spans, overlap,pred, all, prompt = get_prompt(model, text, config,image,golden_ners)
            else:
                image = None
                pred_spans, overlap,pred, all, prompt = get_prompt(model,text, config, image,golden_ners)
            data['prompt'] = prompt
            data['pred_set'] = pred_spans
            data['overlap'] = overlap
            data['pred_num'] = pred
            data['gold_num'] = all
            data['gold_set'] = list(data['gold_set'])
            all_pred_correct += data['overlap']
            all_pred_num += data['pred_num']
            all_gold_num += data['gold_num']
            results.append(data)
        save_json(save_output_path ,results)
        all_result = {}
        all_result['precision'] = all_pred_correct / all_pred_num
        all_result['recall'] = all_pred_correct / all_gold_num
        if all_result['precision'] + all_result['recall'] == 0:
            all_result['f1'] = 0
        else:
            all_result['f1'] = 2*all_result['precision']*all_result['recall'] / (all_result['precision']+all_result['recall'])
        save_json(save_result_path,all_result)

    sub_dataset = get_ner15_data()
    save_dir = make_exp_dir('ner15_span', args.exp_name)
    get_prompt = get_qa4re_spans2_named_mention
    configs = args.configs
    for config in configs:
        sub_exp_dir = 'paper_' + config  +'_/'
        run_exp(args,sub_dataset,get_prompt,config, save_dir, sub_exp_dir)

if __name__ == '__main__':
    run_qa4span()