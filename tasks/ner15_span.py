import os
from tqdm import tqdm
from my_utils import *
from read_dataset import *
import os

import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
def calculate_span_correctness(gold_span, pred_span):
    gold_span1 = set([i.lower() for i in gold_span])
    pred_span1 = set([i.lower() for i in pred_span])
    and_span = gold_span1  & pred_span1
    return len(and_span), len(pred_span1), len(gold_span1)

def get_qa4re_spans_named_mention(model, text, image, gold_span):
    prompt0 = 'Please list all named entity mentions in the sentence that fit the {} category. Answer format is "word1, word2, word3" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    O = ["location", "person", "organization", "miscellaneous"]
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt }, max_length= 10)[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip().lower())
    
    span_cands = filter_spans(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

def run_qa4span():
    parser = base_parser()
    args = parser.parse_args()
    import torch

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

    def run_exp(args,sub_dataset,get_prompt, save_dir, sub_exp_dir):
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
        for data in tqdm(sub_dataset):
            text = data['text']
            golden_spans = data['gold_set']
            img_path = data['img_path']
            if args.debug is False:
                image = get_image(img_path)
                pred_spans, overlap,pred, all, prompt = get_prompt(model,text,image,golden_spans)
            else:
                image = None
                pred_spans, overlap,pred, all, prompt = get_prompt(model,text,image,golden_spans)
            data['prompt'] = prompt
            data['pred_set'] = select_spans(list(pred_spans), text)
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
        all_result['f1'] = 2*all_result['precision']*all_result['recall'] / (all_result['precision']+all_result['recall'])
        save_json(save_result_path,all_result)

    sub_dataset = get_ner15_data(args.num_limit)
    save_dir = make_exp_dir('ner15_span', args.exp_name)
    prompts = [get_qa4re_spans_named_mention]
    for get_prompt in prompts:
        sub_exp_dir = 'paper'
        run_exp(args,sub_dataset,get_prompt, save_dir, sub_exp_dir)

if __name__ == '__main__':
    run_qa4span()