from my_utils import *
from read_dataset import *
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


def calculate_span_correctness(gold_span, pred_span):
    gold_span1 = set([i.lower() for i in gold_span])
    pred_span1 = set([i.lower() for i in pred_span])
    and_span = gold_span1 & pred_span1
    return len(and_span), len(pred_span1), min(1, len(gold_span1))


def choose_topn(choices, topn):
    topn_choices = []
    for choice in choices:
        if choice[0].upper() not in topn_choices and choice[0].isalpha():
            topn_choices.append(choice[0].upper())
        if len(topn_choices) == topn:
            return topn_choices
    return topn_choices


# 0803 finding
def get_most_proba_event_chatgpt_def(model, text, image, topn=1):
    O = ["A. Activities involving the movement or transportation of people or goods from one place to another",
         "B. Interactions between individuals through phone calls or written communication",
         "C. Aggressive actions or assaults by one party against another.",
         "D. Instances where individuals physically meet or come into contact with each other",
         "E. Incidents involving the arrest and subsequent detention in jail or custody of individuals",
         "F. Public displays of disagreement or protest to express opinions or demands",
         "G. The life of a person ends",
         "H. The exchange of money or financial resources between parties"]
    O0 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact",
          "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
    label2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    ret_choices = []
    if image is not None:
        # construct prompt
        prompt = 'Determine which option can be inferred from the given Sentence. Sentence:{}.\n Which option can be inferred from the given Sentence? \n Options:\n {},'
        option_prompt = '\n'.join(O)
        prompt = prompt.format(text, option_prompt)
        pred_choices = model.generate({"image": image, "prompt": prompt}, num_captions=5)
        pred_choices = choose_topn(pred_choices, topn)
        for pred_choice in pred_choices:
            ret_choices.append(O0[label2index[pred_choice]])
        return ret_choices
    else:
        content = O0[: topn]
        return content


def get_most_proba_event_my_def(model, text, image, topn=1):
    prompt = '''Identify which option best summarize the event occured in the sentence.

    Sentence: "{}"

    Options:
    A. Move from a place to the another
    B. Have a meeting
    C. Phone or tax someone
    D. Conflict but not death
    E. Demonstrate
    F. Arrest or Be caught in prison
    G. Death including kill someone or be killed
    H. Tranfer money

    Which option can be inferred from the given sentence?
    Option:'''
    O0 = ["Transport of Movement", "Meet of Contact", "PhoneWrite of Contact", "Attack of Conflict",
          "Demonstrate of Conflict", "Arrest-Jail of Justice", "Die of Life", "Transfer-Money of Transaction"]
    label2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    ret_choices = []
    if image is not None:
        # construct prompt
        prompt = prompt.format(text)
        cnt = 0
        while True:
            try:
                pred_choices = model.generate({"image": image, "prompt": prompt}, num_captions=5 - cnt, max_length=4)
                break
            except Exception as e:
                cnt += 1
                print(e)
                continue

        print('pred_choices', pred_choices)
        pred_choices = choose_topn(pred_choices, topn)
        for pred_choice in pred_choices:
            if pred_choice in label2index:
                ret_choices.append(O0[label2index[pred_choice]])
            else:
                print('fuck output: ', pred_choice)
        return ret_choices
    else:
        content = O0[:topn]
        return content


global_correct, global_all = 0, 0  # 初始化全局变量


# 更换一个prompt
def get_prompt4ee(args, model, text, image, gold_span, O, gold_type, topn):
    O4_to_O1 = {'Movement:Transport': 'Transport of Movement', 'Contact:Phone-Write': 'PhoneWrite of Contact',
                'Conflict:Attack': 'Attack of Conflict', 'Contact:Meet': 'Meet of Contact',
                'Justice:Arrest-Jail': 'Arrest-Jail of Justice', 'Conflict:Demonstrate': 'Demonstrate of Conflict',
                'Life:Die': 'Die of Life', 'Transaction:Transfer-Money': 'Transfer-Money of Transaction',
                'None': 'None'}
    global global_all, global_correct
    global_all += 1
    span_cands = []
    if args.first_choice == 0:
        o_types = get_most_proba_event_chatgpt_def(model, text, image, topn)
    elif args.first_choice == 1:
        o_types = get_most_proba_event_my_def(model, text, image, topn)
    for o_type in o_types:
        o_type = o_type
        # 进行第二阶段抽取
        if 'instruct' in args.mmodel_type:
            prompt = "Please Choose the key word from the verbs and nouns in the sentence that tyriggers the ({}) activit. Note that words can only be noun or verb.  Answer format is \"word1\" \nSentence: {}\n\nAnswer :"
        else:
            # prompt = "Please Choose the most possible word from the verbs and nouns in the sentence that triggers the ({}) activity. Note that words can only be noun or verb.  Answer format is \"word1\" \nSentence: {}\n\nAnswer :"
            prompt = "Please choose the most possible trigger word from the verbs and nouns in the sentence that reflect the {} event. Note that trigger words can only be noun or verb.  Answer format is \"word1\" \nSentence: {}\n\nAnswer :"
        o_prompt = prompt.format(o_type, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt}, max_length=10)[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        if O4_to_O1[gold_type] == o_type:
            global_correct += 1
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)

    return pred_span, overlap, pred, all, prompt, span_i, o_type, O4_to_O1[gold_type]


from tqdm import tqdm


def run_qa4span():
    parser = base_parser()
    parser.add_argument('--first_choice', type=int, default=0, help='choosing the prompt for the first stage')
    parser.add_argument('--topn', type=int, default=2, help='top n choices for the first stage')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        args.debug = True
        # args.exp_name += 'debug'
    if args.debug is False:
        from PIL import Image
        import torch
        from lavis.models import load_model_and_preprocess
        # setup device to use
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model, vis_processors, _ = load_model_and_preprocess(name=args.mmodel_type, model_type=args.model_type,
                                                             is_eval=True, device=device)

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
        model, vis_processors, get_image = None, None, None

    def run_exp(args, sub_dataset, get_prompt, save_dir, sub_exp_dir, O, topn):
        exp_dir = save_dir
        sub_exp_dir = exp_dir + sub_exp_dir + '/'
        if not os.path.exists(sub_exp_dir):
            os.makedirs(sub_exp_dir)
        save_output_path = sub_exp_dir + 'output.json'
        save_result_path = sub_exp_dir + 'result.json'
        save_typedict_path = sub_exp_dir + 'typedict.json'
        results = []
        all_gold_num = 0
        all_pred_num = 0
        all_pred_correct = 0
        # calculate the predict, correct, all of each event type
        # calculate for each event type, what are the top frequent words
        # word_freq_dict = {key: {} for key in O}
        O1 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact",
              "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
        word_freq_dict = {key: {} for key in O1}
        type_num_dict = {key: {'pred_num': 0, 'correct': 0, 'gold_num': 0} for key in O1}
        # import pdb
        # pdb.set_trace()
        for data in tqdm(sub_dataset):
            text = data['text']
            golden_spans = data['gold_set']
            gold_type = data['spans'][0]['type']
            if gold_type not in word_freq_dict:
                word_freq_dict[gold_type] = {}
            if data['spans'][0]['span'] not in word_freq_dict[gold_type]:
                word_freq_dict[gold_type][data['spans'][0]['span']] = 0
            word_freq_dict[gold_type][data['spans'][0]['span']] += 1

            img_path = data['img_path']
            if args.debug is False:
                image = get_image(img_path)
                pred_spans, overlap, pred, all, prompt, span_i, o_type, gold_type = get_prompt(args, model, text, image,
                                                                                               golden_spans, O,
                                                                                               gold_type, topn)
            else:
                image = None
                pred_spans, overlap, pred, all, prompt, span_i, o_type, gold_type = get_prompt(args, model, text, image,
                                                                                               golden_spans, O,
                                                                                               gold_type, topn)

            data['prompt'] = prompt
            data['pred_set'] = list(pred_spans)
            data['overlap'] = overlap
            data['pred_num'] = pred
            data['gold_num'] = all
            data['model_output'] = span_i
            data['model_predict_type'] = o_type
            data['gold_set'] = list(data['gold_set'])
            all_pred_correct += data['overlap']
            all_pred_num += data['pred_num']
            all_gold_num += data['gold_num']
            results.append(data)
        '''
        type_prf = {key: {'precision':0, 'recall':0, 'f1':0} for key in O1}
        for key in type_num_dict:
            type_prf[key]['precision'] = type_num_dict[key]['correct'] / type_num_dict[key]['pred_num']
            type_prf[key]['recall'] = type_num_dict[key]['correct'] / type_num_dict[key]['gold_num']
            type_prf[key]['f1'] = 2*type_prf[key]['precision']*type_prf[key]['recall'] / (type_prf[key]['precision']+type_prf[key]['recall'])
            print(key, type_num_dict[key])
        '''
        for event_type in word_freq_dict:
            word_freq_dict[event_type] = sorted(word_freq_dict[event_type].items(), key=lambda x: x[1], reverse=True)

        global global_correct, global_all
        event_type_precision = global_correct / global_all
        type_num_dict['event_type_precision'] = event_type_precision
        print('event_type_precision', event_type_precision)
        global_correct, global_all = 0, 0
        save_json(save_output_path, results)
        all_result = {}
        all_result['precision'] = all_pred_correct / all_pred_num
        all_result['recall'] = all_pred_correct / all_gold_num
        all_result['f1'] = 2 * all_result['precision'] * all_result['recall'] / (
                all_result['precision'] + all_result['recall'])
        print(all_result)
        save_json(save_result_path, all_result)
        save_json(save_typedict_path, type_num_dict)

    sub_dataset = get_text_ee_data(args.num_limit)
    save_dir = make_exp_dir('tee_span', args.exp_name)
    O1 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact",
          "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]

    sub_exp_dir = 'paper/'
    run_exp(args, sub_dataset, get_prompt4ee, save_dir, sub_exp_dir, O1, args.topn)


if __name__ == '__main__':
    run_qa4span()
