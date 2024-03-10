# read result file from json file
import json
import yaml
import random
import numpy as np
from argparse import ArgumentParser

def read_json(filename):
    f = open(filename, 'r', encoding='utf-8')
    tem = f.read()
    f_dict = json.loads(tem)
    f.close()
    return f_dict

def save_json(filename,ds):
    f = open(filename, 'w', encoding='utf-8')
    json_str1 = json.dumps(ds, indent=4)
    f.write(json_str1+'\n')
    f.close()

def load_yaml(config):
    path = './configs_vanilla/' + config
    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict

#  ./output_new/
def make_exp_dir(task=None,exp_name=''):
# Get the save dir of current experiment, save dir logic is output/task/day/experiment name_1,2,3,4.....
    task_dir = './output/'
    exp_dir =  task_dir + exp_name + '/'+ task + '/'
    return exp_dir

def get_prediction(content,all_choices):
    def get_pred_from_content(generated_content, verbalized_labels):
        """get the prediction from the generated content, for multiple token generation and chatCompletion"""
        candidates = []
        for label in verbalized_labels:
            if label in generated_content:
                candidates.append(label)
            if len(candidates) == 0:  # not generated, randomly choose one.
                pred = random.choice(verbalized_labels)
            elif len(candidates) > 1:  # get the first one in generation order
                start_indexes = [generated_content.index(can) for can in candidates]
                pred = candidates[np.argmin(start_indexes)]
            else:
                pred = candidates[0]
        return pred
    pred = get_pred_from_content(content, all_choices)
    return pred

def save_args(args, path_dir):
    #针对argparser的参数进行保存的函数
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

# back processing for span extraction
## strategy 1: span that are not in the text
## strategy 2: spans that are within other spans
# return lower cased filtered span list
def filter_spans(span_list, text):
    # filter the span in span_list with the above two strategies
    first_filter = []
    #lower case the text
    text = text.lower()
    for span in span_list:
        if span.lower() not in text:
            continue
        else:
            first_filter.append(span)
    # second filter
    second_filter = []
    for span in first_filter:
        flag = True
        for other_span in first_filter:
            if span.lower() == other_span.lower():
                continue
            if span.lower() in other_span.lower():
                flag = False
                break
        if flag:
            second_filter.append(span)
    return second_filter

import nltk
from nltk import PorterStemmer
porter = PorterStemmer()


def filter_spans_ee(span_list, text):
    # filter the span in span_list with the above two strategies
    zero_filter = []
    first_filter = []
    #lower case the text
    text = text.lower()
    text_words = text.split()
    text_words = set([word.strip() for word in text_words])
    for span in span_list:
        # 过滤有多个span的词
        if len(span.split()) > 1:
            continue
        else:
            zero_filter.append(span)
    
    for span in zero_filter:
        if span.lower() not in text:
            flag0 = True
            for word in text_words:
                if porter.stem(span.lower()) == porter.stem(word):
                    first_filter.append(span)
                    flag0 = False
                    break
            if flag0 == True: # 不在文本中，那取一个前缀从2到span的长度，从大到小
                for span_prefix_len in range(len(span)-1, 2, -1):
                    break_flag = False
                    for word in text_words:
                        if span[:span_prefix_len].lower() == word[:span_prefix_len]:
                            first_filter.append(span[:span_prefix_len])
                            break_flag = True
                            break
                    if break_flag == True:
                        break
        else:
            first_filter.append(span)
    # second filter
    second_filter = []
    for span in first_filter:
        flag = True
        for other_span in first_filter:
            if span.lower() == other_span.lower():
                continue
            if porter.stem(span.lower()) == porter.stem(other_span.lower()):
                flag = False
                break
        if flag:
            second_filter.append(span)

    return second_filter


def base_parser():
    parser = ArgumentParser()
    parser.add_argument('--configs', nargs='+', help='YAML configuration files')
    parser.add_argument('--label_template', type=int, default=1)
    parser.add_argument('--num_limit', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exp_name', type=str,default = '',help='The name of the experiment, if the dir exist already, name it with additional number.')
    parser.add_argument('--model_type', type=str,default = "pretrain_flant5xxl")
    parser.add_argument('--mmodel_type', type=str,default = "blip2_t5")
    return parser

def get_today_str():
    import datetime

    # Get the current date
    current_date = datetime.datetime.now().date()

    # Convert the date to a string
    current_date_str = current_date.strftime("%Y-%m-%d")

    return current_date_str
    

if __name__ == '__main__':
    def a():
        return 1
    b = 1
    print(b.__name__)