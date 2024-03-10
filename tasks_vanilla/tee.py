import os
from my_utils import *
from read_dataset import *
import os
from ee_prompt import *
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
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

# base
def get_qa4re_spans(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Please list all event trigger words in the sentence that reflect the {} event. Answer format is "word1, word2, word3" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

# base + 描述变成event trigger words
def get_qa4re_spans1(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Please list all event trigger words in the sentence that reflect the {} event. Note that only noun or verb can be trigger words. Answer format is "word1, word2, word3" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

# base + trigger can only be verbs or nouns
def get_qa4re_spans2(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Please list all trigger words in the sentence that reflect the {} event. Note that trigger words can only be noun or verb.  Answer format is "word1, word2, word3" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

# 此前最好的结果 f1 47
# output_new/tee_span/2023-07-29/tee_span_pred_3/get_qa4re_spans3_O_1
def get_qa4re_spans3(model, text, image, gold_span, O,gold_type, topn):
    #('qa2')
    
    prompt0 = 'Please choose the most possible trigger word from the verbs and nouns which reflect the {} event. Note that trigger words can only be noun or verb. Answer format is "word1" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt


def get_most_proba_event(model,text, image):
    O = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
    if image is not None:
        # construct prompt
        good_prompt = 'Which event does the sentence most reflect? Event type candidates are {}'
        prompt = good_prompt + text
        prompt = prompt.format(str(O))
        content = model.generate({"image": image, "prompt": prompt})[0]
        if content not in O:
            # 如果不在则默认返回第一个
            return O[0]
        else:
            return content
    else:
        content = O[0]
        return content
    
def get_most_proba_event_exp(model,text, image):
    O = ["Transport of Movement : whenever an ARTIFACT or a PERSON is moved from one PLACE to another",
         "PhoneWrite of Contact : written or telephone communication where at least two parties are specified", 
         "Attack of Conflict : a violent physical act causing harm or damage", 
         "Meet of Contact : two or more Entities come together at a single location and interact with one another face-to-face", 
         "Arrest-Jail of Justice : the movement of a PERSON is constrained by a state actor",
         "Demonstrate of Conflict : a large number of people come together in a public area to protest or demand some sort of official action",
         "Die of Life : the life of a PERSON Entity ends",
         "Transfer-Money of Transaction : giving, receiving, borrowing, or lending money when it is not in the context of purchasing something"]
    O0 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]

    if image is not None:
        # construct prompt
        good_prompt = 'Which event does the sentence most reflect? Event types and the definitions are {},'
        prompt = good_prompt + text
        prompt = prompt.format(str(O))
        content = model.generate({"image": image, "prompt": prompt})[0]
        o_type = content.split(':')[0].strip()
        if o_type not in O0:
            return O0
        return o_type

    else:
        content = O0[0]
        return content

def get_most_proba_event_choice(model,text, image):
    O = ["A. An ARTIFACT or a PERSON is moved from one PLACE to another",
         "B. Written or telephone communication where at least two parties are specified", 
         "C. A violent physical act causing harm or damage", 
         "D. Two or more Entities come together at a single location and interact with one another face-to-face", 
         "E. The movement of a PERSON is constrained by a state actor",
         "F. A large number of people come together in a public area to protest or demand some sort of official action, such as  protests, sit-ins, strikes, and riots",
         "G. The life of a PERSON Entity ends",
         "H. Giving, receiving, borrowing, or lending money when it is not in the context of purchasing something"]
    O0 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
    label2index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5,'G':6,'H':7}
    if image is not None:
        # construct prompt
        prompt = 'Determine which option is most relevant to the given sentence. Sentence:{}.\n Which option is most relevant to the given sentence? \n Options:\n {},'
        option_prompt = '\n'.join(O)
        prompt = prompt.format(text, option_prompt)
        content = model.generate({"image": image, "prompt": prompt})[0]
        o_type = O0[label2index[content[0]]]
        if o_type not in O0:
            return O0
        return o_type

    else:
        content = O0[0]
        return content

def choose_topn(choices, topn):
    topn_choices = []
    for choice in choices:
        if choice[0].upper() not in topn_choices and choice[0].isalpha():
            topn_choices.append(choice[0].upper())
        if len(topn_choices) == topn:
            return topn_choices
    return topn_choices

# 0803 finding 
def get_most_proba_event_chatgpt_def(model,text, image, topn=1):
    O = ["A. Activities involving the movement or transportation of people or goods from one place to another",
         "B. Interactions between individuals through phone calls or written communication", 
         "C. Aggressive actions or assaults by one party against another.", 
         "D. Instances where individuals physically meet or come into contact with each other", 
         "E. Incidents involving the arrest and subsequent detention in jail or custody of individuals",
         "F. Public displays of disagreement or protest to express opinions or demands",
         "G. The life of a person ends",
         "H. The exchange of money or financial resources between parties"]
    O0 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
    label2index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5,'G':6,'H':7}
    ret_choices = []
    if image is not None:
        # construct prompt
        prompt = 'Determine which option can be inferred from the given Sentence. Sentence:{}.\n Which option can be inferred from the given Sentence? \n Options:\n {},'
        option_prompt = '\n'.join(O)
        prompt = prompt.format(text, option_prompt)
        pred_choices = model.generate({"image": image, "prompt": prompt},num_captions=5)
        pred_choices = choose_topn(pred_choices, topn)
        for pred_choice in pred_choices:
            ret_choices.append(O0[label2index[pred_choice]])
        return ret_choices
    else:
        content = O0[: topn]
        return content
    
def get_most_proba_event_my_def(model,text, image, topn=1):
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
    O0 = ["Transport of Movement", "Meet of Contact", "PhoneWrite of Contact", "Attack of Conflict",  "Demonstrate of Conflict",  "Arrest-Jail of Justice", "Die of Life", "Transfer-Money of Transaction"]
    label2index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5,'G':6,'H':7}
    ret_choices = []
    if image is not None:
        # construct prompt
        prompt = prompt.format(text)
        #print('event type prompt:', prompt)
        import pdb
        pdb.set_trace()
        pred_choices = model.generate({"image": image, "prompt": prompt},num_captions=5, max_length=10)
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

def get_most_proba_event_exp_current(model,text, image):
    O = ["Transport of Movement : whenever an ARTIFACT or a PERSON is moved from one PLACE to another",
         "PhoneWrite of Contact : written or telephone communication where at least two parties are specified", 
         "Attack of Conflict : a violent physical act causing harm or damage", 
         "Meet of Contact : two or more Entities come together at a single location and interact with one another face-to-face", 
         "Arrest-Jail of Justice : the movement of a PERSON is constrained by a state actor",
         "Demonstrate of Conflict : a large number of people come together in a public area to protest or demand some sort of official action",
         "Die of Life : the life of a PERSON Entity ends",
         "Transfer-Money of Transaction : giving, receiving, borrowing, or lending money when it is not in the context of purchasing something"]
    O0 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]

    if image is not None:
        # construct prompt
        good_prompt = 'Which event does the sentence currently reflect? Event types and the definitions are {},'
        prompt = good_prompt + text
        prompt = prompt.format(str(O))
        content = model.generate({"image": image, "prompt": prompt})[0]
        if content not in O0:
            # 如果不在则默认返回第一个
            return O[0]
        else:
            return content
    else:
        content = O[0]
        return content
            

global_correct = 0
global_all = 0


# a best prompt for t5 xxl
# 'What is the most possible event type in "Syrian Aya Sharqawi, 6, wounded in an airstrike 10 days ago in her hometown of Tel Rifaat, Syria, lays on her bed at a hospital in Kilis, Turkey, Feb. 9, 2016"? Please choose the event type from  ['Movement Transport', 'Phone-Write', 'Attack', 'Meeting', 'Justice Arrest-Jail', 'demonstration', 'Life Die', 'Transaction Transfer-Money', 'None'].\nAnswer:'

#Please list the trigger word in the sentence for the Meeting event. Answer format is "word1, word2, word3" 
#Sentence: "North Korea against world-U.S. President Barack Obama, South Korean President Park Geun-hye and Japanese Prime Minister Shinzo Abe met at the Nuclear Security Summit in Washington on Thursday to underscore their united commitment to exert increasing pressure on North Korea to abandon its nuclear program."
#Answer :
#

# 判断事件类型的准确率达到80%的话，其实瓶颈还是在抽词这块。
def get_qa4re_spans_cmd_best(model, text, image, gold_span, O, gold_type):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    global global_all, global_correct

    global_all += 1
    #('qa2')
    prompt0 = '''Here are the example key words prompt each action type: 
    "Conflict.Demonstrate": rally, protest, march, demonstration \n
    "Justice.Arrest-Jail": arrest, detain, caught \n
    "Movement.Transport": arrive, fled, carry, reach \n
    "Contact.Meet": meet, conference \n
    "Life.Die": kill,die \n
    "Conflict.Attack": attack, shoot, fight, airstrike \n
    "Contact.Phone-Write": speak, call, conversation \n
    "Transaction.Transfer-Money": fund, pay, buy \n
    Sentence:  "{}" \n
    Extract all the key nouns and verbs from the given sentence that prompt the key action above :'''

    prompt = prompt0.format(text)
    all_words = text.split()
    span_cands = []
    # 先使用XXX方式抽取出最有可能的类型是什么
    o_type = 'Justice.Arrest-Jail'
    
    if image is not None:
        span_i = model.generate({"image": image, "prompt": prompt })[0]
    else:
        span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
    if O4_to_O1[gold_type] == o_type:
        global_correct += 1
    span_cands += span_i.split(',')

    span_cands_temp = []
    for span_x in span_cands:
        span_x = span_x.strip().lower()
        for word in all_words:
            if span_x in word.lower():
                span_cands_temp.append(word)
                break
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt, span_i, o_type, O4_to_O1[gold_type]

prompt_w_demon = '''
Here are the example key words prompt each action type: 
"Conflict.Demonstrate": rally, protest, march, demonstration
"Justice.Arrest-Jail": arrest, detain, caught
"Movement.Transport": arrive, fled, carry, reach
"Contact.Meet": meet, conference
"Life.Die": kill,die
"Conflict.Attack": attack, shoot, fight, airstrike
"Contact.Phone-Write": speak, call, conversation
"Transaction.Transfer-Money": fund, pay, buy
Extract all the key nouns and verbs from the given sentence that prompt the key action above
Sentence: "Demonstrators rally against Venezuela's President Nicolas Maduro in Caracas, Venezuela, April 13, 2017"
Output => rally,
Sentence: "FILE - A demonstrator is arrested by riot police while rallying against Venezuela's President Nicolas Maduro's government in Caracas, Venezuela, April 10, 2017"
Output => arrested, rallying
Sentence: "FILE - United States Marines complete quarantine checks as they arrive at a Royal Australian Air Force Base in Darwin"
Output => arrive
Sentence: "Turkey's President Recep Tayyip Erdogan addresses a meeting in Ankara, Turkey, Jan. 17, 2017"
Output => meeting
Sentence: "Bodies of people killed during a rally are seen at the capital's main mosque in Conakry, Guinea-Okey Onyejekwe warns that the entire West African region will be destabilized if Guinea becomes a failed state"
Output => killed, rally
Sentence: "They also blame the group for a deadly attack in Kabul in April that killed nearly 70 people"
Output => attack, killed
Sentence: "Democratic U.S. presidential candidate Hillary Clinton is accompanied by her daughter Chelsea Clinton (R) and her husband, former U.S. President Bill Clinton, as she speaks to supporters at her final 2016 New Hampshire presidential primary night rally"
Output => speaks
Sentence: "Moscow turns up pressure on Kyiv-Russian state-controlled gas company Gazprom has reiterated its threat to stop supplying Ukraine with gas if it does not pay in advance for June deliveries, Russian news agencies reported on Monday"
Output => pay
Sentence:"Pistorius Witness Describes 'Terrified' Screaming-Week four of the Oscar Pistorius trial began Monday with a neighbor saying she heard screams from both a woman and a man on the morning the athlete shot and killed girlfriend Reeva Steenkamp"
Output => shot, killed
Sentence: "{}"
Output => 
'''
# 加入了demonstrate的
def get_qa4re_spans_cmd_best1(model, text, image, gold_span, O, gold_type):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    global global_all, global_correct

    global_all += 1
    #('qa2')
    prompt0 = '''
Here are the example key words prompt each action type: 
"Conflict.Demonstrate": rally, protest, march, demonstration
"Justice.Arrest-Jail": arrest, detain, caught
"Movement.Transport": arrive, fled, carry, reach
"Contact.Meet": meet, conference
"Life.Die": kill,die
"Conflict.Attack": attack, shoot, fight, airstrike
"Contact.Phone-Write": speak, call, conversation
"Transaction.Transfer-Money": fund, pay, buy
Extract all the key nouns and verbs from the given sentence that prompt the key action above
Sentence: "Demonstrators rally against Venezuela's President Nicolas Maduro in Caracas, Venezuela, April 13, 2017"
Output => rally,
Sentence: "FILE - A demonstrator is arrested by riot police while rallying against Venezuela's President Nicolas Maduro's government in Caracas, Venezuela, April 10, 2017"
Output => arrested, rallying
Sentence: "FILE - United States Marines complete quarantine checks as they arrive at a Royal Australian Air Force Base in Darwin"
Output => arrive
Sentence: "Turkey's President Recep Tayyip Erdogan addresses a meeting in Ankara, Turkey, Jan. 17, 2017"
Output => meeting
Sentence: "Bodies of people killed during a rally are seen at the capital's main mosque in Conakry, Guinea-Okey Onyejekwe warns that the entire West African region will be destabilized if Guinea becomes a failed state"
Output => killed, rally
Sentence: "They also blame the group for a deadly attack in Kabul in April that killed nearly 70 people"
Output => attack, killed
Sentence: "Democratic U.S. presidential candidate Hillary Clinton is accompanied by her daughter Chelsea Clinton (R) and her husband, former U.S. President Bill Clinton, as she speaks to supporters at her final 2016 New Hampshire presidential primary night rally"
Output => speaks
Sentence: "Moscow turns up pressure on Kyiv-Russian state-controlled gas company Gazprom has reiterated its threat to stop supplying Ukraine with gas if it does not pay in advance for June deliveries, Russian news agencies reported on Monday"
Output => pay
Sentence:"Pistorius Witness Describes 'Terrified' Screaming-Week four of the Oscar Pistorius trial began Monday with a neighbor saying she heard screams from both a woman and a man on the morning the athlete shot and killed girlfriend Reeva Steenkamp"
Output => shot, killed
Sentence: "{}"
Output => 
'''

    prompt = prompt0.format(text)
    all_words = text.split()
    span_cands = []
    # 先使用XXX方式抽取出最有可能的类型是什么
    o_type = 'Justice.Arrest-Jail'
    
    if image is not None:
        span_i = model.generate({"image": image, "prompt": prompt })[0]
    else:
        span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
    if O4_to_O1[gold_type] == o_type:
        global_correct += 1
    span_cands += span_i.split(',')

    span_cands_temp = []
    for span_x in span_cands:
        span_x = span_x.strip().lower()
        for word in all_words:
            if span_x in word.lower():
                span_cands_temp.append(word)
                break
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt, span_i, o_type, O4_to_O1[gold_type]


# 先判断前两种类型，然后再用demonstrate抽span的
# 先为每个类型定义一个demonstrate列表：



# 选top2,3分开做
def get_qa4re_spans_top_label_sepa(args, model, text, image, gold_span, O, gold_type, topn):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    O1_to_O4 = {'Transport of Movement':'Movement:Transport', 'PhoneWrite of Contact':'Contact:Phone-Write', 'Attack of Conflict':'Conflict:Attack', 'Meet of Contact':'Contact:Meet', 'Arrest-Jail of Justice':'Justice:Arrest-Jail', 'Demonstrate of Conflict':'Conflict:Demonstrate', 'Die of Life':'Life:Die', 'Transfer-Money of Transaction':'Transaction:Transfer-Money', 'None':'None'}
    global global_all, global_correct
    global_all += 1
    #('qa2')
    span_cands = []
    # 先使用XXX方式抽取出最有可能的类型是什么
    if args.first_choice == 0:
        o_types = get_most_proba_event_chatgpt_def(model, text, image, topn)
    elif args.first_choice == 1:
        o_types = get_most_proba_event_my_def(model, text, image, topn)
    for o_type in o_types:
        # 进行第二阶段抽取
        prompt = ee_label_prompt + label2demo[O1_to_O4[o_type]] + output_format
        o_prompt = prompt.format(o_type, text)

        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt})[0]
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
    overlap, pred, all = calculate_ner_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt, span_i, o_type, O4_to_O1[gold_type]

# 更换一个prompt
def get_qa4re_spans_top_label_sepa_prompt1(args, model, config, text, image, gold_span, O, gold_type, topn):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    O1_to_O4 = {'Transport of Movement':'Movement:Transport', 'PhoneWrite of Contact':'Contact:Phone-Write', 'Attack of Conflict':'Conflict:Attack', 'Meet of Contact':'Contact:Meet', 'Arrest-Jail of Justice':'Justice:Arrest-Jail', 'Demonstrate of Conflict':'Conflict:Demonstrate', 'Die of Life':'Life:Die', 'Transfer-Money of Transaction':'Transaction:Transfer-Money', 'None':'None'}
    global global_all, global_correct
    global_all += 1
    #('qa2')
    span_cands = {}
    o_types = O4_to_O1.keys()
    task_instructions = config['task_instructions'].strip()
    prompt = task_instructions + '\n\n' +  config['example_format']
    for o_type in o_types:
        span_cands[o_type] = []
        o_type = o_type
        # 进行第二阶段抽取
        o_prompt = prompt.format(o_type, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt}, max_length= 15)[0]
        else:
            span_i = text.split(' ')[0]
        span_cands_temp = []
        for span_x in span_i.split(','):
            span_cands_temp.append(span_x.strip().lower())
        span_cands[o_type] = set(filter_spans_ee(span_cands_temp, text))
    print('pred', span_cands)
    print('gold', gold_span)
    overlap, pred, all = calculate_ner_correctness(gold_span, span_cands)
    pred_span = set()
    return pred_span,overlap,pred, all, prompt, span_i, o_type, O4_to_O1[gold_type]
# 选top2,3 一起做
def get_qa4re_spans_top2_label_joint(args, model, text, image, gold_span, O, gold_type, topn=2):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    O1_to_O4 = {'Transport of Movement':'Movement:Transport', 'PhoneWrite of Contact':'Contact:Phone-Write', 'Attack of Conflict':'Conflict:Attack', 'Meet of Contact':'Contact:Meet', 'Arrest-Jail of Justice':'Justice:Arrest-Jail', 'Demonstrate of Conflict':'Conflict:Demonstrate', 'Die of Life':'Life:Die', 'Transfer-Money of Transaction':'Transaction:Transfer-Money', 'None':'None'}
    global global_all, global_correct
    global_all += 1
    #('qa2')
    span_cands = []
    # 结果表明chatgpt这个效果要差不少，但是为什么最终的span抽取结果一样呢？
    if args.first_choice == 0:
        o_types = get_most_proba_event_chatgpt_def(model, text, image, 2)
    elif args.first_choice == 1:
        o_types = get_most_proba_event_my_def(model, text, image, 2)

    o_type_all = ''
    prompt = ee_label_prompt.strip() + '\n'
    for o_type in o_types:
        o_type_all = o_type_all + o_type + ', '
        prompt = prompt + label2demo[O1_to_O4[o_type]].strip() + '\n'
    prompt = prompt + output_format.strip()

    o_prompt = prompt.format(o_type_all, text)

    if image is not None:
        span_i = model.generate({"image": image, "prompt": o_prompt})[0]
    else:
        span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
    if O4_to_O1[gold_type] in o_type_all:
        global_correct += 1
    span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt, span_i, o_type_all, O4_to_O1[gold_type]

def get_qa4re_spans_top3_label_joint(args, model, text, image, gold_span, O, gold_type, topn=3):
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    O1_to_O4 = {'Transport of Movement':'Movement:Transport', 'PhoneWrite of Contact':'Contact:Phone-Write', 'Attack of Conflict':'Conflict:Attack', 'Meet of Contact':'Contact:Meet', 'Arrest-Jail of Justice':'Justice:Arrest-Jail', 'Demonstrate of Conflict':'Conflict:Demonstrate', 'Die of Life':'Life:Die', 'Transfer-Money of Transaction':'Transaction:Transfer-Money', 'None':'None'}
    global global_all, global_correct
    global_all += 1
    #('qa2')
    span_cands = []
    # 结果表明chatgpt这个效果要差不少，但是为什么最终的span抽取结果一样呢？
    if args.first_choice == 0:
        o_types = get_most_proba_event_chatgpt_def(model, text, image, 3)
    elif args.first_choice == 1:
        o_types = get_most_proba_event_my_def(model, text, image, 3)

    o_type_all = ''
    prompt = ee_label_prompt.strip() + '\n'
    for o_type in o_types:
        o_type_all = o_type_all + o_type + ', '
        prompt = prompt + label2demo[O1_to_O4[o_type]].strip() + '\n'
    prompt = prompt + output_format.strip()

    o_prompt = prompt.format(o_type_all, text)

    if image is not None:
        span_i = model.generate({"image": image, "prompt": o_prompt})[0]
    else:
        span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
    if O4_to_O1[gold_type] in o_type_all:
        global_correct += 1
    span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt, span_i, o_type_all, O4_to_O1[gold_type]

# 此前最好的结果
def get_qa4re_spans3_ace(model, text, image, gold_span, O,gold_type):
    #('qa2')
    
    prompt0 = 'Consider the event definition in ACE 2005 and choose the most possible trigger word from the verbs and nouns which reflect the {} event. Note that trigger words can only be noun or verb. Answer format is "word1" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

# base + trigger can only be verbs or nouns
def get_qa4re_spans4(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Choose the most possible trigger word from the verbs and nouns which reflect the {} event. Answer format is "word1" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

def get_qa4re_spans5(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Choose the most possible word from the verbs and nouns which triggers the {} event. Answer format is "word1" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

def get_qa4re_spans6(model, text, image, gold_span, O,gold_type):
    #('qa2')
    prompt0 = 'Choose the most possible word which triggers the {} event. Note that the word can only be verb or noun. Answer format is "word1" \nSentence: '
    prompt1 = '{}'
    prompt2 = '\n\nAnswer :'
    prompt = prompt0 + prompt1 + prompt2
    span_cands = []
    for o in O:
        o_prompt = prompt.format(o, text)
        if image is not None:
            span_i = model.generate({"image": image, "prompt": o_prompt })[0]
        else:
            span_i = '' if len(gold_span) == 0 else list(gold_span)[0]
        span_cands += span_i.split(',')
    span_cands_temp = []
    for span_x in span_cands:
        span_cands_temp.append(span_x.strip())
    
    span_cands = filter_spans_ee(span_cands_temp, text)
    pred_span = set(span_cands)
    overlap, pred, all = calculate_span_correctness(gold_span, pred_span)
    
    return pred_span,overlap,pred, all, prompt

from tqdm import tqdm

def run_qa4span():
    parser = base_parser()
    parser.add_argument('--first_choice', type=int, default=0, help='choosing the prompt for the first stage')
    parser.add_argument('--topn', type=int, default=2, help='top n choices for the first stage')
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

    def run_exp(args,sub_dataset,get_prompt, config, save_dir, sub_exp_dir, O, topn):
        exp_dir = save_dir
        sub_exp_dir = exp_dir + sub_exp_dir + '/'
        config = load_yaml(config)
        for key, value in config.items():
            if key != 'eval_params' and type(value) == list:
                assert len(value) == 1, 'key {} has more than one value'.format(key)
                config[key] = value[0]
        if not os.path.exists(sub_exp_dir):
            os.makedirs(sub_exp_dir)
        save_output_path =sub_exp_dir + 'output.json'
        save_result_path =sub_exp_dir + 'result.json'
        save_typedict_path =sub_exp_dir + 'typedict.json'
        results = []
        all_gold_num = 0
        all_pred_num = 0
        all_pred_correct = 0
        # calculate the predict, correct, all of each event type
        # calculate for each event type, what are the top frequent words
        #word_freq_dict = {key: {} for key in O}
        O1 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
        word_freq_dict = {key: {} for key in O1}
        type_num_dict = {key: {'pred_num':0, 'correct':0, 'gold_num':0} for key in O1}
        for data in tqdm(sub_dataset):
            text = data['text']
            golden_spans = data['spans']
            gold_type = data['spans'][0]['type']
            if gold_type not in word_freq_dict:
                word_freq_dict[gold_type] = {}
            if data['spans'][0]['span'] not in word_freq_dict[gold_type]:
                word_freq_dict[gold_type][data['spans'][0]['span']] = 0
            word_freq_dict[gold_type][data['spans'][0]['span']] += 1

            img_path = data['img_path']
            if args.debug is False:
                image = get_image(img_path)
                pred_spans, overlap,pred, all, prompt, span_i, o_type, gold_type = get_prompt(args, model,config, text,image,golden_spans, O, gold_type, topn)
            else:
                image = None
                pred_spans, overlap,pred, all, prompt, span_i, o_type, gold_type = get_prompt(args, model,config, text,image,golden_spans, O, gold_type, topn)
            
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
        all_result = {}
        all_result['precision'] = all_pred_correct / all_pred_num
        all_result['recall'] = all_pred_correct / all_gold_num
        all_result['f1'] = 2*all_result['precision']*all_result['recall'] / (all_result['precision']+all_result['recall'])
        print(all_result)
        save_json(save_result_path,all_result)
        save_json(save_typedict_path, type_num_dict)

    
    sub_dataset = get_text_ee_data(args.num_limit)
    save_dir = make_exp_dir('tee_span', args.exp_name)
    O1 = ["Transport of Movement", "PhoneWrite of Contact", "Attack of Conflict", "Meet of Contact", "Arrest-Jail of Justice", "Demonstrate of Conflict", "Die of Life", "Transfer-Money of Transaction"]
    
    O2 = ["Transport", "PhoneWrite", "Attack", "Meet", "Arrest-Jail", "Demonstrate", "Die", "Transfer-Money"]
    O3 = ["Movement", "Contact", "Conflict", "Justice", "Life", "Transaction"]
    O4 = ['Movement:Transport', 'Contact:Phone-Write', 'Conflict:Attack', 'Contact:Meet', 'Justice:Arrest-Jail', 'Conflict:Demonstrate', 'Life:Die', 'Transaction:Transfer-Money', 'None']
    # create a mapping dict from O4 to O1
    O4_to_O1 = {'Movement:Transport':'Transport of Movement', 'Contact:Phone-Write':'PhoneWrite of Contact', 'Conflict:Attack':'Attack of Conflict', 'Contact:Meet':'Meet of Contact', 'Justice:Arrest-Jail':'Arrest-Jail of Justice', 'Conflict:Demonstrate':'Demonstrate of Conflict', 'Life:Die':'Die of Life', 'Transaction:Transfer-Money':'Transfer-Money of Transaction', 'None':'None'}
    Os = [O1]
    configs = args.configs
    for config in configs:
        prompts = [get_qa4re_spans_top_label_sepa_prompt1]
        for get_prompt in prompts:
            for i, O in enumerate(Os):
                sub_exp_dir = 'paper_' + config  +'_/'
                run_exp(args,sub_dataset,get_prompt,config, save_dir, sub_exp_dir, O, args.topn)

if __name__ == '__main__':
    run_qa4span()