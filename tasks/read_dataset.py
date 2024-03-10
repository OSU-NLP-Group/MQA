# NER reader
# read span dataset
# 参考diffusionner里的加载部分
# 对于span，应该有n条数据，每条数据中国，有一个set，包含所有的span，还有img id
# eval的时候，就是
import json
import os
from my_utils import read_json
# 读取 JSON 数据
def get_ner15_data(limit_num):
    img_dir = './dataset/mner/twitter2015_images/'
    with open('./dataset/mner/twitter2015/test.json', 'r') as file:
        test_data0 = json.load(file)
    test_data = []
    for i, data in enumerate(test_data0):
        if i > limit_num:
            break
        text = ' '.join(data['tokens'])
        spans = []
        spans_set = []
        for span0 in data['entities']:
            start = span0['start']
            end = span0['end']
            type =  span0['type']
            span = ' '.join(data['tokens'][start: end])
            spans.append({'span':span, 'type':type})
            spans_set.append(span)
        img_path = img_dir + str(data['img_id']) + '.jpg'
        test_data.append({'text':text, 'spans':spans, 'gold_set': set(spans_set) , 'img_path':img_path})

    return test_data

def get_ner17_data(limit_num):
    img_dir = './dataset/mner/twitter2017_images/'
    with open('./dataset/mner/twitter2017/test.json', 'r') as file:
    # with open('/mnt/Xsky/syx/project/2023/MMUIE/dataset/mner/twitter2017/train_our_50.json', 'r') as file:
        test_data0 = json.load(file)
    test_data = []
    for i, data in enumerate(test_data0):
        if i > limit_num:
            break
        text = ' '.join(data['tokens'])
        spans = []
        spans_set = []
        for span0 in data['entities']:
            start = span0['start']
            end = span0['end']
            type =  span0['type']
            span = ' '.join(data['tokens'][start: end])
            spans.append({'span':span, 'type':type})
            spans_set.append(span)
        img_path = img_dir + str(data['img_id']) + '.jpg'
        test_data.append({'text':text, 'spans':spans, 'gold_set': set(spans_set) , 'img_path':img_path})

    return test_data

def get_ner_pred_data(limit_num,pred_file):
    with open(pred_file, 'r') as file:
        test_data0 = json.load(file)
    test_data = []
    for i, data in enumerate(test_data0):
        if i > limit_num:
            break
        text = data["text"]
        gold_spans_w_type = data["spans"]
        pred_spans = data["pred_set"]
        img_path = data["img_path"]
        gold_spans = data["gold_set"]

        test_data.append({'text':text, 'pred_spans':pred_spans, 'gold_spans': gold_spans, 'gold_spans_w_type': gold_spans_w_type, 'img_path':img_path})

    return test_data


def get_re_data(sub_num,version ):
    test_data = []
    if '15' in version:
        load_file = './dataset/mre/mre_v1/txt/ours_test.txt'
        img_dir = './dataset/mre/mre_v1/imgs/'
        total_num = 1200
    else:
        load_file = './dataset/mre/mre_v2/txt/ours_test.txt'
        img_dir = './dataset/mre/mre_v2/imgs/'
        total_num = 1614
    rate = sub_num / total_num
    dd = {}
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = eval(line)   # str to dict
            words = line['token']
            text = ' '.join(words)
            relation = line['relation']
            head = line['h']['name']
            tail = line['t']['name']
            img_path = img_dir + line['img_id']
            if relation not in dd:
                dd[relation] = []
            dd[relation].append({'text':text, 'relation': relation, 'head': head, 'tail': tail, 'img_path':img_path})
        
    for rel in dd:
        idx = max(int(len(dd[rel]) * rate), 1)
        test_data += dd[rel][:idx]
    return test_data

def get_re_data_type(sub_num,version):
    
    if '15' in version:
        org_data = read_json('./dataset/mre/mre_v1/txt/test_type.json')
        total_num = 1200
    else:
        org_data = read_json('./dataset/mre/mre_v2/txt/test_type.json')
        total_num = 1614
    test_data = []
    rate = sub_num / total_num
    dd = {}
    for i, line in enumerate(org_data):
        relation = line['relation']
        if relation not in dd:
            dd[relation] = []
        dd[relation].append(line)
        
    for rel in dd:
        idx = max(int(len(dd[rel]) * rate), 1)
        test_data += dd[rel][:idx]
    return test_data

# read text event detection data

def get_text_ee_data(num_limit):
    # import pdb
    # pdb.set_trace()
    # read from dataset/mee/annotations/text_only_event.json
    img_dir = './dataset/mee/raw_data/image/image/'
    test_data = []
    text_event_num = 0
    count1 = 0
    count2 = 0
    for url in ['./dataset/mee/annotations/text_only_event.json', './dataset/mee/annotations/text_multimedia_event.json']:
        origin_data = read_json(url)
        for i, data in enumerate(origin_data):
            if i > num_limit:
                break
            text = data["sentence"]
            gold_event = data["golden-event-mentions"]
            text_event_num += len(gold_event)
            if len(gold_event) == 0:
                continue
            spans_set = []
            spans = []
            types = []
            count1 += 1
            for event in gold_event:
                types.append(event["event_type"])
                spans_set.append(event["trigger"]["text"])
                spans.append({'span':event["trigger"]["text"], 'type':event["event_type"]})
            if len(types) != len(set(types)):
                count2 += 1
            images = data["image"]
            # TODO: 这里简化处理，只使用第一张图片
            for image in images:
                img_path = img_dir + '/' + image
                if os.path.exists(img_path):
                    break
            test_data.append({'text':text,'spans':spans, 'gold_set': set(spans_set), 'img_path':img_path})
    #print(len(test_data))
    #print(text_event_num)
    print('total sentence', count1)
    print('sentence with same event type', count2)
    return test_data

# 这个不太对，image_only_event这里是只有golden的结果，那召回率就和准确率一样了,如果是image，还是应该是读取全部图片，
def get_image_ee_data(num_limit):

    # read from dataset/mee/annotations/text_only_event.json
    fpath = './dataset/mee/annotations/image_only_event.json'
    origin_data = read_json(fpath)
    test_data = []
    for i, data in enumerate(origin_data):
        if i > num_limit:
            break
        text = data["sentence"]
        gold_entity = data["golden-entity-mentions"] # useless yet
        gold_event = data["golden-event-mentions"]
        images = data["images"]
        test_data.append({'text':text, "gold_entity": gold_entity,'gold_event':gold_event, 'images':images})

    return test_data

# 三种image only setting，
# 1. 使用文本Only, 文本Multimedia还有Image Only的图片作为候选图片；默认输入是图片和其caption

# 2. 使用全部图片， 默认输入是图片和其caption

# 3. 使用全部图片，文本端什么都不输入

event_type_dict = {'Movement:Transport', 'Contact.PhoneWrite', 'Conflict:Attack', 'Contact:Meet', 'Justice:Arrest-Jail', 'Conflict:Demonstrate', 'Life:Die', 'Transaction:Transfer-Money'}


# 这个基本可以不用跑了，效果很差。
def get_img_only_w_text(num_limit, constrained=False):
    core_imgs = get_core_imgs()
    img2text = get_img2text()
    img_url = './dataset/mee/raw_data/image/image_url_caption.json'
    img_dir = './dataset/mee/raw_data/image/image/'
    img_caption = read_json(img_url)
    img_anno = read_img_annotations()
    test_data = []
    count = 0
    
    for img_outer, img_inners in img_caption.items():
        for img_inner, caption in img_inners.items():
            count += 1
            if count > num_limit:
                break
            
            each_img_path = img_dir + img_outer + '_' + img_inner + '.jpg'
            each_img_name = img_outer + '_' + img_inner
            each_img_fname = each_img_name + '.jpg'
            if each_img_fname not in core_imgs or constrained:
                continue
            
            label = img_anno.get(each_img_name, {'event_type':'None'})
            test_data.append({'img_path': each_img_path, 'text': img2text[each_img_fname], 'label': label, 'article': img_outer})
            
        if count > num_limit:
            break
    
    return test_data

def get_img_only_w_caption(num_limit, constrained=False):
    core_imgs = get_core_imgs()
    img_url = './dataset/mee/raw_data/image/image_url_caption.json'
    img_dir = './dataset/mee/raw_data/image/image/'
    img_caption = read_json(img_url)
    img_anno = read_img_annotations()
    test_data = []
    count = 0
    avail_articles = get_avail_articles()
    for img_outer, img_inners in img_caption.items():
        if img_outer+'.rsd.txt' not in avail_articles:
            print('invalid article img')
            continue
        for img_inner, caption in img_inners.items():
            count += 1
            if count > num_limit:
                break
            each_img_path = img_dir + img_outer + '_' + img_inner + '.jpg'
            each_img_name = img_outer + '_' + img_inner
            each_img_fname = each_img_name + '.jpg'
            if each_img_fname not in core_imgs or constrained:
                continue
            label = img_anno.get(each_img_name, {'event_type':'None'})
            test_data.append({'img_path': each_img_path, 'label': label, 'caption': caption['caption'],'article': img_outer})
        if count > num_limit:
            break
    return test_data

#看样子图片名是对应article的，可以统计一下：

def get_img_only(num_limit, constrained=False):
    core_imgs = get_core_imgs()
    img_url = './dataset/mee/raw_data/image/image_url_caption.json'
    img_dir = './dataset/mee/raw_data/image/image/'
    img_caption = read_json(img_url)
    img_anno = read_img_annotations()
    test_data = []
    count = 0
    avail_articles = get_avail_articles()
    #img_outer就是article的名字
    for img_outer, img_inners in img_caption.items():
        if img_outer+'.rsd.txt' not in avail_articles:
            print('invalid article img')
            continue
        
        for img_inner, caption in img_inners.items():
            count += 1
            if count > num_limit:
                break
            each_img_path = img_dir + img_outer + '_' + img_inner + '.jpg'
            each_img_name = img_outer + '_' + img_inner
            each_img_fname = each_img_name + '.jpg'
            if each_img_fname not in core_imgs or constrained:
                continue
            label = img_anno.get(each_img_name, {'event_type':'None'})
            test_data.append({'img_path': each_img_path, 'label': label, 'caption': caption, 'article': img_outer})
        if count > num_limit:
            break
    return test_data

def get_avail_articles():
    avail_articles = ['VOA_EN_NW_2017.02.23.3736556.rsd.txt', 'VOA_EN_NW_2015.03.27.2696792.rsd.txt', 'VOA_EN_NW_2017.05.31.3878829.rsd.txt', 'VOA_EN_NW_2013.09.09.1746134.rsd.txt', 'VOA_EN_NW_2014.01.22.1835368.rsd.txt', 'VOA_EN_NW_2015.12.08.3093406.rsd.txt', 'VOA_EN_NW_2014.03.02.1862314.rsd.txt', 'VOA_EN_NW_2015.09.25.2978237.rsd.txt', 'VOA_EN_NW_2016.06.02.3358563.rsd.txt', 'VOA_EN_NW_2016.10.01.3532811.rsd.txt', 'VOA_EN_NW_2013.11.10.1787498.rsd.txt', 'VOA_EN_NW_2016.05.16.3331907.rsd.txt', 'VOA_EN_NW_2015.03.27.2697067.rsd.txt', 'VOA_EN_NW_2017.02.23.3737435.rsd.txt', 'VOA_EN_NW_2016.01.05.3131977.rsd.txt', 'VOA_EN_NW_2013.01.10.1581113.rsd.txt', 'VOA_EN_NW_2016.08.19.3472017.rsd.txt', 'VOA_EN_NW_2016.07.27.3436565.rsd.txt', 'VOA_EN_NW_2017.01.20.3685521.rsd.txt', 'VOA_EN_NW_2013.12.31.1820694.rsd.txt', 'VOA_EN_NW_2013.11.21.1794874.rsd.txt', 'VOA_EN_NW_2016.02.08.3181623.rsd.txt', 'VOA_EN_NW_2016.12.14.3635600.rsd.txt', 'VOA_EN_NW_2014.01.31.1841956.rsd.txt', 'VOA_EN_NW_2013.01.21.1587732.rsd.txt', 'VOA_EN_NW_2014.12.13.2556485.rsd.txt', 'VOA_EN_NW_2017.07.10.3936006.rsd.txt', 'VOA_EN_NW_2017.06.26.3916532.rsd.txt', 'VOA_EN_NW_2016.06.15.3376531.rsd.txt', 'VOA_EN_NW_2017.01.16.3678417.rsd.txt', 'VOA_EN_NW_2009.12.09.416313.rsd.txt', 'VOA_EN_NW_2017.02.02.3702962.rsd.txt', 'VOA_EN_NW_2012.05.26.1105689.rsd.txt', 'VOA_EN_NW_2016.07.12.3415627.rsd.txt', 'VOA_EN_NW_2016.11.09.3589149.rsd.txt', 'VOA_EN_NW_2017.04.28.3829747.rsd.txt', 'VOA_EN_NW_2015.01.16.2600974.rsd.txt', 'VOA_EN_NW_2016.02.17.3194684.rsd.txt', 'VOA_EN_NW_2014.12.17.2562583.rsd.txt', 'VOA_EN_NW_2014.05.12.1912526.rsd.txt', 'VOA_EN_NW_2017.02.24.3739026.rsd.txt', 'VOA_EN_NW_2016.09.04.3493324.rsd.txt', 'VOA_EN_NW_2014.07.14.1957332.rsd.txt', 'VOA_EN_NW_2017.01.30.3699085.rsd.txt', 'VOA_EN_NW_2014.03.24.1877778.rsd.txt', 'VOA_EN_NW_2015.08.11.2913549.rsd.txt', 'VOA_EN_NW_2016.10.09.3542777.rsd.txt', 'VOA_EN_NW_2016.10.13.3549797.rsd.txt', 'VOA_EN_NW_2017.04.20.3818034.rsd.txt', 'VOA_EN_NW_2016.10.19.3557471.rsd.txt', 'VOA_EN_NW_2016.07.19.3424061.rsd.txt', 'VOA_EN_NW_2013.09.27.1758726.rsd.txt', 'VOA_EN_NW_2016.09.27.3527400.rsd.txt', 'VOA_EN_NW_2016.05.11.3325807.rsd.txt', 'VOA_EN_NW_2015.11.26.3074696.rsd.txt', 'VOA_EN_NW_2015.08.04.2902011.rsd.txt', 'VOA_EN_NW_2014.01.06.1823838.rsd.txt', 'VOA_EN_NW_2012.09.12.1506267.rsd.txt', 'VOA_EN_NW_2015.05.28.2793713.rsd.txt', 'VOA_EN_NW_2017.02.01.3702375.rsd.txt', 'VOA_EN_NW_2016.02.10.3184933.rsd.txt', 'VOA_EN_NW_2015.09.11.2959960.rsd.txt', 'VOA_EN_NW_2013.10.30.1779904.rsd.txt', 'VOA_EN_NW_2016.01.13.3142975.rsd.txt', 'VOA_EN_NW_2009.12.13.416444.rsd.txt', 'VOA_EN_NW_2015.09.05.2949130.rsd.txt', 'VOA_EN_NW_2015.07.31.2888384.rsd.txt', 'VOA_EN_NW_2016.12.17.3640413.rsd.txt', 'VOA_EN_NW_2016.11.29.3615580.rsd.txt', 'VOA_EN_NW_2016.11.22.3607054.rsd.txt', 'VOA_EN_NW_2015.07.31.2889903.rsd.txt', 'VOA_EN_NW_2015.01.20.2605914.rsd.txt', 'VOA_EN_NW_2014.08.20.2422182.rsd.txt', 'VOA_EN_NW_2017.01.17.3679550.rsd.txt', 'VOA_EN_NW_2016.06.27.3393616.rsd.txt', 'VOA_EN_NW_2015.10.21.3017239.rsd.txt', 'VOA_EN_NW_2016.09.27.3527025.rsd.txt', 'VOA_EN_NW_2016.10.25.3565688.rsd.txt', 'VOA_EN_NW_2016.08.12.3461808.rsd.txt', 'VOA_EN_NW_2017.05.14.3850909.rsd.txt', 'VOA_EN_NW_2014.11.09.2508360.rsd.txt', 'VOA_EN_NW_2016.03.21.3247676.rsd.txt', 'VOA_EN_NW_2016.05.17.3333843.rsd.txt', 'VOA_EN_NW_2014.12.15.2551521.rsd.txt', 'VOA_EN_NW_2014.08.07.2405842.rsd.txt', 'VOA_EN_NW_2016.02.14.3190340.rsd.txt', 'VOA_EN_NW_2016.04.14.3285847.rsd.txt', 'VOA_EN_NW_2017.02.15.3726266.rsd.txt', 'VOA_EN_NW_2016.09.21.3519438.rsd.txt', 'VOA_EN_NW_2017.03.15.3767889.rsd.txt', 'VOA_EN_NW_2016.06.27.3393580.rsd.txt', 'VOA_EN_NW_2017.05.08.3842885.rsd.txt', 'VOA_EN_NW_2015.07.16.2865144.rsd.txt', 'VOA_EN_NW_2013.01.28.1592104.rsd.txt', 'VOA_EN_NW_2016.11.10.3590360.rsd.txt', 'VOA_EN_NW_2016.03.18.3243744.rsd.txt', 'VOA_EN_NW_2016.01.07.3135549.rsd.txt', 'VOA_EN_NW_2016.10.18.3556203.rsd.txt', 'VOA_EN_NW_2015.12.05.3090416.rsd.txt', 'VOA_EN_NW_2016.07.28.3437456.rsd.txt', 'VOA_EN_NW_2016.07.28.3437601.rsd.txt', 'VOA_EN_NW_2017.02.03.3705362.rsd.txt', 'VOA_EN_NW_2013.07.08.1696918.rsd.txt', 'VOA_EN_NW_2013.04.03.1634074.rsd.txt', 'VOA_EN_NW_2016.04.13.3283669.rsd.txt', 'VOA_EN_NW_2017.04.13.3808915.rsd.txt', 'VOA_EN_NW_2016.01.05.3132000.rsd.txt', 'VOA_EN_NW_2015.08.18.2922213.rsd.txt', 'VOA_EN_NW_2015.05.05.2749920.rsd.txt', 'VOA_EN_NW_2016.09.27.3527595.rsd.txt', 'VOA_EN_NW_2016.09.27.3526898.rsd.txt', 'VOA_EN_NW_2016.07.04.3403532.rsd.txt', 'VOA_EN_NW_2013.09.13.1749095.rsd.txt', 'VOA_EN_NW_2015.06.26.2838241.rsd.txt', 'VOA_EN_NW_2014.12.02.2542349.rsd.txt', 'VOA_EN_NW_2013.11.20.1794343.rsd.txt', 'VOA_EN_NW_2016.03.01.3214677.rsd.txt', 'VOA_EN_NW_2012.12.10.1562068.rsd.txt', 'VOA_EN_NW_2016.05.30.3351608.rsd.txt', 'VOA_EN_NW_2016.06.21.3385836.rsd.txt', 'VOA_EN_NW_2015.10.07.2995399.rsd.txt', 'VOA_EN_NW_2013.08.14.1729937.rsd.txt', 'VOA_EN_NW_2016.10.30.3572132.rsd.txt', 'VOA_EN_NW_2015.08.08.2908562.rsd.txt', 'VOA_EN_NW_2015.08.03.2896400.rsd.txt', 'VOA_EN_NW_2016.04.18.3290978.rsd.txt', 'VOA_EN_NW_2016.11.29.3616534.rsd.txt', 'VOA_EN_NW_2017.06.15.3902402.rsd.txt', 'VOA_EN_NW_2017.04.25.3825261.rsd.txt', 'VOA_EN_NW_2013.11.05.1784209.rsd.txt', 'VOA_EN_NW_2015.12.17.3106664.rsd.txt', 'VOA_EN_NW_2016.02.10.3184811.rsd.txt', 'VOA_EN_NW_2013.05.23.1666803.rsd.txt', 'VOA_EN_NW_2015.05.14.2767071.rsd.txt', 'VOA_EN_NW_2016.12.20.3643261.rsd.txt', 'VOA_EN_NW_2017.04.07.3800274.rsd.txt', 'VOA_EN_NW_2017.02.14.3724228.rsd.txt', 'VOA_EN_NW_2015.11.26.3074855.rsd.txt', 'VOA_EN_NW_2017.06.07.3891137.rsd.txt', 'VOA_EN_NW_2015.10.26.3022860.rsd.txt', 'VOA_EN_NW_2014.12.29.2577251.rsd.txt', 'VOA_EN_NW_2016.05.12.3327048.rsd.txt', 'VOA_EN_NW_2015.04.30.2743015.rsd.txt', 'VOA_EN_NW_2017.04.05.3798460.rsd.txt', 'VOA_EN_NW_2014.12.25.2573211.rsd.txt', 'VOA_EN_NW_2016.12.28.3654540.rsd.txt', 'VOA_EN_NW_2015.11.05.3038535.rsd.txt', 'VOA_EN_NW_2013.08.09.1726740.rsd.txt', 'VOA_EN_NW_2016.04.22.3297950.rsd.txt', 'VOA_EN_NW_2016.10.31.3572766.rsd.txt', 'VOA_EN_NW_2017.01.16.3678663.rsd.txt', 'VOA_EN_NW_2017.05.17.3854850.rsd.txt', 'VOA_EN_NW_2016.03.25.3254676.rsd.txt', 'VOA_EN_NW_2016.03.04.3219688.rsd.txt', 'VOA_EN_NW_2016.01.06.3133182.rsd.txt', 'VOA_EN_NW_2016.07.14.3417836.rsd.txt', 'VOA_EN_NW_2013.10.30.1779954.rsd.txt', 'VOA_EN_NW_2017.01.30.3699018.rsd.txt', 'VOA_EN_NW_2016.12.14.3635692.rsd.txt', 'VOA_EN_NW_2017.06.21.3910361.rsd.txt', 'VOA_EN_NW_2016.04.19.3291980.rsd.txt', 'VOA_EN_NW_2016.09.02.3491404.rsd.txt', 'VOA_EN_NW_2016.01.28.3166988.rsd.txt', 'VOA_EN_NW_2013.07.27.1711271.rsd.txt', 'VOA_EN_NW_2016.12.28.3654650.rsd.txt', 'VOA_EN_NW_2017.03.27.3783190.rsd.txt', 'VOA_EN_NW_2017.05.01.3832748.rsd.txt', 'VOA_EN_NW_2015.12.23.3115897.rsd.txt', 'VOA_EN_NW_2016.08.11.3460137.rsd.txt', 'VOA_EN_NW_2015.05.08.2747846.rsd.txt', 'VOA_EN_NW_2017.03.19.3772413.rsd.txt', 'VOA_EN_NW_2017.01.22.3686805.rsd.txt', 'VOA_EN_NW_2013.03.18.1623546.rsd.txt', 'VOA_EN_NW_2015.07.02.2845998.rsd.txt', 'VOA_EN_NW_2016.06.15.3376757.rsd.txt', 'VOA_EN_NW_2014.05.22.1920650.rsd.txt', 'VOA_EN_NW_2017.05.15.3851683.rsd.txt', 'VOA_EN_NW_2016.07.10.3411182.rsd.txt', 'VOA_EN_NW_2013.11.15.1791049.rsd.txt', 'VOA_EN_NW_2017.06.15.3901731.rsd.txt', 'VOA_EN_NW_2015.03.21.2689721.rsd.txt', 'VOA_EN_NW_2014.12.16.2560677.rsd.txt', 'VOA_EN_NW_2016.05.13.3328601.rsd.txt', 'VOA_EN_NW_2017.05.04.3838167.rsd.txt', 'VOA_EN_NW_2015.03.04.2667016.rsd.txt', 'VOA_EN_NW_2016.06.09.3369222.rsd.txt', 'VOA_EN_NW_2015.04.10.2713895.rsd.txt', 'VOA_EN_NW_2015.03.05.2668808.rsd.txt', 'VOA_EN_NW_2016.07.09.3410778.rsd.txt', 'VOA_EN_NW_2016.11.09.3589252.rsd.txt', 'VOA_EN_NW_2017.01.28.3696450.rsd.txt', 'VOA_EN_NW_2016.12.27.3653223.rsd.txt', 'VOA_EN_NW_2016.02.15.3191284.rsd.txt', 'VOA_EN_NW_2016.08.10.3458597.rsd.txt', 'VOA_EN_NW_2017.03.16.3769606.rsd.txt', 'VOA_EN_NW_2014.05.25.1921616.rsd.txt', 'VOA_EN_NW_2016.02.19.3197746.rsd.txt', 'VOA_EN_NW_2015.03.19.2685042.rsd.txt', 'VOA_EN_NW_2017.04.05.3797975.rsd.txt', 'VOA_EN_NW_2016.03.23.3251117.rsd.txt', 'VOA_EN_NW_2016.11.28.3614228.rsd.txt', 'VOA_EN_NW_2016.04.30.3309655.rsd.txt', 'VOA_EN_NW_2016.04.01.3264272.rsd.txt', 'VOA_EN_NW_2015.11.05.3037829.rsd.txt', 'VOA_EN_NW_2017.04.24.3823632.rsd.txt', 'VOA_EN_NW_2016.09.08.3498453.rsd.txt', 'VOA_EN_NW_2016.06.10.3370380.rsd.txt', 'VOA_EN_NW_2017.03.21.3775715.rsd.txt', 'VOA_EN_NW_2016.04.27.3305148.rsd.txt', 'VOA_EN_NW_2016.03.20.3242323.rsd.txt', 'VOA_EN_NW_2016.12.09.3630378.rsd.txt', 'VOA_EN_NW_2016.05.08.3320391.rsd.txt', 'VOA_EN_NW_2015.05.08.2760181.rsd.txt', 'VOA_EN_NW_2016.11.06.3583381.rsd.txt', 'VOA_EN_NW_2016.11.18.3602889.rsd.txt', 'VOA_EN_NW_2013.05.02.1653029.rsd.txt', 'VOA_EN_NW_2016.07.11.3412014.rsd.txt', 'VOA_EN_NW_2016.04.11.3279945.rsd.txt', 'VOA_EN_NW_2017.02.22.3734870.rsd.txt', 'VOA_EN_NW_2015.09.02.2942501.rsd.txt', 'VOA_EN_NW_2015.10.01.2987042.rsd.txt', 'VOA_EN_NW_2014.10.09.2478154.rsd.txt', 'VOA_EN_NW_2016.08.27.3483319.rsd.txt', 'VOA_EN_NW_2014.09.10.2444962.rsd.txt', 'VOA_EN_NW_2016.08.17.3469525.rsd.txt', 'VOA_EN_NW_2017.01.19.3676842.rsd.txt', 'VOA_EN_NW_2015.10.18.3012400.rsd.txt', 'VOA_EN_NW_2017.02.17.3729391.rsd.txt', 'VOA_EN_NW_2014.06.03.1928789.rsd.txt', 'VOA_EN_NW_2016.04.24.3300395.rsd.txt', 'VOA_EN_NW_2016.05.20.3338580.rsd.txt', 'VOA_EN_NW_2016.07.23.3431608.rsd.txt', 'VOA_EN_NW_2015.08.11.2913225.rsd.txt', 'VOA_EN_NW_2014.04.16.1894548.rsd.txt', 'VOA_EN_NW_2016.06.26.3392569.rsd.txt']
    return set(avail_articles)

def get_text_dataset(num_limit):
    test_data = []
    for url in ['./dataset/mee/annotations/text_multimedia_event.json', './dataset/mee/annotations/text_only_event.json']:
        for sample in read_json(url):
            ret_sample = {}
            ret_sample["golden-event-mentions"] = sample["golden-event-mentions"]
            ret_sample['sentence'] = sample["sentence"]
            ret_sample['image'] = sample['image']
            test_data.append(ret_sample)
            if len(test_data) >= num_limit:
                return test_data
    return test_data

def get_core_imgs():
    # read from text_only_event.json and text_multimedia_event.json
    whole_images = []
    for url in ['./dataset/mee/annotations/text_multimedia_event.json', './dataset/mee/annotations/text_only_event.json']:
        for sample in read_json(url):
            whole_images += sample["image"]
    return set(whole_images)

def get_img2text():
    img2text = {}
    for url in ['./dataset/mee/annotations/text_multimedia_event.json', './dataset/mee/annotations/text_only_event.json']:
        for sample in read_json(url):
            for img in sample['image']:
                img2text.setdefault(img, []).append(sample['sentence'])
    return img2text
# read all the img annotations
def read_img_annotations():
    img_anno = {}
    for url in ['./dataset/mee/annotations/image_only_event.json', './dataset/mee/annotations/image_multimedia_event.json']:
        img_anno.update(read_json(url))
    return img_anno

if __name__ == '__main__':
    import os
    img_url = './dataset/mee/raw_data/image/image_url_caption.json'
    img_dir = './dataset/mee/raw_data/image/image/'
    img_caption = read_json(img_url)
    img_anno = read_img_annotations()
    test_data = []
    count = 0
    img_lists = []
    for img_outer, img_inners in img_caption.items():
        count += 1
        img_lists.append(img_outer+ '.rsd.txt')
    # read the file names in dataset/mee/raw_data/article and save them in a list
    # write the code...
    #print(img_lists)
    dir_path = './dataset/mee/raw_data/article'

    # 使用列表推导式读取目录下的所有文件名
    file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


    print(file_names)