"""
日志配置
"""
import logging

logging.basicConfig(filename='log-step-1.log', level=logging.INFO,
                    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s")


"""
 默认数据存储
"""
all_conv_list = []


"""
读取数据
"""
import os
path = 'in_use/'
files = os.listdir(path)

tmp_conv_list = []

for file in files:
    if not os.path.isdir(file):
        f = open(os.path.join(path, file), mode='r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) < 2:
                if len(tmp_conv_list) > 0:
                    all_conv_list.append(tmp_conv_list.copy())
                    tmp_conv_list = []

            else:
                tmp_conv_list.append(line.strip())

logging.info("原始数据读取如下" + ' 文本数' + '对话数量' + str(len(all_conv_list)))


for item in all_conv_list[:5]:
    logging.info(item)

"""
排序去重
"""

# 变为str 方便进行去重, 使用$$$ 进行分割 
all_conv_str_list = list(set(["$$$".join(item) for item in all_conv_list])) 
all_conv_str_list = sorted(all_conv_str_list)

logging.info('原始对话个数')
logging.info(str(len(all_conv_str_list)))
logging.info('不重复对话所占比例')
logging.info(str(len(all_conv_str_list)/len(all_conv_list)))

"""
繁简转化
"""

#from snownlp import SnowNLP
#tmp_list = []
#for item  in all_conv_str_list:
#    tmp_list.append(SnowNLP(item).han)
#all_conv_str_list = tmp_list


#logging.info("完成繁简转化")

"""
纠错
"""
# import pycorrector
# tmp_list_cor = []
# for line in all_conv_str_list:
#    corrected_sent, detail = pycorrector.correct(line)
#    tmp_list_cor.append(corrected_sent)

# all_conv_str_list = tmp_list_cor

"""
反问/疑问 : 当前已经在 在噪声过滤中去掉，后续考虑使用句法分析等方法加强
"""
"""
话术是否连贯 : pass
"""
"""
无有效中文 : pass
全是标点符号
"""


"""
噪声过滤-包含即删除
"""

# 读取系统敏感词
system_sensitive_list = []
with open('sensitive_results.txt', mode='r', encoding='utf-8') as f:
    for item in f:
        system_sensitive_list.append(item.strip())



# 表情符号
emoj_list = [r'\x', r'\t', r'\u', r'\X', r'\U']
# 标点符号
punc_list = ['#','#', '-','<','>','《','》', "@", '【', '】', '？', ' ——',
             '_','_', '"', "'",':','：', '‘','’',  '°C', 'C°', '】', '【']
# 噪声过滤
ad_list = ['直播','庆祝','销量', '销量','到货']
news_list = ['事件']
dirty_list = ['撸', '鸡巴', '屁']
web_list = ['说出三个', '写出三个', '写出你', '转一个','转运', '达人秀','演唱会','明星', '图片', '文章', 'mv', '视频', '传说中','吃吃吃','礼物','朱艳艳','做客','记录','Nick','中国梦','博鳌','求救']

reference_list = ['这个', '那个', '哪个']
politics_list = ['下台','政绩工程', '毛泽东']  
special_list=[ 'si','鸡鸡','小通','旺财']
security_list = ['查询', '搞蒙', '你这句话']

locations_list = ['成都', '北京', '上海']
greetings_list = ['恭喜','恭喜', '晚安', '早安', '上午好','下午好','晚上好']
time_list = ['早上','清早','凌晨','上午','中午','晚上','今晚','昨天','明天','后天']
weather_lsit = ['PM', '北京空气']

check_list = system_sensitive_list + emoj_list + punc_list + ad_list + news_list + dirty_list + web_list + reference_list + politics_list + special_list + security_list + locations_list + greetings_list + time_list + weather_lsit

def check_error(input_text):
    for item in check_list:
        if item in input_text:
            return True
    return  False

tmp_list = []

for line in all_conv_str_list:
    if check_error(line) is False:
        tmp_list.append(line)

all_conv_str_list = tmp_list 
logging.info('完成噪声过滤-包含即删除-现对话个数')
logging.info(len(all_conv_str_list))

"""
噪声过滤-部分剔除
"""
import re
match_list = ['转起来', '分享自', '链接','转']
#那一年，你还记得你床头的第一张明星的海报，是为谁而贴？转

def replace_list(input_list:list):
    tmp_list = []

    for item in input_list:
        text_re = re.compile('【.*】')
        item = text_re.sub('', item)
        text_re = re.compile('【.*】')
        item = text_re.sub('', item)
        text_re = re.compile('#.*#')
        item = text_re.sub('', item)

        for tmp in match_list:
            item = item.replace(tmp, '')

        tmp_list.append(item)
        
    return tmp_list

all_conv_str_list = replace_list(all_conv_str_list)

logging.info('完成噪声过滤-匹配删除-现对话个数')
logging.info(len(all_conv_str_list))

# 将字符串转化为list
all_conv_list = [ item.split("$$$") for item in all_conv_str_list]

all_conv_list = [ tmp for tmp in all_conv_list if len(tmp) > 1 ] 
# 保证一个对话中最少两句话 

print(all_conv_list[:10])
print(all_conv_str_list[:10])

import pickle

with open("corpus-step-1.pkl", mode='wb') as f:
    pickle.dump(all_conv_list, f)

