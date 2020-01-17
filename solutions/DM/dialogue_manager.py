# pipeline
## 提取query中信息
## 记录历史信息
## 根据query中信息和历史信息来 为 nlg 提供支持

def error_correction(inputs):
    return inputs

def sequence_labeling(inputs):
    return '', '', ''

def syntax_analysis(inputs):
    return ''

def compress(inputs):
    if len(inputs) > 50:
        return inputs # 返回压缩后的结果
    else:
        return inputs

def anaphora_resolution(inputs):
    return inputs

def deal_query(inputs):
    inputs = error_correction(inputs)
    seg, pos, ner = sequence_labeling(inputs)
    syntax = syntax_analysis(inputs)
    compress = compress(inputs)

def clf_query(inputs, history):
    pass