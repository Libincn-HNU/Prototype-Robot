from common_nlu import text_rewrite, discourse_analysis, lexical_analysis, syntax_analysis

if __name__ == "__main__":
    print(text_rewrite('今天上班给老人让坐，四十分鐘的車程,花了80分钟的时间...Oh My God！！！站的脚都麻了'))
    print(discourse_analysis('今天上班给老人让坐，四十分鐘的車程,花了80分钟的时间...Oh My God！！！站的脚都麻了'))
    print(lexical_analysis('今天上班给老人让坐，四十分鐘的車程'))
    print(syntax_analysis('今天上班给老人让坐，四十分鐘的車程'))