import os
import pickle
from stanfordcorenlp import StanfordCoreNLP
import string

string.punctuation
def Syntactic(yic_content):
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')

    #路径设置
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    #dataset: 20ng  mr  ohsumed R8 R52
    dataset ='mr'
    input = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
    output = os.sep.join(['..', 'data_tgcn', dataset, 'stanford'])
    file = open(input + '.chinesefen.txt', "w",encoding="utf-8")
    # file = open(input + '.candidates_fen.txt', "w", encoding="utf-8")
    #读取病例
    # yic_content_list = []
    # f = open(input + '.chinese.txt', 'r', encoding="utf-8")
    # lines = f.readlines()
    # for line in lines:
    #     yic_content_list.append(line.strip())
    # f.close()
    yic_content_list = yic_content
    # stop_words = set(stopwords.words(""))

    stop_words = set()
    with open(input + '.stop.txt', 'r', encoding="utf-8") as f:
        for line in f:
            stop_words.add(line.strip())

    #获取句法依存关系对
    rela_pair_count_str = {}
    for doc_id in range(len(yic_content_list)):
        # print(doc_id)
        words = yic_content_list[doc_id]
        words = words.split("\n")
        rela = []
        for window in words:
            if window==' ':
                continue
            punctuation_string = string.punctuation
            for i in punctuation_string:
                window = window.replace(i, '')
            # print(window)
            res = nlp.dependency_parse(window)
            # print('Tokenize:', nlp.word_tokenize(window))
            file.write(' '.join(nlp.word_tokenize(window)))
            file.write("\n")
            fen = nlp.word_tokenize(window)
            # print(type(nlp.word_tokenize(window)))
            # print('Part of Speech:', nlp.pos_tag(window))
            # print('Named Entities:', nlp.ner(window))
            # print('Constituency Parsing:', nlp.parse(window))
            # print('Dependency Parsing:', nlp.dependency_parse(window))

            for tuple in res:
                rela.append(tuple[0] + ', ' + str(tuple[1])+ ', ' +str(tuple[2]))
            for pair in rela:
                pair=pair.split(", ")
                if pair[0]=='ROOT' or pair[1]=='ROOT':
                    continue
                if fen[int(pair[1])-1] == fen[int(pair[2])-1]:
                    continue
                if fen[int(pair[1])-1] in string.punctuation or fen[int(pair[2])-1] in string.punctuation:
                    continue
                if fen[int(pair[1])-1] in stop_words or fen[int(pair[2])-1] in stop_words:
                    continue
                word_pair_str = fen[int(pair[1])-1] + ',' + fen[int(pair[2])-1]
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # two orders
                word_pair_str = fen[int(pair[2])-1] + ',' + fen[int(pair[1])-1]
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
    file.close()
    nlp.close()
    # print(rela_pair_count_str)
    # 将rela_pair_count_str存成pkl格式
    output1=open(output + '/{}_chsy.pkl'.format(dataset),'wb')
    # output1 = open(output + '/{}_candidates.pkl'.format(dataset), 'wb')
    pickle.dump(rela_pair_count_str, output1)


# candidates = []
# candidates_path = '../data_tgcn/mr/build_train/candidates1.txt'
# f = open(candidates_path, 'r', encoding="utf-8")
# lines = f.readlines()
# for line in lines:
#     candidates.append(line.strip())
# f.close()
# Syntactic(candidates)
