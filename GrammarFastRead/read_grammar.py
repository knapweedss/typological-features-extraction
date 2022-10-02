import PyPDF2
import os
from src.preprocess_data import preprocess_text
from rank_bm25 import BM25Okapi
curr_dir = os.getcwd()
data_dir = curr_dir + '/grammars'


def read_features(data_directory):
    with open(f"{data_directory}/wals_features_list.txt", encoding='UTF-8') as f:
        file = f.readlines()
        keys, values = [], []
        for f in file[::2]:
            keys.append(f[8:-3])
        for f in file[1::2]:
            values.append(f)
    return dict(zip(keys, values))


def get_grammar_text(data_directory):
    corpora = []
    page_texts = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            fileReader = PyPDF2.PdfFileReader(data_dir + '/' + file)
            page = fileReader.pages
            for p in page:
                page_text = p.extract_text()
                page_texts.append(page_text)
                corpora.append(preprocess_text(page_text))
    return corpora, page_texts


def BM25(corpora, query):
    bm25 = BM25Okapi(corpora)
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    return sorted(range(len(scores)), key=lambda i: scores[i])[-10:]


d = read_features(curr_dir + '/wals_data')
query = '81A'
data = get_grammar_text(data_dir)
corpus = data[0]
file_page = data[1]
print(BM25(corpus, d[query]))
# страница пдф - индекс + 1
