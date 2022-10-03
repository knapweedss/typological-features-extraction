import PyPDF2
import os
from src.preprocess_data import preprocess_text
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
model = SentenceTransformer('all-MiniLM-L6-v2')
curr_dir = os.getcwd()
data_dir = curr_dir + '/grammars'


def read_features(data_directory):
    with open(f"{data_directory}/wals_features_list.txt", encoding='UTF-8') as f:
        file = f.readlines()
        keys, values = [], []
        for f in file[::2]:
            if f[8] != 1:
                keys.append(f[8:-3])
            else:
                keys.append(f[8:-2])
        for f in file[1::2]:
            values.append(f)
    return dict(zip(keys, values))


def get_grammar_text(data_directory):
    corpora = []
    page_texts = []
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            fileReader = PyPDF2.PdfFileReader(data_directory + '/' + file)
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


def embeddings_and_cos(bm25_res):
    cos_scores = []
    pdf_indexes = []
    for ind in bm25_res:
        sentences1 = file_page[ind]
        pdf_indexes.append(ind)
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(query, convert_to_tensor=True)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        cos_scores.append(cosine_scores.numpy()[0][0])
        keys, values = pdf_indexes, cos_scores
    x = dict(zip(keys, values))
    newdict = (dict(sorted(x.items(), key=lambda item: item[1])))
    return list(newdict.keys())[::-1]


def penn_to_wn_tags(pos_tag):
    if pos_tag.startswith('J'):
        return wn.ADJ
    elif pos_tag.startswith('V'):
        return wn.VERB
    elif pos_tag.startswith('N'):
        return wn.NOUN
    elif pos_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn_tags(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sent1, sent2):
    sent1 = pos_tag(word_tokenize(sent1))
    sent2 = pos_tag(word_tokenize(sent2))

    syns1 = [tagged_to_synset(*tagged_word) for tagged_word in sent1]
    syns2 = [tagged_to_synset(*tagged_word) for tagged_word in sent2]

    syns1 = [ss for ss in syns1 if ss]
    syns2 = [ss for ss in syns2 if ss]

    score, count = 0.0, 0

    for synset in syns1:
        best_score = max([synset.path_similarity(ss) for ss in syns2])

        if best_score is not None:
            score += best_score
            count += 1

    score /= count
    return score


d = read_features(curr_dir + '/wals_data')
query = 'Order of Adjective and Noun'
#print(d)
data = get_grammar_text(data_dir)
corpus = data[0]
file_page = data[1]
bm25_results = (BM25(corpus, query))
cos_emb_ind = (embeddings_and_cos(bm25_results))

final_pages, final_scores = [], []
for s in cos_emb_ind:
    try:
        # +1 для соответствия страницы в пдф
        final_pages.append(s + 1)
        final_scores.append(sentence_similarity(file_page[s], query))
    except:
        Exception
print(final_pages, final_scores)