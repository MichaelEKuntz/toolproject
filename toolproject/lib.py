import numpy as np
from gensim.models import Word2Vec


def string_converter(input_list):
    output_list = []
    for elem in input_list:
        output_list.append(elem.split())
    return output_list


def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        try:
            embedded_sentence.append(word2vec.wv[word])
        except KeyError:
            continue
    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    embedded_sentences = []
    for sentence in sentences:
        embedded_sentences.append(embed_sentence(word2vec, sentence))
    return embedded_sentences


def word2vec(sentences,size=20,min_count=10,window=20):
    return Word2Vec(sentences=sentences,size=20,min_count=10,window=20)


if __name__ == '__main__':
    input_list = [['Du bist ein richtiger Vollidiot.'], ['Ich freue mich auf das Ende dieser Odyssee.']]
    print(string_converter(input_list))
