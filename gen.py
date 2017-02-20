from gensim.models import Word2Vec

gmodel=Word2Vec.load_word2vec_format('final_embedding_dic.txt', binary=True)
"""
len(w2v_model.vocab)
"""

ms=gmodel.most_similar(['Breast'])
for x in ms:
    print x[0],x[1]

    """

print(gmodel.similarity('data', 'system'))

"""