import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx
import rouge
# nltk.download('punkt')

def textrank_summarization(text):   
    text1 = str(text)
    # Preprocessing 
    sentences = sent_tokenize(text1)
    sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]

    # Tokenization 
    sentence_tokens = []
    for sentence in sentences_clean:
        tokens = word_tokenize(sentence)
        sentence_tokens.append(tokens)
    w2v=Word2Vec(sentence_tokens,vector_size=1,min_count=1,epochs=1000)

    # Generating word embeddings     
    sentence_embeddings=[[w2v.wv[word][0] for word in words] for words in sentence_tokens]

    max_len=max([len(tokens) for tokens in sentences_clean])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    
    # Building similarity matrix
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Ranking 
    top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences_clean)}
    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:3])
    top_outline=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:1])
    summary=""
    outline=""
 
    for sent in sentences_clean:
        if sent in top.keys():
            summary+=sent+"."
    for sent in sentences_clean:
        if sent in top_outline.keys():
            outline+=sent

    return summary

if __name__ == "__main__": 
    text = "anything else stop sum artist think translate online profile words Twitter allows entire page indulgence website would allow Bring salient features creativity experience passion reasons painting Make clear readers artist loves art produces high quality art true champion art great words find friend help really important aspect selling online establishment credibility reliability"
    textrank_summarization(text)