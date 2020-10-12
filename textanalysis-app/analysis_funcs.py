# Standard Libraries
import os 
import re 
import string 
import numpy as np
from collections import Counter

# Text Processing Library 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud
from gensim import utils
import streamlit as st
import pprint
import gensim
import gensim.downloader as api
import warnings
import spacy
from spacy import displacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import tempfile
warnings.filterwarnings(action='ignore')


# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns
import spacy_streamlit
from PIL import Image


# Constants 
STOPWORDS = stopwords.words('english')
STOPWORDS + ['said']
# Cleaning Function 
def clean_text(text):
    '''
        Function which returns a clean text 
    '''    
    # Lower case 
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d', '', text)
    
    # Replace \n and \t functions 
    text = re.sub(r'\n', '', text)
    text = text.strip()
    
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove Stopwords and Lemmatise the data
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    
    return text

# Create a word cloud function 
def create_wordcloud(text, image_path = None):
    '''
    Pass a string to the function and output a word cloud
    
    :param text: The text for wordcloud
    :param image_path (optional): The image mask with a white background (default None)
    '''
    
    st.write('Creating Word Cloud..')

    text = clean_text(text)
    
    if image_path == None:
        # Generate the word cloud
        wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    stopwords = STOPWORDS, 
                    min_font_size = 10).generate(text) 
    
    else:
        mask = np.array(Image.open(image_path))
        wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    stopwords = STOPWORDS,
                    mask=mask,
                    min_font_size = 5).generate(text) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation = 'nearest') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()     


def plot_ngrams(text, n=2, topk=15):
    '''
    Function to plot the most commonly occuring n-grams in bar plots 
    
    '''
    text = clean_text(text)
    tokens = text.split()
    
    # get the ngrams 
    ngram_phrases = ngrams(tokens, n)
    
    # Get the most common ones 
    most_common = Counter(ngram_phrases).most_common(topk)
    
    # Make word and count lists 
    words, counts = [], []
    for phrase, count in most_common:
        word = ' '.join(phrase)
        words.append(word)
        counts.append(count)
    
    # Plot the barplot 
    plt.figure(figsize=(10, 6))
    title = "Most Common " + str(n) + "-grams in the text"
    plt.title(title)
    ax = plt.bar(words, counts)
    plt.xlabel("n-grams found in the text")
    plt.ylabel("Ngram frequencies")
    plt.xticks(rotation=90)
    plt.show()


nlp = spacy.load('en_core_web_md')
def pos_tag_counts(doc):
    """
    Calculates frequency distribution of univerasal POS Tags
    
    :param doc: spacy nlp object
    
    :returns frquency: tag counts (dictionaty)
    """
    tags = []
    for token in doc:
        tags.append(token.pos_)
    frequency = dict(Counter(tags).most_common())    
    return frequency
    
def entity_counts(doc):
    """
    Calculates frequency distribution of entities
    
    :param doc: spacy nlp object
    
    :returns frquency: entity counts (dictionaty)
    """
        
    tags = []
    for token in doc.ents:
        tags.append(token.label_)
    frequency = dict(Counter(tags).most_common())

    return frequency

def structure_anslysis(text, display = None):
    """
    visualizes POS tag counts, Entity counts & word-entity highlighted text
    
    :param text: The text to perform analysis on
    :param display: whether to diplay highlighted text (default None)
    :param entities: options for diplay (default None)
    
    :returns pos_freq (dictionary), ent_freq(dictionary)
    """
    st.write('Text Size Exceeded! Truncating...')
    doc = nlp(text[:100000])
    pos_freq = pos_tag_counts(doc)
    ent_freq = entity_counts(doc)
    
    fig, axs = plt.subplots(1, 2, figsize = (15, 6))
    
    sns.barplot(list(pos_freq.keys()), list(pos_freq.values()), color='#e84118', ax = axs[0])
    axs[0].set_title('POS COUNTS')
    axs[0].set_xticklabels(labels = list(pos_freq.keys()), rotation = 90)
    
    sns.barplot(list(ent_freq.keys()), list(ent_freq.values()), color='#273c75', ax = axs[1])
    axs[1].set_title('ENTITY COUNTS')
    axs[1].set_xticklabels(labels = list(ent_freq.keys()), rotation = 90)    
    
    plt.show()
    
    if display:
        spacy_streamlit.visualize_ner(doc, labels = nlp.get_pipe('ner').labels)
              
        
    return pos_freq, ent_freq


# similarity analysis
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):        
        for line in self.corpus.split('\n'):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def get_top_similarity(obj, model = None, n = 10, plot = False):
    """
    :param obj: The string for which word simmilarity to be calculated
    :param model: The model to use (if None google-news-300 is used)
    :param n: Top n similar words to return
    :param plot: Whethe to plot word similarity bar plot (default False)
    """
    
    obj = obj.lower().strip()    

    try:
        if len(obj.split()) >=1:
            vector = np.array([model[word] for word in obj.split(' ')]).mean(axis = 0)
        else:
            vector = model[obj]
        
        similarity = model.wv.similar_by_vector(vector, topn = n+1)[1:]        
        
        if plot:
            keys = [k for k,v in similarity if k!=obj]
            values = [v for k,v in similarity if k!= obj]
            sns.barplot(x = keys, y = values)
            plt.xticks(rotation = 90)
            plt.show()
            st.pyplot()
            
        return similarity
    except:
        st.write('Try another word or model.')