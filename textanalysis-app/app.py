import streamlit as st
import numpy as np
import pandas as pd
from gensim import utils
import pprint
import gensim
import warnings
import tempfile
import analysis_funcs as nlp
from PIL import  Image
import urllib

st.title('Text Analyzer')
rad = st.sidebar.radio('Navigation', 
['Home ', 'Word Cloud', 'N-Gram Analysis', 'Part of Speech Analysis', 'Similarity Analysis'])


display = Image.open('display.png')
display = np.array(display)
st.image(display, use_column_width = True)    


st.set_option('deprecation.showfileUploaderEncoding', False)
st.header('Enter text or upload file')
text = st.text_area('Type Something', height = 400)
    
file_text = st.file_uploader('Text File', encoding = 'ISO-8859-1')
    
if file_text!=None:
    text = file_text.read()
    

# word cloud
if rad == 'Word Cloud':
    st.header('Generate Word Cloud')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    mask = st.file_uploader('Use Image Mask', type = ['jpg'])
    nlp.create_wordcloud(text, mask)
    st.pyplot()

# ngram
if rad == 'N-Gram Analysis':
    st.header('N Gram Analysis')    
    n = st.slider('N gram', min_value = 1, max_value = 10, step = 1, value = 2)
    topk = st.slider('Top k most common', 
    min_value = 10, max_value = 100, step = 10, value = 10)
    nlp.plot_ngrams(text, n = n, topk = topk)
    st.pyplot()

# POS
if rad == 'Part of Speech Analysis':        
    st.header('POS Tagging')
    #text = st.text_area('Enter Text..Max Length:1000000 ', height = 400)
    display = st.checkbox('Display')
    nlp.structure_anslysis(text, display=display)
    st.pyplot()
    
# word similarity
if rad == 'Similarity Analysis':
    st.header('Similarity Analysis')
    rads = st.radio('Select Model Type',['Custom', 'Pre-Trained'])
    
    if rads == 'Pre-Trained':
        st.write('Loading google-news-300....')        
        model = gensim.models.KeyedVectors.load_word2vec_format('word2vec-google-news-300.gz', binary=True, limit = 500000)        
        st.write('model loaded')
    else:
        corpus = text
        # training word2vec model on our corpus
        st.write('Training Model...')
        sentences = nlp.MyCorpus(corpus)            
        model = gensim.models.Word2Vec(sentences = sentences, min_count = 5, size = 300)

    obj = st.text_input('Input text to calculate similarity')
    n = st.slider('Top N most similar', 
    min_value = 10, max_value = 50, step = 5, value = 10)
    plot = st.checkbox('Plot')
    similarity = nlp.get_top_similarity(obj, model, n, plot)
         
    if similarity is not None:     
        similarity = {word:sim for word,sim in similarity}      
        data = pd.DataFrame.from_dict(similarity, orient = 'index', columns = ['Cosine Similarity'])

        st.dataframe(data)        
        

        
    







    









    





