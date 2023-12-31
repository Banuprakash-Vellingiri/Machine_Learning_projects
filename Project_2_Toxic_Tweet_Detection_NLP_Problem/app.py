import streamlit as st
import spacy
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
#------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Toxic Tweet detection by Banuprakash Vellingiri",
    page_icon="	:feather:",
    layout="wide",
    initial_sidebar_state="expanded")
#------------------------------------------------------------------------------------------------
st.title('  🔍:orange[Toxic Tweet] Detector &nbsp;')
st.markdown("### :blue[Enter the Tweet Below ] :arrow_double_down: ")
text_input=st.text_input("")
submit_button=st.button(":orange[SUBMIT]")
if text_input and submit_button:
   nlp=spacy.load("en_core_web_sm") #Using Small English Model 
   #------------------------------------------------------------------------------------------------
   #Function for Text Preoprocessing
   def text_preprocess(text):
        nlp=spacy.load("en_core_web_sm") 
        processed_tweet=[]
        data=nlp(text)
        for token in data:
            if token.is_stop or token.is_punct:
                continue
            elif token.text.startswith('@'):
                continue
            else:
                processed_tweet.append(token.lemma_) 
        return " ".join(processed_tweet)  
   tweet_input=text_preprocess(text_input)
   #------------------------------------------------------------------------------------
   #Loading Random Forest Classifier Model
   import pickle
   with open ("rf.pkl","rb")as file2:
       model=pickle.load(file2)   
   with open ("tf-idf_vectorizer.pkl","rb")as file1:
      tfidf_vectorizer=pickle.load(file1)
   #------------------------------------------------------------------------------------    
   vectorized_tweet  = tfidf_vectorizer.transform([tweet_input])
   prediction= model.predict(vectorized_tweet )  
   #------------------------------------------------------------------------------------   
   if prediction==1:
       st.markdown("# Tweet is :red[Toxic] &nbsp; :pleading_face: ")
   else:
       print("Tweet is Non-Toxic")
       st.write("## Tweet is :green[Non-Toxic] &nbsp; :nerd_face:")
       #   streamlit run app.py


st.markdown("# ")       
st.markdown("# ")     
st.markdown("# ")    
st.markdown("# ")       
st.markdown("# ")     
      



st.text("-Created by banuprakash vellingiri ❤️")