import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from streamlit_option_menu import option_menu
from sentiment_analysis import SentimentAnalysis
from keyword_extraction import KeywordExtractor
from part_of_speech_tagging import POSTagging
from emotion_detection import EmotionDetection
from named_entity_recognition import NamedEntityRecognition

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    return SentimentAnalysis()

@st.cache(allow_output_mutation=True)
def load_keyword_model():
    return KeywordExtractor()

@st.cache(allow_output_mutation=True)
def load_pos_model():
    return POSTagging()

@st.cache(allow_output_mutation=True)
def load_emotion_model():
    return EmotionDetection()

@st.cache(allow_output_mutation=True)
def load_ner_model():
    return NamedEntityRecognition()


sentiment_analyzer = load_sentiment_model()
keyword_extractor = load_keyword_model()
pos_tagger = load_pos_model()
emotion_detector = load_emotion_model()
ner = load_ner_model()

example_text = "This is example text that contains both names of organizations like Hugging Face and cities like New York, all while portraying an upbeat attitude."

with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Sentiment Analysis",
                                "Keyword Extraction",
                                "Part of Speech Tagging",
                                "Emotion Detection",
                                "Named Entity Recognition"],
                       icons=["house-door",
                              "chat-dots",
                              "key",
                              "tag",
                              "emoji-heart-eyes",
                              "building"],
                       default_index=0
                       )

st.title('Open-source NLP')

if page == "Welcome!":
    st.header('Welcome!')

    st.markdown("![Alt Text](https://media.giphy.com/media/2fEvoZ9tajMxq/giphy.gif)")
    st.write(
        """
     
     
        """
    )

    st.subheader("Quickstart")
    st.write(
        """
        Replace the example text below and flip through the pages in the menu to perform NLP tasks on-demand!
        Feel free to use the example text for a test run. 
        """
    )

    text = st.text_area("Paste text here", value=example_text)

    st.subheader("Introduction")
    st.write("""
        Hello! This application is a celebration of open-source and the power that programmers have been granted today
        by those who give back to the community. This tool was constructed using Streamlit, Huggingface Transformers, 
        Transformers-Interpret, NLTK, Spacy, amongst other open-source Python libraries and models. 
        
        Utilizing this tool you will be able to perform a multitude of Natural Language Processing Tasks on a range of
        different tasks. All you need to do is paste your input, select your task, and hit the start button! 
        
        * This application currently supports:
            * Sentiment Analysis
            * Keyword Extraction
            * Part of Speech Tagging
            * Emotion Detection
            * Named Entity Recognition
            
        More features may be added in the future including article/tweet/youtube input, improved text annotation, model quality improvements, 
        depending on community feedback. Please reach out to me at miesner.jacob@gmail.com or at my Linkedin page listed 
        below if you have ideas or suggestions for improvement.
        
        If you would like to contribute yourself, feel free to fork the Github repository listed below and submit a merge request.
        """
    )
    st.subheader("Notes")
    st.write(
        """
        * This dashboard was constructed by myself, but every resource used is open-source! If you are interested in my other works you can view them here:
        
           [Project Github](https://github.com/MiesnerJacob/nlp-dashboard)
           
           [Jacob Miesner's Github](https://github.com/MiesnerJacob)
           
           [Jacob Miesner's Linkedin](https://www.linkedin.com/in/jacob-miesner-885050125/)
           
           [Jacob Miesner's Website](https://www.jacobmiesner.com)
              
        * The prediction justification for some of the tasks are printed as the model views them. For this reason the text may contain special tokens like [CLS] or [SEP] or even hashtags splitting words. If you are are familiar with language models you will recognize these, if you do not have prior experience with language models you can ignore these characters.  
        """
    )

elif page == "Sentiment Analysis":
    st.header('Sentiment Analysis')
    st.markdown("![Alt Text](https://media.giphy.com/media/XIqCQx02E1U9W/giphy.gif)")
    st.write(
        """


        """
    )

    text = st.text_area("Paste text here", value=example_text)

    if st.button('🔥 Run!'):
        with st.spinner("Loading..."):
            preds, html = sentiment_analyzer.run(text)
            st.success('All done!')
            st.write("")
            st.subheader("Sentiment Predictions")
            st.bar_chart(data=preds, width=0, height=0, use_container_width=True)
            st.write("")
            st.subheader("Sentiment Justification")
            raw_html = html._repr_html_()
            st.components.v1.html(raw_html, height=500)

elif page == "Keyword Extraction":
    st.header('Keyword Extraction')
    st.markdown("![Alt Text](https://media.giphy.com/media/xT9C25UNTwfZuk85WP/giphy-downsized-large.gif)")
    st.write(
        """


        """
    )

    text = st.text_area("Paste text here", value=example_text)

    max_keywords = st.slider('# of Keywords Max Limit', min_value=1, max_value=10, value=5, step=1)

    if st.button('🔥 Run!'):
        with st.spinner("Loading..."):
            annotation, keywords = keyword_extractor.generate(text, max_keywords)
            st.success('All done!')

        if annotation:
            st.subheader("Keyword Annotation")
            st.write("")
            annotated_text(*annotation)
            st.text("")

        st.subheader("Extracted Keywords")
        st.write("")
        df = pd.DataFrame(keywords, columns=['Extracted Keywords'])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Keywords to CSV', csv, file_name='news_intelligence_keywords.csv')

        data_table = st.table(df)

elif page == "Part of Speech Tagging":
    st.header('Part of Speech Tagging')
    st.markdown("![Alt Text](https://media.giphy.com/media/WoWm8YzFQJg5i/giphy.gif)")
    st.write(
        """


        """
    )

    text = st.text_area("Paste text here", value=example_text)

    if st.button('🔥 Run!'):
        with st.spinner("Loading..."):
            preds = pos_tagger.classify(text)
            st.success('All done!')
            st.write("")
            st.subheader("Part of Speech tags")
            annotated_text(*preds)
            st.write("")
            st.components.v1.iframe('https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html', height=1000)

elif page == "Emotion Detection":
    st.header('Emotion Detection')
    st.markdown("![Alt Text](https://media.giphy.com/media/fU8X6ozSszyEw/giphy.gif)")
    st.write(
        """


        """
    )

    text = st.text_area("Paste text here", value=example_text)

    if st.button('🔥 Run!'):
        with st.spinner("Loading..."):
            preds, html = emotion_detector.run(text)
            st.success('All done!')
            st.write("")
            st.subheader("Emotion Predictions")
            st.bar_chart(data=preds, width=0, height=0, use_container_width=True)
            raw_html = html._repr_html_()
            st.write("")
            st.subheader("Emotion Justification")
            st.components.v1.html(raw_html, height=500)

elif page == "Named Entity Recognition":
    st.header('Named Entity Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/lxO8wdWdu4tig/giphy.gif)")
    st.write(
        """


        """
    )

    text = st.text_area("Paste text here", value=example_text)

    if st.button('🔥 Run!'):
        with st.spinner("Loading..."):
            preds, ner_annotation = ner.classify(text)
            st.success('All done!')
            st.write("")
            st.subheader("NER Predictions")
            annotated_text(*ner_annotation)
            st.write("")
            st.subheader("NER Prediction Metadata")
            st.write(preds)
