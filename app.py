import openai
import streamlit as st
from prompts import PROMPT_SUMMARY, REFINE_PROMPT_SUMMARY
from llm_functions import initialize_llm, initialize_summary, split_text
from dotenv import load_dotenv
import os
load_dotenv()
#setting open api key
openai_api_key=os.getenv("OPENAI_API_KEY")
with st.container():
    st.markdown("""
    # MOM
    """)



uploaded_file = st.file_uploader("Upload a text file", type=["json"])
if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8")

# If there's a transcript (either from audio or uploaded text), process it.
if 'transcript' in locals() and transcript:
    transcript_chunks = split_text(data=transcript, chunk_size=14000, chunk_overlap=0)
    llm = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0)
    llm2 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0.5)
    summarize_chain = initialize_summary(llm=llm, chain_type="refine", question_prompt=PROMPT_SUMMARY, refine_prompt=REFINE_PROMPT_SUMMARY)
    summary = summarize_chain.run(transcript_chunks)
    st.code(summary)
