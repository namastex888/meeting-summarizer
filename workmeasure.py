import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import markdown
load_dotenv()
#setting open api key
openai_api_key=os.getenv("OPENAI_API_KEY")


#Function to initialize the large model
def initialize_llm(openai_api_key, model_name, temperature):
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    return llm

def split_text(data, chunk_size, chunk_overlap):
    text_splitter= TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts=text_splitter.split_text(data)
    #Creates documents for further processing
    docs=[Document(page_content=t) for t in texts]
    return docs

with st.container():
    st.markdown("""
    # WORK UNIT CALCULATOR
    """)

uploaded_file = st.file_uploader("Upload a text file", type=["json","srt","txt"])
if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8")

# If there's a transcript (either from audio or uploaded text), process it.
if 'transcript' in locals() and transcript:
       
    llm_gpt3 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0)
    llm_gpt4 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-0613", temperature=0.5)
    #summarize_chain = initialize_summary(llm=llm, chain_type="refine", question_prompt=PROMPT_SUMMARY, refine_prompt=REFINE_PROMPT_SUMMARY)
    
    # Map
    map_template = """
    Consider the following partial git log for some contributor:
    ----------
    {docs}
    ----------
    Analyze the provided partial git log of a contributor. The log will contain code additions ('+') and deletions ('-'), and the type of code (HTML, CSS, frontend non-HTML, lock files that doenst have any value). Calculate the work units using these factors:

    Code Addition: Assign 0.5 points for each new line of HTML or CSS code added, 1 point for frontend non-HTML code, and 3 points for backend code.

    Code Modification: Assign 0.5 points for each existing line modified, regardless of code type.

    Code Deletion: Assign 0.2 points for each line deleted, regardless of code type.

    Affected Files: Add 3 points for each file affected.

    Code Review: Assign 4 points for each code review conducted.

    Complexity Points: For each function, calculate its cyclomatic complexity (number of linearly independent paths through a program's source code). Assign 0.5 points for each point of cyclomatic complexity.

    Sum these to get the total work units for this log. The output should be in the following format:

    Code Additions: [number of lines added] lines ([points for HTML] points HTML, [points for frontend] points frontend, [points for backend] points backend) Code Modifications: [number of lines modified] lines ([points] points) Code Deletions: [number of lines deleted] lines ([points] points) Affected Files: [number of affected files] ([points] points) Code Reviews: [number of code reviews] ([points] points) Complexity Points: [number of complexity points] ([points] points) Total Points: [total points]

    Please ensure the response is generated consistently with these instructions.
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)
    
    # Reduce
    reduce_template = """
    Given the work units assigned to this worker, calculate the total contribution. Note that all chunks belong to the same person.
    -----
    {doc_summaries}.
    -----
    Sum these to get the total work units for this log. The output should be in the following format:
    Math Reasoning: [math reasoning]
    Total Points: [total points]"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # Run chain
    reduce_chain = LLMChain(llm=llm_gpt4, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=8000,
    )
    
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    
    transcript_chunks = split_text(data=transcript, chunk_size=13000, chunk_overlap=0)
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=12288, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(transcript)
    
    summary = map_reduce_chain.run(transcript_chunks)

    html = markdown.markdown(summary)
    st.markdown(html, unsafe_allow_html=True)
    st.code(summary)
    