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
    # MOM
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
    Analyze the git commit log to estimate the work done. Lines with '-' are code removals, and lines with '+' are code additions. Assign work units based on the following factors:

    Complexity: If the code involves intricate logic, such as changes in the React components or Next.js pages, add 2 work units. For less complex changes, add 1 work unit.

    Importance: If the change involves critical components like LoginModal.tsx or Hero.tsx, add 2 work units. For changes in less critical components, add 1 work unit.

    Workload: Analyze code removed vs code added, and assign 1 work unit for every 10 lines changed. However, remember that the quality of code can outweigh the quantity, and small, efficient code changes can be more valuable than large, inefficient ones.

    Sum these to get the total work units for this commit. Provide the output in the following format: '[total work units] work units.'
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)
    
    # Reduce
    reduce_template = """
    Given the work units assigned to this worker, calculate the total contribution. Note that all chunks belong to the same person.
    -----
    {doc_summaries}.
    -----
    Sum the work units of get the total work units.
    
    Results (in Markdown):"""
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
    