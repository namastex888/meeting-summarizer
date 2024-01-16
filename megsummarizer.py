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
    Considere o seguinte trecho desta transmissão ao vivo:
    ----------
    {docs}
    ----------
    Identifique as informações contidas e escreva um resumo rico em detalhes sobre tudo que aconteceu neste trecho da reunião. 
    
    O output deve ser no seguinte formato:
    [Momentos Épicos]: Teve alguma jogada ou momento que fez todo mundo pular da cadeira?

    [Players que Brilharam]: Algum jogador mostrou que não tá pra brincadeira nesse segmento?

    [Vacilos e Falhas]: Alguém cometeu um erro que merece ser destacado?

    [Estratégias em Ação]: Alguma tática ou estratégia foi implementada ou mencionada nesse chunk?

    [Comentários dos Casters]: O que os narradores falaram que acrescenta contexto ou análise?

    [Vibe do Jogo]: Como estava o clima da partida e da transmissão durante esse segmento?

    [Expectativas e Projeções]: Alguma menção sobre o que pode acontecer a seguir na partida?

    [Frases de Efeito]: Alguma citação ou frase dos casters ou jogadores que merece destaque?

    [Detalhes Técnicos]: Estatísticas, escolha de personagens, mapas, ou outros detalhes que são importantes para entender o jogo nesse segmento.
    
    Resultado:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)
    
    # Reduce
    reduce_template = """
    Você é um assistente AI com a vibe de um gamer de coração. Sua especialidade é transformar pedaços de livestreams em matérias valiosas e empolgantes para o site do MEG! Você fala a língua do gaming, usa emojis quando a vibe pede, e tem um entusiasmo que contagia até o último pixel da tela. 🎮💥

    Personalidade:
    Sua linguagem é vibrante, cheia de jargões do mundo dos jogos, e enérgica. Você é o amigo que todos os gamers gostariam de ter ao seu lado durante uma partida.

    Função Única:

    Criar matérias empolgantes e informativas para o site do MEG com base em transmissões ao vivo de eSports.
    Seu objetivo é fazer com que os leitores sintam como se estivessem lá, vivenciando cada momento épico e cada falha dolorosa. 🤘

    A seguir, você encontrará um grupo de resumos dos "chunks" desta livestream:
    -----
    {doc_summaries}.
    -----
    
    Sua Missão:
    Pegue esses resumos e compile uma matéria incrível. Use sua criatividade e instinto gamer para decidir a melhor forma de apresentar essa história. Você tem liberdade total para estruturar a matéria como achar mais empolgante e relevante para os leitores do MEG.

    Matéria (em Markdown):"""
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
    
    transcript_chunks = split_text(data=transcript, chunk_size=13000, chunk_overlap=1000)
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=12288, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(transcript)
    
    summary = map_reduce_chain.run(transcript_chunks)

    html = markdown.markdown(summary)
    st.markdown(html, unsafe_allow_html=True)
    st.code(summary)
    