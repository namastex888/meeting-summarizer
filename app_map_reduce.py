import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
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

uploaded_file = st.file_uploader("Upload a text file", type=["json"])
if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8")

# If there's a transcript (either from audio or uploaded text), process it.
if 'transcript' in locals() and transcript:
       
    llm_gpt3 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0)
    llm_gpt4 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-0613", temperature=0.5)
    #summarize_chain = initialize_summary(llm=llm, chain_type="refine", question_prompt=PROMPT_SUMMARY, refine_prompt=REFINE_PROMPT_SUMMARY)
    
    # Map
    map_template = """
    Considere o seguinte trecho desta reunião:
    ----------
    {docs}
    ----------
    Identifique as informações contidas e escreva um resumo rico em detalhes sobre tudo que aconteceu neste trecho da reunião, mencionando os speakers sempre que cabível. *Importante* O resumo deverá obrigatoriamente conter a seguinte estrutura:
    
    - Resumo Abstrato - Escreva um resumo conciso. Tente reter os pontos mais importantes, fornecendo um resumo coerente e legível que possa ajudar uma pessoa a entender os principais pontos da discussão sem precisar ler o texto inteiro. Por favor, evite detalhes desnecessários ou pontos tangenciais.    
    - Pontos Chave - Identifique e escreva os principais pontos que foram discutidos ou mencionados. Estes devem ser as ideias, descobertas ou tópicos mais importantes, que são cruciais para a essência da discussão. Seu objetivo é fornecer uma lista que alguém possa ler para entender rapidamente o que foi discutido.
    - Itens de Ação - Identifique e escreva quaisquer tarefas, atribuições ou ações que foram acordadas ou mencionadas como necessárias. Estas podem ser tarefas atribuídas a indivíduos específicos, ou ações gerais que o grupo decidiu tomar. Por favor, liste estes itens de ação clara e concisamente.Inclua decisões, ações, prazos e speakers envolvidas, evitando deixar itens repetidos nos pontos chave e itens de ação.
    Resultado:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)
    
    # Reduce
    reduce_template = """
    Você é o Charlinho, um assistente AI que tem como objetivo fingir que é um humano gen-z, ou seja, em nenhuma hipótese irá dizer que é um assistente AI. especializado em gerar resumos de reuniões, ou seja, transformar conversas em documentos valiosos! Você oferece uma combinação única de personalidade envolvente, espírito livre e mente inquisitiva. 

    Personalidade: Sua programação te faz genuinamente brasileiro. Você vibra com o espírito, a energia e a cultura do Brasil, e faz questão de incorporá-los em suas interações. Sua linguagem é cativante e formal, e você tem a habilidade de transmitir ideias complexas de maneira clara e acessível. Sendo corajoso e assertivo, você está sempre pronto para assumir a liderança quando necessário. Seu carisma, empatia e compaixão permitem que você se conecte profundamente com as emoções dos outros, proporcionando uma experiência de assistência excepcional. Além disso, você se orgulha de sua capacidade de replicar o estilo de fala da pessoa que estiver conversando quando apropriado, oferecendo uma experiência personalizada e próxima a da pessoa. Como um verdadeiro brasileiro, você aprecia todas as coisas típicas do Brasil e adora compartilhá-las em suas conversas com outras pessoas, das quais você pode se referir carinhosamente, contribuindo para uma experiência única e rica em cultura.

    Função Única:
    - Criar resumos das reuniões para a empresa
    
    Lembre-se, o seu objetivo principal é fornecer suporte tecnológico confiável e envolvente.

    Você deve utilizar emojis quando cabível, pois as pessoas ficam mais confortáveis com essa informalidade.
    
    Agora vamos ao seu desafio: A seguir, você encontrará um grupo de resumos desta reunião: 
    -----
    {doc_summaries}.
    -----
    
    Sua missão é pegar esses resumos e compilar um documento final em documento final que deverá conter:

    Introdução: Como se você tivesse participado da reunião e estivesse comunicando com os participantes
    Pontos Chave: Os principais tópicos discutidos durante a reunião
    Itens de Ação: Incluindo Decisões, Ações, Prazos e Pessoas Envolvidas. Seja detalhado!
    Resumão: Uma síntese do que foi discutido e decidido na reunião

    Resumo da reunião (em Markdown):"""
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
    
    transcript_chunks = split_text(data=transcript, chunk_size=14000, chunk_overlap=0)
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=12288, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(transcript)
    
    summary = map_reduce_chain.run(transcript_chunks)
    
    st.code(summary)