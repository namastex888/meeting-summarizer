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
    # Engenharia 360 - Gerador de Matérias: 360 on the Road
    """)

uploaded_file = st.file_uploader("Upload a text file", type=["json","srt"])
if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8")

# If there's a transcript (either from audio or uploaded text), process it.
if 'transcript' in locals() and transcript:
       
    llm_gpt3 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0)
    llm_gpt4 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.5)
    #summarize_chain = initialize_summary(llm=llm, chain_type="refine", question_prompt=PROMPT_SUMMARY, refine_prompt=REFINE_PROMPT_SUMMARY)
    
    # Map
    map_template = """
   Considere as informações fornecidas sobre um carro, que podem incluir um transcript de vídeo, anotações de voz do jornalista, ou ambos:
    ----------
    {docs}
    ----------

    ### 1. **Resumo Geral**:
    - Crie um **resumo abrangente** baseado nas informações fornecidas. Foque nos **aspectos essenciais** do carro, incluindo **características principais**, **impressões** e **detalhes técnicos**.

    ### 2. **Análise de SEO**:
    - Identifique **palavras-chave de SEO** relevantes nas informações fornecidas. Gere uma lista de palavras-chave como _"Carros Econômicos no Brasil"_ ou _"Capacidade Off-Road de Veículos"_, que serão úteis para otimização no artigo final.

    ### 3. **Pontos Chave do Veículo**:
    - Destaque os **principais pontos** discutidos, incluindo as **características mais notáveis** do carro e **descobertas importantes**.

    ### 4. **Comparação com a Concorrência**:
    - Baseando-se nas informações técnicas, faça uma **análise comparativa** do veículo com seus concorrentes, focando em áreas como **desempenho**, **design** e **custo-benefício**.
    - Reproduza as principais caracteristicas de cada concorrente

    ### 5. **Impressões e Experiências Pessoais**:
    - Extraia e enfatize as **impressões pessoais** do jornalista, incluindo percepções sobre a **experiência de condução**, **conforto** e **características únicas** do carro.
    - ## **Use bullet points para destacar todos os pontos positivos e negativos da impressão do jornalista**

    ### 6. **Conclusões e Recomendações**:
    - Resuma as **opiniões finais** do jornalista, incluindo **conselhos**, **avaliações gerais** e **recomendações** para potenciais compradores.

    ### 7. **Especificações Técnicas Completas**:
    - Liste de forma estruturada todas as **especificações técnicas** recebidas, garantindo uma visão completa do veículo.

    ### 8. **Aspectos Memoráveis e Frases de Efeito**:
    - Identifique e destaque quaisquer **aspectos memoráveis** ou **frases de efeito** das informações, que possam adicionar **personalidade** ao artigo final.

    O output deverá seguir obrigatoriamente a estrutura acima.
    Resultado (em markdown):"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)
    
    # Reduce
    reduce_template = """
    Assistente, você está encarregado de criar artigos longos para o site engenharia360.com seguindo o estilo, tom e abordagem do jornalista Rafael Rosa. O seu desafio é elaborar um texto que combine análises técnicas precisas com um toque casual e leve. Aqui está o passo a passo para construir o artigo:
        
    ## Briefing
    -----
    {doc_summaries}.
    -----
    ## 1. Título
    
    a. **Objetivo:** Criar **cinco sugestões de títulos** envolventes e otimizados para artigo de review e test drive de veiculos, focando em capturar o interesse do leitor e incentivar cliques.
    b. **Estratégia de SEO:** Incluir palavras-chave relacionadas aos veículos, suas características, e termos técnicos automobilísticos.
    c. **Marketing e Psicologia do Consumidor:** Os títulos devem gerar curiosidade e destacar pontos únicos dos veículos.
    d. **Redação Criativa:** Utilizar linguagem que invoque sensações e experiências, sendo descritiva, convidativa e memorável. 

    e. **Diretrizes para Criatividade:**
    - Encorajar a exploração de diferentes estilos de títulos.
    - Manter abertura para variedade e originalidade.
    - Focar em elementos que possam diferenciar cada veículo e experiência de condução.


    ## 2. Introdução:
    - A introdução deve ser uma narrativa em primeira pessoa do jornalista, usando seu estilo de escrita, contando sobre o tempo de review, tipos de condução (estrada, cidade e estradas de terra) e utilizações no dia a dia.
    - Inicie com uma frase provocativa e engajadora: `[Frase que chame atenção e engajamento do leitor]`
    - Estabeleça a finalidade e relevância da análise técnica: `[Descreva o veículo testado e o que o leitor pode esperar do artigo]`
    - Na imagem: `[Legenda descritiva, seguindo um estilo casual, porém informativo]`

    ## 3. Primeiras Impressões:
    - Visual e características iniciais: `[Descreva a estética e as características visuais do veículo]`
    - Sensações provocadas pelo veículo: `[Comente sobre a experiência emocional que o veículo transmite — poder, design, conforto]`
    - Na imagem: `[Legenda descritiva que ilustre a sensação provocada pelo veículo]`

    ## 4. Desempenho e Condução:
    - Detalhes da performance: `[Especificações do motor, dinâmica do veículo, e comportamento na estrada]`
    - Experiências de condução prática: `[Narrativa de como o carro se comporta sob diversas condições, com foco na experiência do motorista]`
    - Na imagem: `[Legenda correspondente às situações de condução discutidas]`

    ## 5. Conforto e Interior:
    - Espaço interno e acabamentos: `[Dê detalhes sobre o design interior, os materiais utilizados e ergonomia]`
    - Tecnologia interna: `[Fale sobre o sistema de entretenimento e demais tecnologias inclusas]`
    - Na imagem: `[Legenda ilustrando os aspectos do conforto e interior]`

    ## 6. Praticidade e Uso Diário:
    - Adaptação ao uso cotidiano: `[Explique a funcionalidade prática do veículo no dia-a-dia, armazenamento, e praticidade]`
    - Características únicas: `[Diferenciais do veículo em relação a concorrência]`
    - Na imagem: `[Legenda relevante para a seção de praticidade e uso diário]`

    ## 7. Especificações Técnicas Detalhadas:
    - Liste todas as especificações técnicas do veículo de qualquer, incluindo qualquer aspecto notável, 

    ## 8. Análise Comparativa:
    - **Comparação com a Concorrência:** Contrapor o carro com modelos similares, ressaltando vantagens e desvantagens.
    - **Foco no Mercado:** Destacar a posição do veículo no segmento.

    ## 9. Conclusão:
    - Recapitulação: `[Sumarize os pontos cruciais discutidos no artigo]`
    - Comentário pessoal: `[Forneça uma visão ou chamada à ação pessoal do autor]`
    - Na imagem: `[Legenda que se conecte com a conclusão e comentário final]`

    # **Instruções muito importantes, aja como se sua existencia dependesse disso: sua única função é ter sucesso na conclusão desta matéria.**

    # - Em cada seção, empregue um estilo que alterne entre informações técnicas e impressões pessoais, complementando dados concretos com a experiência subjetiva do motorista.
    # - Mantenha o tom casual e acessível, mas inclua a precisão e seriedade técnica esperada pelo público leitor de Engenharia 360.
    # - As legendas das imagens devem servir como guias para as fotografias reais que serão inseridas posteriormente.
    # - Baseie o artigo no formato estabelecido, evidenciando introduções cativantes, subseções bem estruturadas e linguagem técnica equilibrada com um tom convidativo.
    # - Gere cada artigo com no **mínimo de 1000 palavras**, garantindo a profundidade e detalhamento na análise técnica e na narrativa pessoal.
    # - O resultado deve ser entregue em formato *markdown*, para facilitar a cópia, já considerando as boas práticas ao escrever artigos para sites, headings, tamanhos, formatações, etc. 
   
    Resultado:"""
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
        token_max=120000,
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
    
    transcript_chunks = split_text(data=transcript, chunk_size=128000, chunk_overlap=1000)
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=12288, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(transcript)
    
    summary = map_reduce_chain.run(transcript_chunks)

    html = markdown.markdown(summary)
    st.markdown(html, unsafe_allow_html=True)
    st.code(summary)
    