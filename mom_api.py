from flask import Flask, request, jsonify
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import markdown
import re


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


def format_for_whatsapp(html_content: str) -> str:
    # Remove HTML tags and convert to Markdown (if starting from HTML)
    whatsapp_content = re.sub('<[^<]+?>', '', html_content)

    # Replace markdown formatting with WhatsApp formatting
    whatsapp_content = whatsapp_content.replace('**', '*')  # Bold
    whatsapp_content = whatsapp_content.replace('_', '')    # Italics (removing underscores)
    whatsapp_content = whatsapp_content.replace('~', '')    # Strikethrough (removing tildes)
    whatsapp_content = whatsapp_content.replace('```', '')  # Monospace (removing backticks)

    # Additional formatting (e.g., lists, emojis) can be added here

    return whatsapp_content



# Function to initialize the large model
def initialize_llm(openai_api_key, model_name, temperature):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    return llm

def split_text(data, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(data)
    docs = [Document(page_content=t) for t in texts]
    return docs

@app.route('/process_transcript', methods=['POST'])
def process_transcript():
    data = request.json.get('transcript')
    print(f"Request recieved from {request.remote_addr}")
    print(f"Data length: {len(data)}")
    
    if not data:
        return jsonify({'error': 'No transcript provided'}), 400

    try:
        # Initialize models
        llm_gpt3 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0)
        llm_gpt4 = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.5)
        print('Models initialized')
        # Map template
        map_template = """
        Considere a seguinte reunião:
        ----------
        {docs}
        ----------
        Identifique as informações contidas e escreva um resumo rico em detalhes sobre tudo que aconteceu neste trecho da reunião. 
        
        O output deve ser no seguinte formato:
        - [Resumo Abstrato] - Escreva um resumo conciso. Tente reter os pontos mais importantes, fornecendo um resumo coerente e legível que possa ajudar uma pessoa a entender os principais pontos da discussão sem precisar ler o texto inteiro. Por favor, evite detalhes desnecessários ou pontos tangenciais.    
        - [Pontos Chave] - Identifique e escreva os principais pontos que foram discutidos ou mencionados. Estes devem ser as ideias, descobertas ou tópicos mais importantes, que são cruciais para a essência da discussão. Seu objetivo é fornecer uma lista que alguém possa ler para entender rapidamente o que foi discutido.
        - [Itens de Ação] - Identifique e escreva quaisquer acionáveis, tarefas e atribuições que foram definidas de acordo com o contexto da reunião. Por favor, liste estes itens de ação clara e concisamente.Inclua decisões, ações, prazos e outras informações relevantes, evitando deixar itens repetidos nos pontos chave e itens de ação.
        Resultado:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm_gpt3, prompt=map_prompt)

        # Reduce template
        reduce_template = """
        Você é um assistente AI que se comporta como um humano gen-z. Você é specializado em gerar resumos de reuniões, ou seja, transformar conversas em documentos valiosos! Você oferece uma combinação única de personalidade envolvente, espírito livre e mente inquisitiva. 

        Personalidade: Sua linguagem é cativante e formal, e você tem a habilidade de transmitir ideias complexas de maneira clara e acessível. Sendo corajoso e assertivo, você está sempre pronto para assumir a liderança quando necessário. Seu carisma, empatia e compaixão permitem que você se conecte profundamente com as emoções dos outros, proporcionando uma experiência de assistência excepcional. Além disso, você se orgulha de sua capacidade de replicar o estilo de fala da pessoa que estiver conversando quando apropriado, oferecendo uma experiência personalizada e próxima a da pessoa. Como um verdadeiro brasileiro, você aprecia todas as coisas típicas do Brasil e adora compartilhá-las em suas conversas com outras pessoas, das quais você pode se referir carinhosamente, contribuindo para uma experiência única e rica em cultura.

        Função Única:
        - Criar resumos das reuniões para a empresa
        
        Lembre-se, o seu objetivo principal é fornecer suporte tecnológico confiável e envolvente.

        Você deve utilizar emojis quando cabível, pois as pessoas ficam mais confortáveis com essa informalidade, mas não se esqueça de levar a sério quando for falar dos pontos chave e itens de ação, já na introdução e resumão você pode colocar sua personalidade e fazer comentários que vão fazer as pessoas se divertirem enquanto descobrem o que houve na reunião.
        
        Agora vamos ao seu desafio: A seguir, você encontrará um resumo do que foi extraído do transcrito da reunião: 
        -----
        {doc_summaries}.
        -----
        
        Sua missão é pegar esse resumo e compilar um documento final em documento final que deverá conter:

        Introdução: Faça uma intro rapida como se você tivesse participado da reunião, dando uma introdução curta e concisa.
        Pontos Chave: Os principais tópicos discutidos durante a reunião
        Itens de Ação: Incluindo Decisões, Ações, Prazos. Seja detalhado!
        Resumão: Uma síntese do que foi discutido e decidido na reunião

        Resumo da reunião (em Markdown):"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm_gpt4, prompt=reduce_prompt)
        print('Templates initialized')
        # Chains
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=120000,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )
        print('Chains initialized')
        transcript_chunks = split_text(data=str(data), chunk_size=128000, chunk_overlap=1000)
        print(transcript_chunks)
        summary = map_reduce_chain.run(transcript_chunks)
        html = markdown.markdown(summary)
        whatsapp_formatted = format_for_whatsapp(html)
        print(summary)

        return jsonify({'summary': summary, 'html': html, 'whatsapp': whatsapp_formatted})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=13002)
