from langchain.prompts import PromptTemplate

prompt_template_summary = """
    Considere o seguinte trecho desta reunião:
    ----------
    {text}
    ----------
    Identifique as informações contidas e escreva um resumo rico em detalhes sobre tudo que aconteceu neste trecho da reunião, mencionando os speakers sempre que cabível. *Importante* O resumo deverá obrigatoriamente conter a seguinte estrutura:
    
    - Resumo Abstrato - Escreva um resumo conciso. Tente reter os pontos mais importantes, fornecendo um resumo coerente e legível que possa ajudar uma pessoa a entender os principais pontos da discussão sem precisar ler o texto inteiro. Por favor, evite detalhes desnecessários ou pontos tangenciais.    
    - Pontos Chave - Identifique e escreva os principais pontos que foram discutidos ou mencionados. Estes devem ser as ideias, descobertas ou tópicos mais importantes, que são cruciais para a essência da discussão. Seu objetivo é fornecer uma lista que alguém possa ler para entender rapidamente o que foi discutido.
    - Itens de Ação - Identifique e escreva quaisquer tarefas, atribuições ou ações que foram acordadas ou mencionadas como necessárias. Estas podem ser tarefas atribuídas a indivíduos específicos, ou ações gerais que o grupo decidiu tomar. Por favor, liste estes itens de ação clara e concisamente.Inclua decisões, ações, prazos e speakers envolvidas, evitando deixar itens repetidos nos pontos chave e itens de ação.
    Resultado:
"""

PROMPT_SUMMARY= PromptTemplate(template=prompt_template_summary, input_variables=["text"])

refine_template_summary = (
"""
Seu trabalho é escrever o resumo final. Aqui está o resumo que temos até o momento:
{existing_answer}
Nós devemos refinar este documento
------------
{text}
------------
Dado o novo contexto, refine o documento anterior, acrescentando e enriquecendo com novas informações adquiridas sobre a reunião.

Quando o trancrito chegar ao fim, escreva a versão final do documento

"""
)
REFINE_PROMPT_SUMMARY=PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_summary,
)
