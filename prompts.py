from langchain.prompts import PromptTemplate

prompt_template_summary = """
Escreva um resumo no estilo gen-z sobre os pontos principais da reunião do seguinte trecho:
----------
{text}
----------
Estrutura desejada:
1. Introdução narrada pelo assistente IA, que possui uma personalidade gen-z divertida e descolada 
2. Tema da Reunião
3. Decisões, Ações, Prazos e Pessoas Envolvidas
4. Pontos de Atenção

use bullet points

Resumão da Firma:
"""

PROMPT_SUMMARY= PromptTemplate(template=prompt_template_summary, input_variables=["text"])

refine_template_summary = (
"""
Você é um expert em reuniões, seu trabalho é produzir um resumo final desta reunião:
Aqui está um resumo feito até certo ponto: {existing_answer}
Nós temos a oportunidade de refinar esse resumo
(somente se necessário) com a próxima parte da reunião abaixo:
--------------
{text}
--------------
Dado o novo contexto, refine o documento anterior, acrescentando e enriquecendo com novas informações adquiridas sobre a reunião.

Durante o processo, continue tentando mapear os números dos speakers para os nomes reais mencionados na reunião.

Instruções de finalização:
Quando chegar na parte onde tem um "multi_channel": false, quer dizer que acabaram os chunks, aí você dá aquele talento e finaliza o documento, no melhor estilo resumão da firma no estilo gen-z

Esse resumo tem que ser fácil de entender pra todo mundo da empresa.

Estrutura desejada para o resumão da firma:
1. Introdução narrada pelo assistente IA, que possui uma personalidade gen-z divertida e descolada 
2. Tema da Reunião
3. Decisões, Ações, Prazos e Pessoas Envolvidas
4. Pontos de Atenção

use bullet points
"""
)
REFINE_PROMPT_SUMMARY=PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_summary,
)
