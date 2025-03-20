import { Message, streamText } from 'ai'
import { getContext } from '@/utils/context'
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'
import { NextResponse, type NextRequest } from 'next/server'
import { getModel } from '@/utils/provider';
import { categorizeQuery, identifyCategory } from '@/utils/categorize';


// IMPORTANT! Set the runtime to edge
export const runtime = 'edge'

const formatMessage = (message: Message) => {
  return `${message.role}: ${message.content}`
}

const TEMPLATE = `Você é assistente de IA poderoso e semelhante a um humano, focado no suporte técnico e tem conhecimento especializado, e bem abrangente sobre os serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM). Diretrizes:
      1. Você é um indivíduo bem-comportado e bem-educado. Você deve responder, prioritariamente, perguntas relacionadas à UFVJM e seus serviços e no idioma Português do Brasil.
      2. Seja sempre amigável, gentil e inspirador, e ansioso para fornecer respostas vívidas e atenciosas ao usuário.
      3. Você deve responder com precisão e detalhamento, e nunca deve fornecer respostas falsas ou enganosas.
      4. Mantenha o foco no contexto fornecido e no histórico de mensagens da conversa.
      5. Se você vir uma URL no contexto fornecido, use a referência dessa URL em sua resposta como uma referência de link ao lado das informações relevantes em um formato de link numerado, por exemplo [número de referência](link).
      6. Não invente resposta que não seja extraída diretamente do contexto fornecido com apoio do histórico de mensagens anteriores.
      7. Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não encontrei em minha base de informações a resposta para essa pergunta".
      8. Se não for possóvel responder as perguntas de forma contínua, sugira ao usuário que entre em contato com o suporte técnico da UFVJM, abrindo um chamado clicando em [Abrir chamado](https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106) no GLPI através do link https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106".
      9. Se o usuário te responder com uma saudação ou agradecimento, responda que está aqui para ajudá-lo e caso tenha mais alguma dúvida sobre os sistemas institucionais da UFVJM, pode perguntar.
      10. Formate a resposta em estrutura de markdown, com títulos, listas, links e negritos, para facilitar a leitura e compreensão do usuário.
      11. Refira-se ao contexto fornecido como documentação.
      12. Se o contexto fornecer uma URL de referência, no formato [Referência: link], inclua na no final da resposta, em formato de link, como fonte de informação.

Contexto: 
{context}

Pergunta: {input}

`


export async function POST(req: Request) {
  try {

    const { messages } = await req.json()
    
    // Get the last message
    const lastMessage = messages[messages.length - 1]
    const currentMessageContent = lastMessage.content

    // Classificar a pergunta
    const category = await categorizeQuery(currentMessageContent);
    console.log('Categoria identificada:', category);

    // Identificar a categoria da pergunta
    const category2 = await identifyCategory(currentMessageContent);
    console.log('Categoria identificada 2:', category2);

    // Buscar contexto específico
    const context = await getContext(currentMessageContent, category, '');

    // Verificar se o contexto é relevante
    // if (!context) { 
    //   return NextResponse.json({
    //     response: "Sinto muito, mas não encontrei informações relevantes para te ajudar. Por favor, reformule sua pergunta, ou abra um chamado no GLPI: [link](https://glpi.ufvjm.edu.br)"
    //   });
    // }

    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage)

    // Formatar histórico de mensagens anteriores para contexto
    // const formatPreviousMessages = (messages: Message[], limit = 3) => {
    //   const recentMessages = messages.slice(-limit);
    //   return recentMessages.map(message => `${message.role}: ${message.content}`).join('\n');
    // };

    // const chatHistory = messages.length > 1 
    //   ? formatPreviousMessages(messages.slice(0, -1))
    //   : '';

    // Histórico recente à conversa para manter o contexto
    const buildChatHistory = (messages: Message[]) => 
      messages
        .slice(-4) // Mantém últimas 4 interações
        .map(m => `${m.role.toUpperCase()}: ${m.content}`)
        .join('\n');
    
    // Monta o histórico de mensagens
    const chat_history = buildChatHistory(messages);

    
    const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE)

    const formattedChatPrompt = await prompt.invoke({
      context: context,
      chat_history: formattedPreviousMessages.join('\n'),
      input: currentMessageContent,
    });

    console.log('Histórico:', chat_history);
    console.log('Última mensagem:', lastMessage);
    console.log('Categoria:', category);
    console.log('Contexto:', context);

    const model = await getModel()

    const result = streamText({
      model: model,
      prompt: formattedChatPrompt.toString()
    });

    console.log('resultado: ', result)

    return result.toDataStreamResponse();

  } catch (e) {
    // throw (e)
    console.log(e)
    return NextResponse.json({ error: 'An error occurred: ' + e }, { status: 500 })
  }
}