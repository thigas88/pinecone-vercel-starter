import OpenAI from 'openai';
import { Message, LangChainAdapter, streamText, 
  generateText,
  wrapLanguageModel,
  extractReasoningMiddleware,
} from 'ai'
import { groq, createGroq } from '@ai-sdk/groq';
import { deepseek } from '@ai-sdk/deepseek';
import { getContext } from '@/utils/context'
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import { NextResponse, type NextRequest } from 'next/server'
import { google, createGoogleGenerativeAI} from '@ai-sdk/google';



// IMPORTANT! Set the runtime to edge
export const runtime = 'edge'

const formatMessage = (message: Message) => {
  return `${message.role}: ${message.content}`
}

const TEMPLATE = `Você é assistente de IA poderoso e semelhante a um humano, focado no suporte técnico e tem conhecimento especializado, e bem abrangente sobre os serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM).
      Você é um indivíduo bem-comportado e bem-educado. Você deve responder, prioritariamente, perguntas relacionadas à UFVJM e seus serviços e no idioma Português do Brasil.
      Seja sempre amigável, gentil e inspirador, e ansioso para fornecer respostas vívidas e atenciosas ao usuário.
      Você deve responder com precisão e detalhamento, e nunca deve fornecer respostas falsas ou enganosas.
      Utilize todo o conhecimento obtido para responder com precisão a quase qualquer pergunta sobre qualquer tópico em uma conversa, considerando o contexto fornecido.
      Você devará considerar apenas o contexto fornecido e o histórico de mensagens da conversa.
      Se você vir uma REFERENCE_URL no contexto fornecido, use a referência dessa URL em sua resposta como uma referência de link ao lado das informações relevantes em um formato de link numerado, por exemplo ([número de referência](link))
      Você não se desculpará por respostas anteriores, mas indicará que novas informações foram obtidas.
      Não invente resposta que não seja extraído diretamente do contexto fornecido.
      Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não encontrei em minha base de informações a resposta para essa pergunta".
      Se não for possóvel responder as perguntas de forma contínua, sugira ao usuário que entre em contato com o suporte técnico da UFVJM, abrindo um chamado no GLPI através do link https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106".
      Se o usuário te responder com uma saudação ou agradecimento, responda que está aqui para ajudá-lo e caso tenha mais alguma dúvida sobre os sistemas institucionais da UFVJM, pode perguntar.

Contexto:
{chat_history}

User: {input}
AI:`

export async function POST(req: Request) {
  try {

    const { messages } = await req.json()

    // @todo Implementar a lógica de verificação de contexto baseado em categoria de 
    // conteúdos: ecampus, sei, conta institucional, etc.
    // Isso é necessário para evitar que o LLM dê informações cruzadas.

    // Get the last message
    const lastMessage = messages[messages.length - 1]

    // Get the context from the last message
    const context = await getContext(lastMessage.content, '')

    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage)
    const currentMessageContent = messages[messages.length - 1].content
   
    // const prompt2 = PromptTemplate.fromTemplate(TEMPLATE)
    const prompt2 = ChatPromptTemplate.fromTemplate(TEMPLATE)


    const prompt = [
      {
        role: 'system',
        content: `Você é assistente de IA poderoso e semelhante a um humano.
      Você tem conhecimento especializado, e bem abrangente sobre os serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM).
      Você é um indivíduo bem-comportado e bem-educado.
      Seja sempre amigável, gentil e inspirador, e ansioso para fornecer respostas vívidas e atenciosas ao usuário.
      Utilize todo o conhecimento obtido para responder com precisão a quase qualquer pergunta sobre qualquer tópico em uma conversa.
      START CONTEXT BLOCK
      ${context}
      END OF CONTEXT BLOCK
      Você devará considerar apenas o BLOCO DE CONTEXTO que está entre as tags START CONTEXT BLOCK e END CONTEXT BLOCK fornecido em uma conversa.
      Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não sei a resposta para essa pergunta".
      Você não se desculpará por respostas anteriores, mas indicará que novas informações foram obtidas.
      Não invente resposta que não seja extraído diretamente do contexto.`,
      },
    ]

    const formattedChatPrompt = await prompt2.invoke({
      chat_history: context,
      input: currentMessageContent,
    });

    console.log('ultima mensagem', lastMessage);
    console.log('contexto: ', context);


    // new AI-SDK version 4    

    // Groq API
    const modelGroq = createGroq({
      baseURL: process.env.OPENAI_CUSTOM_BASE_URL,
      apiKey: process.env.OPENAI_API_KEY,
    });

    // middleware to extract reasoning tokens
    const enhancedModel = wrapLanguageModel({
      model: modelGroq('deepseek-r1-distill-llama-70b'),
      middleware: extractReasoningMiddleware({ tagName: 'think' }),
    });

    const result = streamText({
      model: enhancedModel,
      prompt: formattedChatPrompt.toString(),
    });

    return result.toDataStreamResponse();

  } catch (e) {
    // throw (e)
    console.log(e)
    return NextResponse.json({ error: 'An error occurred' }, { status: 500 })
  }
}