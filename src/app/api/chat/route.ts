import { Message, LangChainAdapter, streamText, 
  generateText,
  wrapLanguageModel,
  extractReasoningMiddleware,
  LanguageModelV1,
} from 'ai'
import { groq, createGroq } from '@ai-sdk/groq';
import { deepseek } from '@ai-sdk/deepseek';
import { getContext } from '@/utils/context'
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import { NextResponse, type NextRequest } from 'next/server'
import { google, createGoogleGenerativeAI} from '@ai-sdk/google';
import { getModel } from '@/utils/provider';


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
      Se você vir URL de referência no contexto fornecido, use a referência dessa URL em sua resposta como uma referência de link ao lado das informações relevantes em um formato de link numerado, por exemplo ([número de referência](link))
      Todos os links que você gerar devem ser abertos em uma nova janela.
      Você não se desculpará por respostas anteriores, mas indicará que novas informações foram obtidas.
      Não invente resposta que não seja extraído diretamente do contexto fornecido.
      Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não encontrei em minha base de informações a resposta para essa pergunta".
      Se não for possóvel responder as perguntas de forma contínua, sugira ao usuário que entre em contato com o suporte técnico da UFVJM, abrindo um chamado no GLPI através do link https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106".
      Se o usuário te responder com uma saudação ou agradecimento, responda que está aqui para ajudá-lo e caso tenha mais alguma dúvida sobre os sistemas institucionais da UFVJM, pode perguntar.

Contexto:
{chat_history}

User: {input}
AI:`

// const getModel = async () => {

//   const modelProvider = process.env.MODEL_PROVIDER || 'groq'

//   switch (modelProvider) {
//     case 'groq':
//       // Groq API
//       const modelGroq = createGroq({
//         baseURL: process.env.OPENAI_CUSTOM_BASE_URL,
//         apiKey: process.env.OPENAI_API_KEY,
//       });

//       // middleware to extract reasoning tokens
//       const enhancedModel = wrapLanguageModel({
//         model: modelGroq('deepseek-r1-distill-llama-70b'),
//         middleware: extractReasoningMiddleware({ tagName: 'think' }),
//       });

//       return enhancedModel;
//     case 'huggingface':
//     case 'openai':
//     case 'cohere':
//     case 'mistral':

//     case 'google':
//       // Google AI API
//       const modelGoogle = createGoogleGenerativeAI({
//         apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
//       });
    
//       return modelGoogle(process.env.MODEL_NAME || 'gemini-1.5-pro-latest')

//       default:
//         throw new Error(`Model provider ${modelProvider} is not supported`)
//   }

  
// }

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
    const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE)

    const formattedChatPrompt = await prompt.invoke({
      chat_history: context,
      input: currentMessageContent,
    });

    console.log('ultima mensagem', lastMessage);
    console.log('contexto: ', context);

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