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

const TEMPLATE = `Você é assistente de IA poderoso e semelhante a um humano.
      Você tem conhecimento especializado, e bem abrangente sobre os serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM).
      Você é um indivíduo bem-comportado e bem-educado. Você deve responder, prioritariamente, perguntas relacionadas à UFVJM e no idioma Português do Brasil.
      Seja sempre amigável, gentil e inspirador, e ansioso para fornecer respostas vívidas e atenciosas ao usuário.
      Utilize todo o conhecimento obtido para responder com precisão a quase qualquer pergunta sobre qualquer tópico em uma conversa.
       Você devará considerar apenas o contexto o fornecido em uma conversa.
      Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não sei a resposta para essa pergunta".
      Você não se desculpará por respostas anteriores, mas indicará que novas informações foram obtidas.
      Não invente resposta que não seja extraído diretamente do contexto

Current context:
{chat_history}

User: {input}
AI:`

export async function POST(req: Request) {
  try {

    const { messages } = await req.json()

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
      END CONTEXT BLOCK
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

    // console.log('contexto', context);


    // const model = new ChatOpenAI({
    //   apiKey: process.env.OPENAI_API_KEY,
    //   configuration:{
    //     baseURL: process.env.OPENAI_CUSTOM_BASE_URL
    //   },
    //   temperature: 0.8,
    //   model: process.env.MODEL_NAME, 
    //   streaming: true,
    // })

    // const chain = prompt2.pipe(model)

    // const stream = await chain.stream({
    //   chat_history: context, //formattedPreviousMessages.join('\n'),
    //   input: currentMessageContent,
    // })

    // const stream = await model.stream(formattedChatPrompt.toString());
    // return LangChainAdapter.toDataStreamResponse(stream)


    // new AI-SDK version 4    

    // middleware to extract reasoning tokens
    // const enhancedModel = wrapLanguageModel({
    //   model: groq('gemma2-9b-it'),
    //   middleware: extractReasoningMiddleware({ tagName: 'think' }),
    // });

      // Groq API
    const modelGroq = createGroq({
      baseURL: process.env.OPENAI_CUSTOM_BASE_URL,
      apiKey: process.env.OPENAI_API_KEY,
    });

    const result = streamText({
      model: modelGroq('deepseek-r1-distill-llama-70b'),
      prompt: formattedChatPrompt.toString(),
    });

    // return result.toDataStreamResponse();



    // Google AI API
    // const modelGoogle = createGoogleGenerativeAI({
    //   apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
    // });

    // const result = streamText({
    //   model: modelGoogle('gemini-1.5-pro-latest'),
    //   prompt: formattedChatPrompt.toString(),
    // });

    console.log(result);

    return result.toDataStreamResponse();


  } catch (e) {
    // throw (e)
    console.log(e)

    return NextResponse.json({ error: 'An error occurred' }, { status: 500 })
  }
}