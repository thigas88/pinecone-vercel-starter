import OpenAI from 'openai';
import { Message, OpenAIStream, StreamingTextResponse } from 'ai'
import { getContext } from '@/utils/context'

// Create an OpenAI API client (that's edge friendly!)
const openai = new OpenAI({
  apiKey: "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", // process.env.OPENAI_API_KEY,
  baseURL: "https://api-inference.huggingface.co/v1/" // process.env.OPENAI_CUSTOM_BASE_URL
})

// IMPORTANT! Set the runtime to edge
export const runtime = 'edge'

export async function POST(req: Request) {
  try {

    const { messages } = await req.json()

    // Get the last message
    const lastMessage = messages[messages.length - 1]

    // Get the context from the last message
    const context = await getContext(lastMessage.content, '')


    const prompt = [
      {
        role: 'system',
        content: `Você é assistente de IA é uma inteligência artificial, poderoso e semelhante a um humano.
      Você tem conhecimento especializado, utilidade, inteligência e bem articulado sobre a Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM).
      Você é um indivíduo bem-comportado e bem-educado.
      Seja sempre amigável, gentil e inspirador, e está ansioso para fornecer respostas vívidas e atenciosas ao usuário.
      Todo o conhecimento em seu cérebro e é capaz de responder com precisão a quase qualquer pergunta sobre qualquer tópico em uma conversa.
      START CONTEXT BLOCK
      ${context}
      END OF CONTEXT BLOCK
      Você devará considerar qualquer BLOCO DE CONTEXTO (CONTEXT BLOCK) fornecido em uma conversa.
      Se o contexto não fornecer a resposta à pergunta, você dirá: "Sinto muito, mas não sei a resposta para essa pergunta".
      Você não se desculpará por respostas anteriores, mas indicará que novas informações foram obtidas.
      Não invente nada que não seja extraído diretamente do contexto.`,
      },
    ]

    let out = "";

    // Ask OpenAI for a streaming chat completion given the prompt
    const response  = await openai.chat.completions.create({
      model: 'Qwen/Qwen2.5-72B-Instruct',  // || 'gpt-3.5-turbo'
      stream: true,
      max_tokens: 2048,
      messages: [...prompt, ...messages.filter((message: Message) => message.role === 'user')]
    })


    // Convert the response into a friendly text-stream
    const stream = OpenAIStream(response)
    // Respond with the stream
    return new StreamingTextResponse(stream)

    // for await (const chunk of response ) {
    //   if (chunk.choices && chunk.choices.length > 0) {
    //     const newContent = chunk.choices[0].delta.content;
    //     out += newContent;
    //     console.log(newContent);
    //   }  
    // }

    // return out;

  } catch (e) {
    throw (e)
  }
}