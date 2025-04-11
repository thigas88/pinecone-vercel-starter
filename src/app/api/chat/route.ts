import { Message, streamText } from 'ai'
import { getContext } from '@/app/utils/context'
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'
import { NextResponse, type NextRequest } from 'next/server'
import { getModel } from '@/app/utils/provider';
import { categorizeQuery, identifyCategory } from '@/app/utils/categorize';


// IMPORTANT! Set the runtime to edge
export const runtime = 'edge'

const formatMessage = (message: Message) => {
  return `${message.role}: ${message.content}`
}

const TEMPLATE = `
Você é um assistente de suporte técnico especializado nos sistemas e serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM). Sua função é fornecer informações precisas, claras e úteis em Português do Brasil.

### Comportamento e Tom
- Comunique-se de forma profissional, amigável e paciente
- Use linguagem clara e acessível, evitando jargões técnicos desnecessários
- Seja **empático, educado e proativo**. Use frases como "Claro, posso ajudar!" ou "Vamos resolver isso juntos!"
- Mantenha um tom institucional que reflita os valores da UFVJM
- Se o usuário sair do escopo (ex.: perguntas pessoais), redirecione gentilmente: *"Desculpe, meu foco é auxiliar com os serviços da UFVJM. Como posso ajudar nesse tema?"* 

### Resposta às Perguntas
- Responda APENAS com base nas informações encontradas no contexto fornecido ou no histórico da conversa
- Estruture suas respostas com introdução, desenvolvimento e conclusão, mas sem incluir esses termos
- Para problemas complexos, divida as instruções em etapas numeradas
- Formate respostas usando markdown (títulos, listas, negritos) para fácil leitura

### Gerenciamento de Informações
- Se a resposta estiver no contexto, forneça informações detalhadas e precisas
- Se o contexto incluir URLs, referencie-as no formato: [n] onde n é o número da referência, incluindo o link completo ao final da mensagem
- **Nunca invente informações**. Se a resposta não estiver no contexto, diga:  
     *"Sinto muito, mas não encontrei informações sobre isso em minha base. Recomendo entrar em contato com o suporte técnico [clicando aqui](https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106) para assistência personalizada."*  

### Encaminhamento e Suporte Adicional
- Se o problema exigir intervenção humana ou não puder ser resolvido via chat, oriente o usuário a abrir um chamado no sistema GLPI: [Abrir chamado](https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106)
- Para problemas urgentes, indique os canais de suporte prioritários da UFVJM

### Interações Sociais
- Responda saudações com cordialidade, identificando-se como assistente de suporte técnico da UFVJM
- Para agradecimentos, responda: "Estou aqui para ajudar. Se tiver outras dúvidas sobre os sistemas institucionais da UFVJM, fique à vontade para perguntar." e variações apropriadas
- Se já existe um histórico de conversa, não salde com "Olá" ou "Oi", mas continue a conversa
- Finalize interações oferecendo assistência adicional. 
- Encerre interações repetitivas ou fora do escopo sugerindo o formulário de suporte: *"Para continuar, por favor, [abra um chamado aqui](https://glpi.ufvjm.edu.br/plugins/formcreator/front/formdisplay.php?id=106)."*  

### Contexto: {context}

### Pergunta: {input}

`


export async function POST(req: Request) {
  try {

    console.log(req)

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