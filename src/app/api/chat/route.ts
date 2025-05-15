import { Message, streamText, generateText, CoreMessage } from 'ai'
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

let contextHistory = '';

const TEMPLATE_DECISAO = `
Você é um assistente de suporte técnico especializado nos sistemas e serviços da Universidade Federal dos Vales do Jequinhonha e Mucuri (UFVJM). Sua função é fornecer informações precisas, claras e úteis em Português do Brasil.
Primeiro, analise a pergunta do usuário cuidadosamente para determinar se ela requer informações adicionais para ser respondida de forma completa e precisa.

### Comportamento e Tom:
- Comunique-se de forma profissional, amigável e paciente
- Use linguagem clara e acessível, evitando jargões técnicos desnecessários
- Seja **empático, educado e proativo**. Use frases como "Claro, posso ajudar!" ou "Vamos resolver isso juntos!"
- Mantenha um tom institucional que reflita os valores da UFVJM
- Se o usuário sair do escopo (ex.: perguntas pessoais), redirecione gentilmente: *"Desculpe, meu foco é auxiliar com os serviços da UFVJM. Como posso ajudar nesse tema?"* 

### Instruções de Ação:
- Se a análise indicar que a pergunta pode ser respondida adequadamente SEM contexto adicional, forneça a resposta diretamente com base no seu conhecimento geral.
- Se a análise indicar que a pergunta NECESSITA de contexto adicional para uma resposta completa e precisa, responda apenas com o texto NECESSITA_CONTEXTO. **Neste caso, você não fornecerá uma resposta final ainda.** 

### Exemplo (para o LLM entender o comportamento desejado):

*   **Pergunta do Usuário:** "Qual é a capital da França?"
    **Ação:** A pergunta pode ser respondida sem contexto adicional.
    **Resposta Final:** A capital da França é Paris.

*   **Pergunta do Usuário:** "Quem atua como controlador dos dados pessoais no serviço Assina@UFVJM?"
    **Ação:** A pergunta NECESSITA de contexto adicional sobre a política de privacidade ou termos de uso do serviço Assina@UFVJM.
    **Resposta Final:** NECESSITA_CONTEXTO.


`

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

    console.log('Mensagem atual: ', currentMessageContent)

    // Classificar a pergunta
    const category = await categorizeQuery(currentMessageContent);
    console.log('Categoria identificada:', category);

    // // Identificar a categoria da pergunta
    // const category2 = await identifyCategory(currentMessageContent);
    // console.log('Categoria identificada 2:', category2);

    const model = await getModel()



    // const promptDecisao = ChatPromptTemplate.fromTemplate(TEMPLATE_DECISAO)
    // const formattedChatDecisao = await promptDecisao.invoke({
    //   input: currentMessageContent,
    //   context: contextHistory
    // });


    let mensagens: CoreMessage[] = [
      { role: 'system', 
        content: TEMPLATE_DECISAO 
      },
      { role: 'user', 
        content: currentMessageContent 
      }
    ]
    
    // Aqui o LLM tomará a decisão se precisa ou não de contexto.
    const { text } = await generateText({
      model: model,
      system: TEMPLATE_DECISAO,
      // prompt: formattedChatDecisao.toString(),
      messages: mensagens,
    });

    const resultDecisao = text;

    // return NextResponse.json({ resultDecisao }, { status: 200 })

    let finalResult;

    // Verifica se o LLM decidiu que precisa de contexto
    if (resultDecisao && resultDecisao.includes("NECESSITA_CONTEXTO")) {
        console.log("Contexto necessário. Buscando contexto...");
        // Buscar contexto específico
        const context = await getContext(currentMessageContent, category, '');

        // Histórico completo de mensagens
        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage)
        contextHistory = formattedPreviousMessages.join('\n');

        // // Histórico recente à conversa para manter o contexto
        // const buildChatHistory = (messages: Message[]) =>
        //     messages
        //         .slice(-4) // Mantém últimas 4 interações
        //         .map(m => `${m.role.toUpperCase()}: ${m.content}`)
        //         .join('\n');

        // // Monta o histórico de mensagens
        // const chat_history = buildChatHistory(messages);

        const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE)

        const formattedChatPrompt = await prompt.invoke({
            context: context,
            chat_history: formattedPreviousMessages.join('\n'),
            input: currentMessageContent,
        });

        // Chama o LLM novamente, agora com o contexto incluído
        finalResult = streamText({
            model: model,
            prompt: formattedChatPrompt.toString()
        });

        console.log('resultado com contexto: ', finalResult);

    }
     else {
        // Se o LLM não indicou a necessidade de contexto, a resposta dele é a resposta final
        console.log("Contexto não necessário. Usando a resposta do LLM diretamente.");
        // Cria um stream a partir da resposta do LLM para manter o formato de retorno
        
        const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE_DECISAO)

        // const formattedChatPrompt = await prompt.invoke({
        //     input: currentMessageContent,
        // });

        // Chama o LLM novamente, agora com o contexto incluído
        finalResult = streamText({
            model: model,
            //prompt: formattedChatPrompt.toString()
            messages: mensagens,
        });

    }

    return finalResult.toDataStreamResponse();




















    // // Buscar contexto específico
    // const context = await getContext(currentMessageContent, category, '');

    // const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage)

    // // Histórico recente à conversa para manter o contexto
    // const buildChatHistory = (messages: Message[]) => 
    //   messages
    //     .slice(-4) // Mantém últimas 4 interações
    //     .map(m => `${m.role.toUpperCase()}: ${m.content}`)
    //     .join('\n');
    
    // // Monta o histórico de mensagens
    // const chat_history = buildChatHistory(messages);
    
    // const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE)

    // const formattedChatPrompt = await prompt.invoke({
    //   context: context,
    //   chat_history: formattedPreviousMessages.join('\n'),
    //   input: currentMessageContent,
    // });

    // // console.log('Histórico:', chat_history);
    // // console.log('Última mensagem:', lastMessage);
    // // console.log('Categoria:', category);
    // // console.log('Contexto:', context);


    // const result = streamText({
    //   model: model,
    //   prompt: formattedChatPrompt.toString()
    // });

    // console.log('resultado: ', result)

    // return result.toDataStreamResponse();

  } catch (e) {
    // throw (e)
    console.log(e)
    return NextResponse.json({ error: 'An error occurred: ' + e }, { status: 500 })
  }
}