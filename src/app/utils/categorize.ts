
import { getModel } from '@/utils/provider';
import { generateText } from 'ai';
import { ScoredPineconeRecord } from "@pinecone-database/pinecone";



type ContentCategory = 'ecampus' | 'sei' | 'conta-institucional' | 'revista' | 'outros';

export async function categorizeQuery(query: string): Promise<ContentCategory> {
  const classifierPrompt = `
    Classifique a pergunta do usuário em uma das categorias:
    - ecampus: perguntas sobre cursos, disciplinas, notas
    - sei: sobre processos administrativos
    - conta-institucional: emails, acesso
    - revista: revista eletronica, publicar revista, criar revista
    - outros: demais assuntos

    Pergunta: "${query}"
  `;

  const model = await getModel()


  // const { content } = await model.invoke([new HumanMessage(classifierPrompt)]);

  const { text } = await generateText({
    model: model,
    system: 'Você é um especialista em categorizar perguntas.',
    prompt: classifierPrompt,
  });
  
  return text?.toLowerCase().trim() as ContentCategory || 'outros';
}








// Função para identificar a categoria da pergunta
export async function identifyCategory(question: string) {
    // Lista de palavras-chave por categoria
    const categories = {
      ecampus: ['ecampus', 'e-campus', 'nota', 'disciplina', 'matrícula', 'professor', 'aluno', 'turma', 'curso'],
      sei: ['sei', 'processo', 'documento', 'protocolo', 'assinatura eletrônica', 'peticionamento'],
      contaInstitucional: ['email', 'e-mail', 'conta', 'senha', 'login', 'acesso', 'redefinir', 'institucional'],
      glpi: ['chamado', 'ticket', 'suporte', 'problema técnico', 'não consigo acessar', 'erro', 'bug'],
      revista: ['revista', 'publicar', 'criar', 'artigo', 'artigos', 'publicação', 'publicações'],
    };
    
    question = question.toLowerCase();
    
    // Verificar em qual categoria a pergunta se encaixa melhor
    let bestMatch = { category: 'geral', count: 0 };
    
    for (const [category, keywords] of Object.entries(categories)) {
      const matches = keywords.filter(keyword => question.includes(keyword.toLowerCase()));
      if (matches.length > bestMatch.count) {
        bestMatch = { category, count: matches.length };
      }
    }
    
    return bestMatch.category;
  }




