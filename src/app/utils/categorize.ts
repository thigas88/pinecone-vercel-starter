
import { getModel } from '@/app/utils/provider';
import { generateText } from 'ai';

type ContentCategory = 'ecampus' | 'dados-abertos' | 'relatorios-gerenciais' | 'sei' | 'assinador' | 'conta-institucional' | 'revista-eletronica' | 'pagamento-digital' | 'glpi' | 'eduroam' | 'outros';

export async function categorizeQuery(query: string): Promise<ContentCategory> {
  const classifierPrompt = `
    Classifique a pergunta do usuário em uma das categorias e responda apenas com a identificação da categoria.

    As categorias são:
    - ecampus: plano de oferta de disciplina, disciplinas, exibir notas, sistema acadêmico, ecampus
    - glpi: perguntas sobre chamados, tickets, suporte técnico
    - sei: sobre processos administrativos, ofícios, documentos adminsitrativos, sei
    - conta-institucional: emails, acesso, criar conta instituciona, alterar senha da conta institucional
    - revista-eletronica: revista eletronica, publicar revista, criar revista
    - assinador: assinatura digital, assinatura eletronica, assinar documento, validar assinatura
    - eduroam: wifi, internet wifi, acesso wifi, acesso a internet
    - dados-abertos: dados abertos, dados públicos, dados abertos da universidade
    - pagamento-digital: Pag@UFVJM, pagamento digital, pagamento de boleto, GRU, pix
    - relatorios-gerenciais: relatórios gerenciais, metabase, 
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
      contaInstitucional: ['email', 'e-mail', 'conta', 'senha', 'login', 'acesso', 'redefinir', 'institucional', 'alterar senha'],
      glpi: ['chamado', 'ticket', 'suporte', 'problema técnico', 'não consigo acessar', 'erro', 'bug'],
      revista: ['revista', 'publicar artigo', 'criar artigo', 'artigo', 'artigos', 'publicação', 'publicações', 'portal revistas'],
    };
    
    question = question.toLowerCase();
    
    // Verificar em qual categoria a pergunta se encaixa melhor
    let bestMatch = { category: 'outros', count: 0 };
    
    for (const [category, keywords] of Object.entries(categories)) {
      const matches = keywords.filter(keyword => question.includes(keyword.toLowerCase()));
      if (matches.length > bestMatch.count) {
        bestMatch = { category, count: matches.length };
      }
    }
    
    return bestMatch.category;
  }




