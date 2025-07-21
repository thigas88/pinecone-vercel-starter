
import { getModel } from '@/app/utils/provider';
import { generateText } from 'ai';
import { Category as ContentCategory } from '@/types/Category';

// type ContentCategory = 'ecampus' | 'sei' | 'containstitucional' | 'revista' | 'glpi' | 'outros' | 'assinador' | 'eduroam' | 'dadosabertos' | 'pagamentodigital' | 'relatoriosgerenciais';

/**
 * Classifica uma pergunta do usuário em uma das categorias predefinidas utilizando um modelo de IA.
 * @param query A pergunta do usuário a ser classificada.
 * @returns A categoria da pergunta.
 */
export async function categorizeQuery(query: string): Promise<ContentCategory> {
  const classifierPrompt = `
    Classifique a pergunta do usuário em uma das categorias e responda apenas com a identificação da categoria.

    As categorias são:
    - ecampus: plano de oferta de disciplina, disciplinas, exibir notas, sistema acadêmico, ecampus
    - glpi: perguntas sobre chamados, tickets, suporte técnico
    - sei: sobre processos administrativos, ofícios, documentos adminsitrativos, sei, modelos de documentos, peticionamento
    - containstitucional: email institucional, acesso a conta institucional, criar conta instituciona, alterar senha da conta institucional
    - revista: revista eletronica, publicar revista, criar revista
    - assinador: assinatura digital, assinatura eletronica, assinar documento, validar assinatura
    - eduroam: wifi, internet wifi, acesso wifi, acesso a internet
    - dadosabertos: dados abertos, dados públicos, dados abertos da universidade
    - pagamentodigital: Pag@UFVJM, pagamento digital, pagamento de boleto, GRU, pix
    - relatoriosgerenciais: gerar relatório, criar relatório, relatórios gerenciais, metabase, 
    - outros: demais assuntos

Responda apenas com o texto da categoria específica, sem repetir o prompt ou outras explicações. Por exemplo: *ecampus*

    Pergunta: "${query}"
  `;

  const model = await getModel()

  const { text } = await generateText({
    model: model,
    system: 'Você é um especialista em categorizar perguntas sobre os sistemas e serviços da UFVJM.',
    prompt: classifierPrompt,
  });
  
  return text?.toLowerCase().trim() as ContentCategory || 'outros';
}




/**
 * Identifica a categoria de uma pergunta com base em palavras-chave.
 * @param question A pergunta do usuário.
 * @returns A categoria identificada.
 */
export async function identifyCategory(question: string) {
    // Lista de palavras-chave por categoria
    const categories = {
      ecampus: ['ecampus', 'e-campus', 'nota', 'disciplina', 'matrícula', 'professor', 'aluno', 'turma', 'curso'],
      sei: ['sei', 'processo', 'documento', 'protocolo', 'assinatura eletrônica', 'peticionamento'],
      containstitucional: ['email', 'e-mail', 'conta', 'senha', 'login', 'acesso', 'redefinir', 'institucional', 'alterar senha'],
      glpi: ['chamado', 'ticket', 'suporte', 'problema técnico', 'não consigo acessar', 'erro', 'bug'],
      revista: ['revista', 'publicar artigo', 'criar artigo', 'artigo', 'artigos', 'publicação', 'publicações', 'portal revistas'],
      assinador: ['assinatura digital', 'assinatura eletrônica', 'assinar documento', 'validar assinatura', 'assinador'],
      eduroam: ['eduroam', 'wifi', 'internet', 'rede wifi', 'acesso wifi', 'conectar wifi', 'rede eduroam'],
      dadosabertos: ['dados abertos', 'dados públicos', 'dados abertos da universidade', 'dados abertos ufvjm', 'dados abertos universidade'],
      pagamentodigital: ['pagamento digital', 'pag@ufvjm', 'boleto', 'GRU', 'pix', 'pagamento', 'taxa', 'cobrança'],
      relatoriosgerenciais: ['relatório', 'metabase', 'gerar relatório', 'criar relatório', 'relatórios gerenciais', 'dados gerenciais', 'painel gerencial'],
      outros: ['outros', 'geral', 'diversos', 'não entendo', 'não sei responder', 'não sei', 'não entendi', 'não compreendo', 'não sei o que é isso'],
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




