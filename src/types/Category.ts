// create a type for Category with options: 'ecampus' | 'sei' | 'containstitucional' | 'revista' | 'glpi' | 'outros' | 'assinador' | 'eduroam' | 'dadosabertos' | 'pagamentodigital' | 'relatoriosgerenciais';
export type Category = 
  | 'ecampus'
  | 'sei'
  | 'containstitucional'
  | 'revista'
  | 'glpi'
  | 'outros'
  | 'assinador'
  | 'eduroam'
  | 'dadosabertos'
  | 'pagamentodigital'
  | 'relatoriosgerenciais';

// create a list of categories
export const categories: Category[] = [
  'ecampus',
  'sei',
  'containstitucional',
  'revista',
  'glpi',
  'outros',
  'assinador',
  'eduroam',
  'dadosabertos',
  'pagamentodigital',
  'relatoriosgerenciais'
];  