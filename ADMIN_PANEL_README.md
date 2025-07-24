# Painel Administrativo ChatRAG

## Visão Geral

O Painel Administrativo do ChatRAG é uma interface web completa para gerenciamento de documentos, processamentos e análise de dados do sistema de Retrieval-Augmented Generation (RAG).

## Funcionalidades

### 1. Dashboard Principal (`/admin`)
- Visão geral do sistema com estatísticas em tempo real
- Total de documentos e chunks indexados
- Tamanho do banco de dados
- Documentos recentes
- Distribuição por categoria

### 2. Gerenciamento de Documentos (`/admin/documents`)
- Listagem completa de todos os documentos
- Filtros por status, categoria e busca por texto
- Adição de novos documentos com conteúdo direto
- **Reindexação de documentos com URL web válida**
  - Preview do conteúdo antes da reindexação
  - Busca automática de conteúdo da web
  - Processamento com diferentes métodos de divisão
- Visualização do histórico de processamento
- Status em tempo real (indexado, processando, erro, agendado)

### 3. Gerenciamento de Processamentos (`/admin/processing`)
- Monitoramento de todos os processamentos em tempo real
- Auto-refresh configurável (5s, 10s, 30s)
- Estatísticas de processamento:
  - Total de processamentos
  - Taxa de sucesso
  - Processamentos ativos
  - Total de chunks processados
- Detalhes de cada processamento:
  - Tempo de execução
  - Mensagens de erro
  - Chunks indexados

### 4. Agendamentos (`/admin/schedules`)
- Visualização de processamentos agendados
- Criação de novos agendamentos
- Execução imediata de agendamentos pendentes
- Timeline organizada por data
- Status de cada agendamento

### 5. Estatísticas (`/admin/statistics`)
- Análise detalhada do sistema
- Distribuição por categoria com gráficos
- Status dos documentos
- Métodos de divisão utilizados
- Tempos médios de processamento
- Informações do sistema (chunks mais antigos/recentes)
- Filtros por período (7, 30, 90 dias ou todo período)

### 6. Busca Avançada (`/admin/search`)
- Busca vetorial semântica nos documentos indexados
- Filtros avançados:
  - Por categoria
  - Por tags
  - Score mínimo de relevância
  - Número de resultados
- Destaque dos termos buscados
- Visualização expandida dos resultados
- Links para documentos originais

## Recursos Técnicos

### Reindexação de Documentos Web
O sistema permite reindexar documentos que possuem URLs web válidas:

1. **Preview de Conteúdo**: Antes de reindexar, é possível visualizar o conteúdo que será extraído da URL
2. **Extração Automática**: O sistema faz o download e extração do conteúdo HTML
3. **Limpeza de Conteúdo**: Remove scripts, estilos e formata o texto
4. **Processamento Inteligente**: Divide o conteúdo em chunks usando o método configurado
5. **Atualização de Vetores**: Remove chunks antigos e cria novos vetores no TimescaleVector

### Métodos de Divisão Suportados
- **Character**: Divisão por número de caracteres
- **Sentence**: Divisão por sentenças completas
- **Semantic**: Divisão baseada em similaridade semântica
- **Markdown**: Preserva estrutura de documentos Markdown

### Integração com Backend
- API RESTful em FastAPI
- Suporte CORS para requisições do frontend
- Processamento assíncrono em background
- Armazenamento vetorial com TimescaleVector
- Embeddings com modelos Hugging Face ou Google

## Instalação e Configuração

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
npm install
npm run dev
```

### Acesso
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Painel Admin: http://localhost:3000/admin

## Endpoints da API

### Documentos
- `GET /admin/documents` - Lista todos os documentos
- `POST /admin/ingest/document` - Adiciona novo documento
- `POST /admin/documents/{id}/reindex` - Reindexa documento da web
- `GET /admin/documents/{id}/fetch-content` - Preview do conteúdo web
- `GET /admin/documents/{id}/history` - Histórico de processamento
- `GET /admin/documents/{id}/chunks` - Lista chunks do documento

### Estatísticas
- `GET /admin/vector-store/stats` - Estatísticas do vector store
- `GET /search` - Busca vetorial com filtros

## Tecnologias Utilizadas

### Frontend
- Next.js 13+ com App Router
- TypeScript
- Tailwind CSS
- Lucide Icons
- React Hooks para estado e efeitos

### Backend
- FastAPI
- SQLAlchemy com PostgreSQL
- TimescaleVector para armazenamento vetorial
- LangChain para processamento de texto
- BeautifulSoup para extração de conteúdo web
- Sentence Transformers para embeddings

## Próximas Melhorias

1. **Autenticação e Autorização**
   - Sistema de login
   - Controle de acesso por roles
   - Audit log de ações

2. **Processamento em Lote**
   - Upload de múltiplos arquivos
   - Importação via CSV
   - Processamento de diretórios

3. **Análises Avançadas**
   - Gráficos interativos
   - Exportação de relatórios
   - Métricas de qualidade dos embeddings

4. **Otimizações**
   - Cache de resultados
   - Paginação server-side
   - Compressão de chunks

## Contribuindo

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT.