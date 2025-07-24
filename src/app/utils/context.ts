import { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { getMatchesFromEmbeddings } from "./pinecone";
import { getEmbeddings } from './embeddings'

export type Metadata = {
  url: string,
  category: string,
  document_id: number,
  title: string,
  tags: [],
  chunk_hash: string,
  chunk_index: number,
  url_ref: string
}

/**
 * Define a estrutura de um único resultado da busca.
 */
interface SearchResult {
  content: string;
  // metadata: Record<string, any>; // Objeto com metadados variados
  metadata: Metadata;
  score: number;
}

/**
 * Define a estrutura completa da resposta da API.
 */
interface ApiResponse {
  query: string;
  results: SearchResult[];
  count: number;
}

/**
 * Define os parâmetros que a nossa função de busca pode receber.
 */
interface SearchParams {
  query: string;
  k?: number; // O '?' torna o parâmetro opcional
  score_threshold?: number;
}
/**
 * Formata os resultados da busca
 */
// function formatResults(results: ScoredPineconeRecord<Metadata>[]) {
//   return results.map(result => {
//     const { metadata } = result;
//     let formattedContent = metadata?.chunk;
    
//     // Adicionar URL de referência se disponível
//     if (metadata?.url) {
//       formattedContent += `\n\n[Referência: ${metadata.url}]\n`;
//     }
    
//     return formattedContent;
//   }).join("\n\n");
// }

// // The function `getContext` is used to retrieve the context of a given message
// export const getContext = async (message: string, category: string, namespace: string = '', maxTokens = 5000, 
//   minScore = 0.2, getOnlyText = true, topK = 10): Promise<string | ScoredPineconeRecord[]> => {

//   try {
//     // Get the embeddings of the input message
//     const embedding = await getEmbeddings(message);

//     // Retrieve the matches for the embeddings from the specified namespace
//     const matches = await getMatchesFromEmbeddings(embedding, topK, namespace, category);

//     // Filter out the matches that have a score lower than the minimum score
//     const qualifyingDocs = matches.filter(m => m.score && m.score > minScore);
//     console.log("Encontrado " + matches.length + " matches")

//     if (!getOnlyText) {
//       // Use a map to deduplicate matches by URL
//       return qualifyingDocs
//     }

//     let docs = matches ? qualifyingDocs.map(match => {
//       let txt = (match.metadata as Metadata).chunk
//       if ((match.metadata as Metadata).url) {
//         txt += `\n\n[Referência: ${(match.metadata as Metadata).url}]\n`;
//       }
//       return txt;
//     }) : [];

//     console.log('Total de chunks mais relevantes com score maior que '+ minScore +': ' + matches.length)
//     qualifyingDocs.map((match) => {
//       console.log( `CHUNCK ID: ${match.id} SCORE: ${match.score}`);
//     });

//     //return formatResults(qualifyingDocs).substring(0, maxTokens);

//     // Join all the chunks of text together, truncate to the maximum number of tokens, and return the result
//     return docs.join("\n").substring(0, maxTokens);
    
//   } catch (error) {
//     console.error("Failed to get context:", error);
//     throw error;
//   }
// }



// The function `getContext` is used to retrieve the context of a given message
export const getContext = async (message: string, category: string = '', namespace: string = '', maxTokens = 5000, 
  minScore = 0.2, getOnlyText = true, topK = 10): Promise<string> => {

    // Define a URL base do seu endpoint
    const baseUrl = 'http://localhost:8000/search';

    // Usa URLSearchParams para construir a query string de forma segura,
    // lidando automaticamente com a codificação de caracteres especiais.
    const urlParams = new URLSearchParams();

    urlParams.append('query', message);
    //urlParams.append('category', category.toString());
    urlParams.append('k', topK.toString());    
    urlParams.append('score_threshold', minScore.toString());

    // Constrói a URL final
    const fullUrl = `${baseUrl}?${urlParams.toString()}`;
    console.log(`Fazendo requisição para: ${fullUrl}`);

  try {
    const response = await fetch(fullUrl, {
      method: 'GET', // Método da requisição
      headers: {
        // Define o tipo de conteúdo que esperamos receber
        'Accept': 'application/json',
      },
    })

    // Verifica se a resposta da rede foi bem-sucedida (status 200-299)
    if (!response.ok) {
      // Se não foi, lança um erro com o status da resposta
      throw new Error(`Erro na requisição: ${response.status} ${response.statusText}`);
    }

    // Converte a resposta do formato JSON para um objeto JavaScript.
    // A tipagem <ApiResponse> garante que o TypeScript saiba qual a estrutura do objeto.
    const data: ApiResponse = await response.json();

    let docs = data.results ? data.results.map(match => {
      let txt = (match as SearchResult).content
      if ((match as SearchResult).metadata.url_ref) {
        txt += `\n\n[Referência: ${(match as SearchResult).metadata.url_ref}]\n`;
        txt += '----------------------------------------------------------------';
      }
      return txt;
    }) : [];

    console.log('Total de chunks mais relevantes com score maior ou igual que '+ minScore +': ' + data.count)    
    data.results.forEach((result, index) => {
      console.log(`\nResultado ${index + 1}:`);
      console.log(`  Score: ${result.score}`);
      console.log(`  Conteúdo: ${result.content.substring(0, 100)}...`); // Mostra os primeiros 100 caracteres
      console.log(`  Metadados:`, result.metadata);
    });

    // Join all the chunks of text together, truncate to the maximum number of tokens, and return the result
    return docs.join("\n\n").substring(0, maxTokens);
    
  } catch (error) {
    console.error("Failed to get context:", error);
    throw error;
  }
}
