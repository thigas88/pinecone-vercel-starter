import { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { getMatchesFromEmbeddings } from "./pinecone";
import { getEmbeddings } from './embeddings'

export type Metadata = {
  url: string,
  text: string,
  chunk: string,
}


/**
 * Formata os resultados da busca
 */
function formatResults(results: ScoredPineconeRecord<Metadata>[]) {
  return results.map(result => {
    const { metadata } = result;
    let formattedContent = metadata?.chunk;
    
    // Adicionar URL de referência se disponível
    if (metadata?.url) {
      formattedContent += `\n\n[Referência: ${metadata.url}]\n`;
    }
    
    return formattedContent;
  }).join("\n\n");
}

// The function `getContext` is used to retrieve the context of a given message
export const getContext = async (message: string, category: string, namespace: string = '', maxTokens = 5000, 
  minScore = 0.6, getOnlyText = true, topK = 8): Promise<string | ScoredPineconeRecord[]> => {

  try {
    // Get the embeddings of the input message
    const embedding = await getEmbeddings(message);

    // Retrieve the matches for the embeddings from the specified namespace
    const matches = await getMatchesFromEmbeddings(embedding, topK, namespace);

    // Filter out the matches that have a score lower than the minimum score
    const qualifyingDocs = matches.filter(m => m.score && m.score > minScore);

    if (!getOnlyText) {
      // Use a map to deduplicate matches by URL
      return qualifyingDocs
    }

    let docs = matches ? qualifyingDocs.map(match => {
      let txt = (match.metadata as Metadata).chunk
      if ((match.metadata as Metadata).url) {
        txt += `\n\n[Referência: ${(match.metadata as Metadata).url}]\n`;
      }
      return txt;
    }) : [];

    console.log('chunks mais relevantes com score maior que '+ minScore +': ')
    qualifyingDocs.map((match) => {
      console.log( `CHUNCK ID: ${match.id} SCORE: ${match.score}`);
    });

    //return formatResults(qualifyingDocs).substring(0, maxTokens);

    // Join all the chunks of text together, truncate to the maximum number of tokens, and return the result
    return docs.join("\n").substring(0, maxTokens);
    
  } catch (error) {
    console.error("Failed to get context:", error);
    throw error;
  }
}
