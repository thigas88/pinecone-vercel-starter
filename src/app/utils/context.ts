import { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { getMatchesFromEmbeddings } from "./pinecone";
import { getEmbeddings } from './embeddings'

export type Metadata = {
  url: string,
  text: string,
  chunk: string,
}

// The function `getContext` is used to retrieve the context of a given message
export const getContext = async (message: string, namespace: string, maxTokens = 5000, 
  minScore = 0.6, getOnlyText = true, topK = 8): Promise<string | ScoredPineconeRecord[]> => {

  try {
    // Get the embeddings of the input message
    const embedding = await getEmbeddings(message);

    // Retrieve the matches for the embeddings from the specified namespace
    const matches = await getMatchesFromEmbeddings(embedding, topK, namespace);

    console.log('chunks encontrados até o limite de '+topK+' : ' + matches)

    // Filter out the matches that have a score lower than the minimum score
    const qualifyingDocs = matches.filter(m => m.score && m.score > minScore);

    if (!getOnlyText) {
      // Use a map to deduplicate matches by URL
      return qualifyingDocs
    }

    let docs = matches ? qualifyingDocs.map(match => (match.metadata as Metadata).chunk) : [];

    console.log('chunks mais relevantes com score maior que '+ minScore +': ')
    qualifyingDocs.map((match) => {
      const metadata = match.metadata as Metadata;
      console.log( `CHUNCK ID: ${match.id} SCORE: ${match.score}`);
    });

    // Join all the chunks of text together, truncate to the maximum number of tokens, and return the result
    return docs.join("\n").substring(0, maxTokens);
    
  } catch (error) {
    console.error("Failed to get context:", error);
    throw error;
  }
}
