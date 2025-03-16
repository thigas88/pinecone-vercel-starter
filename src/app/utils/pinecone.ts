import { Pinecone, type ScoredPineconeRecord } from "@pinecone-database/pinecone";

// Obtain a client for Pinecone
const pinecone = new Pinecone();

export type Metadata = {
  url: string,
  text: string,
  chunk: string,
  hash: string
}

// The function `getMatchesFromEmbeddings` is used to retrieve matches for the given embeddings
const getMatchesFromEmbeddings = async (embeddings: number[], topK: number, namespace: string): Promise<ScoredPineconeRecord<Metadata>[]> => {
  // Obtain a client for Pinecone
  // const pinecone = new Pinecone();

  const indexName: string = process.env.PINECONE_INDEX || '';
  if (indexName === '') {
    throw new Error('PINECONE_INDEX environment variable not set')
  }

  // Retrieve the list of indexes to check if expected index exists
  const indexes = (await pinecone.listIndexes())?.indexes;
  if (!indexes || indexes.filter(i => i.name === indexName).length !== 1) {
    throw new Error(`Index ${indexName} does not exist`)
  }

  // Get the Pinecone index
  const index = pinecone!.Index<Metadata>(indexName);

  // Get the namespace
  const pineconeNamespace = index.namespace(namespace ?? '')

  try {
    // Query the index with the defined request
    const queryResult = await pineconeNamespace.query({
      vector: embeddings,
      topK,
      includeMetadata: true,
    })
    return queryResult.matches || []
  } catch (e) {
    // Log the error and throw it
    console.log("Error querying embeddings: ", e)
    throw new Error(`Error querying embeddings: ${e}`)
  }
}

export { getMatchesFromEmbeddings }


export async function resetIndex(indexName: string) {
  await deleteIndex(indexName);
  await createIndexIfNecessary(indexName);
}

async function deleteIndex(indexName: string) {
  await pinecone.deleteIndex(indexName);
}

export async function createIndexIfNecessary(indexName: string, dimension: number = 1024) {
  await pinecone.createIndex({
    name: indexName,
    dimension: dimension,
    spec: {
      serverless: {
        cloud: 'aws',
        region: 'us-east-1',
      }
    },
    waitUntilReady: true,
    suppressConflicts: true
  });
}

export async function pineconeIndexHasVectors(indexName: string): Promise<boolean> {
  try {
    const targetIndex = pinecone.index(indexName)

    const stats = await targetIndex.describeIndexStats();

    return (stats.totalRecordCount && stats.totalRecordCount > 0) ? true : false;
  } catch (error) {
    console.error('Error checking Pinecone index:', error);
    return false;
  }
}

export async function pineconeIndexExists(indexName: string): Promise<boolean> {
  try {
    const { indexes } = await pinecone.listIndexes();

    // Check if index already exists
    const indexNames = (indexes && indexes.length ? indexes.map(index => index.name) : []);

    if (!indexNames.includes(indexName)) {
      return false;
    }

    return true

  } catch (error) {
    console.error('Error checking Pinecone index:', error);
    return false;
  }
}