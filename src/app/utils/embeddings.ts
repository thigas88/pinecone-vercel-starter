import { getModelEmbedding } from "./provider";


export async function getEmbeddings(input: string) {

  const model = await getModelEmbedding();

  try {
    // console.log('fazendo embeddings: ' + input)
    const embedding = await model.embedDocuments([input.replace(/\n/g, ' ')])
    return embedding[0] as number[]

  } catch (e) {
    console.log("Error calling NomicEmbeddings embedding : ", e);
    throw new Error(`Error calling NomicEmbeddings embedding : ${e}`);
  }
}

