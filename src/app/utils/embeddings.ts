
import OpenAI from 'openai';

const config = {
  apiKey: process.env.OPENAI_EMBEDDINGS_API_KEY || process.env.OPENAI_API_KEY,
  baseUrl: process.env.OPENAI_EMBEDDINGS_BASE_URL || 'https://api.openai.com/v1'
}

const openai = new OpenAI(config)

export async function getEmbeddings(input: string) {
  try {
    console.log('fazendo embeddings: ' + input)
    const embedding  = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: [input.replace(/\n/g, ' ')],
      encoding_format: "float",
    })
    return embedding.data[0].embedding as number[]

  } catch (e) {
    console.log("Error calling OpenAI embedding API: ", e);
    throw new Error(`Error calling OpenAI embedding API: ${e}`);
  }
}