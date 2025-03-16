
// import OpenAI from 'openai';

// const config = {
//   apiKey: process.env.OPENAI_EMBEDDINGS_API_KEY || process.env.OPENAI_API_KEY,
//   baseUrl: process.env.OPENAI_EMBEDDINGS_BASE_URL || 'https://api.openai.com/v1'
// }

// const openai = new OpenAI(config)

// export async function getEmbeddings(input: string) {
//   try {
//     console.log('fazendo embeddings: ' + input)
//     const embedding  = await openai.embeddings.create({
//       model: "text-embedding-3-small",
//       input: [input.replace(/\n/g, ' ')],
//       encoding_format: "float",
//     })
//     return embedding.data[0].embedding as number[]

//   } catch (e) {
//     console.log("Error calling OpenAI embedding API: ", e);
//     throw new Error(`Error calling OpenAI embedding API: ${e}`);
//   }
// }



// CohereEmbeddings

// import { CohereEmbeddings } from "@langchain/cohere";

// const embeddings = new CohereEmbeddings({
//   model: "embed-multilingual-v3.0",
// });


// export async function getEmbeddings(input: string) {
//   try {
//     console.log('fazendo embeddings: ' + input)
//     const embedding  = await embeddings.embedDocuments([input.replace(/\n/g, ' ')]);
//     return embedding[0] as number[]

//   } catch (e) {
//     console.log("Error calling OpenAI embedding API: ", e);
//     throw new Error(`Error calling OpenAI embedding API: ${e}`);
//   }
// }



import { NomicEmbeddings } from "@langchain/nomic";


export async function getEmbeddings(input: string) {

  const model = new NomicEmbeddings({
    apiKey: process.env.NOMIC_API_KEY,
    modelName: process.env.NOMIC_EMBEDDINGS_NAME,
  });

  try {
    // console.log('fazendo embeddings: ' + input)
    const embedding = await model.embedDocuments([input.replace(/\n/g, ' ')])
    return embedding[0] as number[]

  } catch (e) {
    console.log("Error calling NomicEmbeddings embedding : ", e);
    throw new Error(`Error calling NomicEmbeddings embedding : ${e}`);
  }
}

