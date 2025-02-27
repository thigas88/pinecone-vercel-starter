
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

import { CohereEmbeddings } from "@langchain/cohere";

const embeddings = new CohereEmbeddings({
  model: "embed-multilingual-v3.0",
});


export async function getEmbeddings(input: string) {
  try {
    console.log('fazendo embeddings: ' + input)
    const embedding  = await embeddings.embedDocuments([input.replace(/\n/g, ' ')]);
    return embedding[0] as number[]

  } catch (e) {
    console.log("Error calling OpenAI embedding API: ", e);
    throw new Error(`Error calling OpenAI embedding API: ${e}`);
  }
}



/// HuggingFaceTransformersEmbeddings

// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";


// export async function getEmbeddings(input: string) {

//   const { HuggingFaceTransformersEmbeddings } = await import("@langchain/community/embeddings/huggingface_transformers");

//   const model = new HuggingFaceTransformersEmbeddings({
//     model: "sentence-transformers/all-MiniLM-L6-v2",
//   });

//   try {
//     //console.log('fazendo embeddings: ' + input)
//     const embedding = await model.embedDocuments([input.replace(/\n/g, ' ')])
//     return embedding

//   } catch (e) {
//     console.log("Error calling HuggingFaceTransformersEmbeddings embedding : ", e);
//     throw new Error(`Error calling HuggingFaceTransformersEmbeddings embedding : ${e}`);
//   }
// }


// /* Embed queries */
// const res = await model.embedQuery(
//   "What would be a good company name for a company that makes colorful socks?"
// );
// console.log({ res });
// /* Embed documents */
// const documentRes = await model.embedDocuments(["Hello world", "Bye bye"]);
// console.log({ documentRes });