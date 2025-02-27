import { HfInference } from "@huggingface/inference";

const HF_TOKEN = "hf_...";

const inference = new HfInference(HF_TOKEN);

//         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

// nomic-ai/nomic-embed-text-v1


// // Chat completion API
// const out = await inference.chatCompletion({
//   model: "meta-llama/Llama-3.1-8B-Instruct",
//   messages: [{ role: "user", content: "Hello, nice to meet you!" }],
//   max_tokens: 512
// });
// console.log(out.choices[0].message);

// // Streaming chat completion API
// for await (const chunk of inference.chatCompletionStream({
//   model: "meta-llama/Llama-3.1-8B-Instruct",
//   messages: [{ role: "user", content: "Hello, nice to meet you!" }],
//   max_tokens: 512
// })) {
//   console.log(chunk.choices[0].delta.content);
// }

// /// Using a third-party provider:
// await inference.chatCompletion({
//   model: "meta-llama/Llama-3.1-8B-Instruct",
//   messages: [{ role: "user", content: "Hello, nice to meet you!" }],
//   max_tokens: 512,
//   provider: "sambanova", // or together, fal-ai, replicate, cohere …
// })

// await inference.textToImage({
//   model: "black-forest-labs/FLUX.1-dev",
//   inputs: "a picture of a green bird",
//   provider: "fal-ai",
// })



// // You can also omit "model" to use the recommended model for the task
// await inference.translation({
//   inputs: "My name is Wolfgang and I live in Amsterdam",
//   parameters: {
//     src_lang: "en",
//     tgt_lang: "fr",
//   },
// });

// // pass multimodal files or URLs as inputs
// await inference.imageToText({
//   model: 'nlpconnect/vit-gpt2-image-captioning',
//   data: await (await fetch('https://picsum.photos/300/300')).blob(),
// })

// // Using your own dedicated inference endpoint: https://hf.co/docs/inference-endpoints/
// const gpt2 = inference.endpoint('https://xyz.eu-west-1.aws.endpoints.huggingface.cloud/gpt2');
// const { generated_text } = await gpt2.textGeneration({inputs: 'The answer to the universe is'});

// // Chat Completion
// const llamaEndpoint = inference.endpoint(
//   "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.1-8B-Instruct"
// );
// const out = await llamaEndpoint.chatCompletion({
//   model: "meta-llama/Llama-3.1-8B-Instruct",
//   messages: [{ role: "user", content: "Hello, nice to meet you!" }],
//   max_tokens: 512,
// });
// console.log(out.choices[0].message);