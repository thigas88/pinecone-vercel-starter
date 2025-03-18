import {
  Message,
  LangChainAdapter,
  streamText,
  generateText,
  wrapLanguageModel,
  extractReasoningMiddleware,
  LanguageModelV1,
} from "ai";

// import models from ai-sdk
import { groq, createGroq } from "@ai-sdk/groq";
import { google, createGoogleGenerativeAI } from "@ai-sdk/google";
import { cohere } from '@ai-sdk/cohere';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { createOllama } from 'ollama-ai-provider';

// imports for embeddings from langchain
import { NomicEmbeddings } from "@langchain/nomic";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { CohereEmbeddings } from "@langchain/cohere";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { OllamaEmbeddings } from "@langchain/ollama";
/* 
 Models 
*/

const ollamaProvider = async () => {
  // ollama API
  const ollama = createOllama({
    baseURL: 'https://api.ollama.com'
  });

  return ollama(process.env.MODEL_NAME || 'phi4');
};

const oprouterProvider = async () => {
  // OpenRouter API
  const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY
  });

  return openrouter.chat(process.env.MODEL_NAME || 'anthropic/claude-3.5-sonnet');
};

const googleProvider = async () => {
  // Google AI API
  const modelGoogle = createGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
  });

  return modelGoogle(process.env.MODEL_NAME || "gemini-1.5-pro-latest");
};

const groqProvider = async () => {
  // Groq API
  const modelGroq = createGroq({
    baseURL: process.env.OPENAI_CUSTOM_BASE_URL,
    apiKey: process.env.OPENAI_API_KEY,
  });
  // middleware to extract reasoning tokens
  const enhancedModel = wrapLanguageModel({
    model: modelGroq("deepseek-r1-distill-llama-70b"),
    middleware: extractReasoningMiddleware({ tagName: "think" }),
  });

  return enhancedModel;
};

export async function getModel() {
  const modelProvider = process.env.MODEL_PROVIDER || "groq";

  console.log('modelProvider: ', modelProvider)

  switch (modelProvider) {
    case "ollama":
      return await ollamaProvider();
    case "groq":
      return await groqProvider();
    case "huggingface":
      // Adicione o provider para huggingface se necessário
      throw new Error(`Model provider ${modelProvider} is not supported yet`);
    case "openai":
      // Adicione o provider para openai se necessário
      throw new Error(`Model provider ${modelProvider} is not supported yet`);
    case "cohere":
      // Adicione o provider para cohere se necessário
      throw new Error(`Model provider ${modelProvider} is not supported yet`);
    case "mistral":
      // Adicione o provider para mistral se necessário
      throw new Error(`Model provider ${modelProvider} is not supported yet`);
    case "openrouter":
      return await oprouterProvider();
    case "google":
      return await googleProvider();
    default:
      throw new Error(`Model provider ${modelProvider} is not supported`);
  }
}

/* 
 Models embeddings from langchain
*/

const ollamaEmbeddingProvider = async () => {
  const embeddings = new OllamaEmbeddings({
    baseUrl: process.env.OLLAMA_BASE_URL || 'https://api.ollama.com',
    model: process.env.MODEL_EMBEDDINGS_NAME || 'nomic-embed-text',
  });
  return embeddings;
}


const nomicEmbeddingProvider = async () => {
  const embeddings = new NomicEmbeddings({
    apiKey: process.env.NOMIC_API_KEY,
    modelName: process.env.MODEL_EMBEDDINGS_NAME || "nomic-embed-text-v1.5",
  });
  return embeddings;
}

const huggingfaceEmbeddingProvider = async () => {
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
  });
  return embeddings;
}

const cohereEmbeddingProvider = async () => {
  const embeddings = new CohereEmbeddings({
    model: "embed-multilingual-v3.0"
  });
  return embeddings;
}

const mistralaiEmbeddingProvider = async () => {
  const embeddings = new MistralAIEmbeddings({
    model: "mistral-embed"
  });
  return embeddings;
}

const googleEmbeddingProvider = async () => {
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
    model: process.env.MODEL_EMBEDDINGS_NAME || "text-embedding-004", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT
  });

  return embeddings;
}

export async function getModelEmbedding() {
  const modelEmbeddingProvider = process.env.MODEL_EMBEDDINGS_PROVIDER || "nomic";

  switch (modelEmbeddingProvider) {
    case "ollama":
      return await ollamaEmbeddingProvider();
    case "nomic":
      return await nomicEmbeddingProvider();
    case "huggingface":
      return await huggingfaceEmbeddingProvider();
    case "openai":
      // Adicione o provider para openai se necessário
      throw new Error(`Model embedding provider ${modelEmbeddingProvider} is not supported yet`);
    case "cohere":
      return await cohereEmbeddingProvider();
    case "mistral":
      return await mistralaiEmbeddingProvider();
    case "google":
      return await googleEmbeddingProvider();
    default:
      throw new Error(`Model embedding provider ${modelEmbeddingProvider} is not supported`);
  }
}
