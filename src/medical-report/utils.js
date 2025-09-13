import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// --------------------  Setup LLM Model --------------------
export function setupModel() {
  return new Ollama({
    baseUrl: "http://localhost:11434",
    model: "smollm2",
    // model: 'deepseek-r1:8b'
  });
}


//  -------------------- Setup Vector Store --------------------
function setupEmbeddings() {
  // Setup Embeddings
  return new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });
}

export async function insert2VectorStore(docs) {
  const embeddings = setupEmbeddings();
  const resp = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: "medical-reports",
    url: "http://localhost:8000",
  });
  return resp;
}

export function loadVectorStore() {
  const embeddings = setupEmbeddings();
  return Chroma.fromExistingCollection(embeddings, {
    collectionName: "medical-reports",
    url: "http://localhost:8000",
  });
}


// --------------------  Prompt Template --------------------
export function setupPrompt() {
  return ChatPromptTemplate.fromTemplate(
    `You are a medical assistant. Use the provided context only.

Context:
{context}

Question: {question}`
  );
}
