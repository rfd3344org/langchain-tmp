import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import { Chroma } from "@langchain/community/vectorstores/chroma";

// import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";

// -------------------- 1. Setup Model --------------------
const model = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "smollm2",
  // model: "deepseek-r1:8b",
});

// -------------------- 2. Prepare Documents --------------------
const text = fs.readFileSync("../docs/sample.txt", "utf8");

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 50 });

const docs = await splitter.createDocuments([text]);
// console.warn('splitter', docs)


// -------------------- 3. Embeddings + Vector Store --------------------
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

await Chroma.fromDocuments(docs, embeddings, {
  collectionName: "rag-demo",
  // url: null,
  url: "http://localhost:8000",
});


// -------------------- 4. Retriever --------------------
const vectorStore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: "rag-demo",
  // url: null,
  url: "http://localhost:8000",
});

// const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);


// -------------------- 5. Prompt + Chain --------------------
const prompt = ChatPromptTemplate.fromTemplate(
  `You are a helpful assistant that only answers using the provided context.

Context:
{context}

Question: {question}`
);

// -------------------- 4. Create Chain --------------------
const chain = new LLMChain({ llm: model, prompt });

// -------------------- 5. Ask Question --------------------
const query = "Summarize the document in 2 sentences. and who is Jinhui?";
const retriever = vectorStore.asRetriever();
const retrievedDocs = await retriever.getRelevantDocuments(query);
const context = retrievedDocs.map((d) => d.pageContent).join("\n\n");

console.warn("Running with retrieved context...");

const res = await chain.invoke({
  context,
  question: query,
});

console.log("Bot:", res.text);
