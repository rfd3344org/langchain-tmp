import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";

// -------------------- 1. Setup Model --------------------
function setupModel() {
  return new Ollama({
    baseUrl: "http://localhost:11434",
    model: "smollm2",
    // model: "deepseek-r1:8b",
  });
}

// -------------------- 2. Load & Split Documents --------------------
async function loadDocuments(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  return splitter.createDocuments([text]);
}

// -------------------- 3. Create Embeddings --------------------
function setupEmbeddings() {
  return new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });
}

// -------------------- 4. Setup Vector Store --------------------
async function setupVectorStore(docs, embeddings) {
  // Store documents (indexing)
  await Chroma.fromDocuments(docs, embeddings, {
    collectionName: "rag-demo",
    url: "http://localhost:8000",
  });

  // Load existing collection for querying
  return Chroma.fromExistingCollection(embeddings, {
    collectionName: "rag-demo",
    url: "http://localhost:8000",
  });
}

// -------------------- 5. Create Prompt Template --------------------
function setupPrompt() {
  return ChatPromptTemplate.fromTemplate(
    `You are a helpful assistant that only answers using the provided context.

Context:
{context}

Question: {question}`
  );
}

// -------------------- 6. Setup Chain --------------------
function setupChain(model, prompt) {
  return new LLMChain({ llm: model, prompt });
}

// -------------------- 7. Run Query --------------------
async function runQuery(chain, vectorStore, query) {
  // const retriever = vectorStore.asRetriever({ searchType: "mmr", k: 5 });
  const retriever = vectorStore.asRetriever({ k: 5 }); // similarity search only

  const retrievedDocs = await retriever.getRelevantDocuments(query);
  const context = retrievedDocs.map((d) => d.pageContent).join("\n\n");

  console.warn("Running with retrieved context...");

  const res = await chain.invoke({ context, question: query });
  // console.log("Bot:", res.text);
  return res.text;
}

// -------------------- Main Flow --------------------
async function main() {
  const model = setupModel();
  const docs = await loadDocuments("./sample.txt");
  const embeddings = setupEmbeddings();
  console.warn('loaded embeddings');
  const vectorStore = await setupVectorStore(docs, embeddings);
  console.warn('loaded vectorStore');
  const prompt = setupPrompt();
  console.warn('loaded prompt');
  const chain = setupChain(model, prompt);
  console.warn('loaded chain');

  const query = "Summarize the document in 2 sentences. and who is Jinhui?";
  const response = await runQuery(chain, vectorStore, query);
  console.warn('response:', response);
}

main();
