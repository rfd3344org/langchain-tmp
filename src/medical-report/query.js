// import { Ollama } from "@langchain/ollama";
// import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { setupModel, loadVectorStore, setupPrompt } from "./utils.js";




// -------------------- 7. Run Query with Metadata Filter --------------------
async function runQuery(chain, vectorStore, question, filter = null) {
  const retriever = vectorStore.asRetriever({ k: 3, filter });
  const retrievedDocs = await retriever.getRelevantDocuments(question);

  const context = retrievedDocs.map((d) => d.pageContent).join("\n\n");
  console.warn("Retrieved docs:", retrievedDocs.map((d) => d.metadata));

  const res = await chain.invoke({ context, question });
  console.log("Bot:", res.text);
}

// -------------------- Main --------------------
async function main() {
  const model = setupModel();

  const vectorStore = await loadVectorStore();
  const prompt = setupPrompt();
  const chain = new LLMChain({ llm: model, prompt });
  // console.warn('chain', chain)
  // console.warn('vectorStore', vectorStore)


  // Example queries
  console.log("\n--- Query 1 (general) ---");
  await runQuery(chain, vectorStore, "What lifestyle changes are recommended?");

  console.log("\n--- Query 2 (filtered: only John Doe) ---");
  await runQuery(chain, vectorStore, "What recommendations were given?", {
    patient_name: "John Doe",
  });

  console.log("\n--- Query 3 (filtered: only Jane Smith) ---");
  await runQuery(chain, vectorStore, "What tests are recommended?", {
    patient_name: "Jane Smith",
  });
}

main();
