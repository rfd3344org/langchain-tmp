import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";



// -------------------- 1. Setup Model --------------------
function setupModel() {
  return new Ollama({
    baseUrl: "http://localhost:11434",
    model: "smollm2",
    // model: 'deepseek-r1:8b'
  });
}

// -------------------- 2. Example Medical Reports --------------------
const reports = [
  {
    patient_name: "John Doe",
    report_date: "2025-09-01",
    text: `
    Patient: John Doe
    Date: 2025-09-01
    Age: 58
    Condition: Hypertension, mild chest pain

    Summary:
    Patient reports mild chest discomfort when exercising. Blood pressure consistently above normal range.
    Recommending medication adjustment and lifestyle changes.

    Recommendations:
    - Reduce sodium intake
    - Increase physical activity
    - Schedule follow-up in 3 months
    `,
  },
  {
    patient_name: "Jane Smith",
    report_date: "2025-08-20",
    text: `
    Patient: Jane Smith
    Date: 2025-08-20
    Age: 62
    Condition: Arrhythmia

    Summary:
    Patient experiences irregular heartbeat. Monitoring with ECG recommended.
    Medication adjustment suggested.

    Recommendations:
    - ECG monitoring
    - Avoid caffeine
    - Follow-up in 1 month
    `,
  },
];

// -------------------- 3. Chunk Reports --------------------
async function prepareDocuments() {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 50,
  });

  let docs = [];
  for (const report of reports) {
    const chunks = await splitter.createDocuments([report.text], {
      patient_name: report.patient_name,
      report_date: report.report_date,
    });
    docs = docs.concat(chunks);
  }
  return docs;
}

// -------------------- 4. Setup Embeddings --------------------
function setupEmbeddings() {
  return new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });
}

// -------------------- 5. Setup Vector Store --------------------
async function setupVectorStore(docs, embeddings) {
  await Chroma.fromDocuments(docs, embeddings, {
    collectionName: "medical-reports",
    url: "http://localhost:8000",
  });

  return Chroma.fromExistingCollection(embeddings, {
    collectionName: "medical-reports",
    url: "http://localhost:8000",
  });
}

// -------------------- 6. Prompt Template --------------------
function setupPrompt() {
  return ChatPromptTemplate.fromTemplate(
    `You are a medical assistant. Use the provided context only.

Context:
{context}

Question: {question}`
  );
}

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
  const docs = await prepareDocuments();
  const embeddings = setupEmbeddings();

  const vectorStore = await setupVectorStore(docs, embeddings);
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
