import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

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

// -------------------- Main --------------------
async function main() {
  // const model = setupModel();
  const docs = await prepareDocuments();
  const embeddings = setupEmbeddings();

  const vectorStore = await setupVectorStore(docs, embeddings);
  // const prompt = setupPrompt();
  // const chain = new LLMChain({ llm: model, prompt });
  // console.warn('chain', chain)
  // console.warn('vectorStore', vectorStore)

}

main();
