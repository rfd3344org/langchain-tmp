

import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";



function setupEmbeddings() {
  return new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
    // model: { dtype: "fp32" }, // or "fp16" if GPU supported
  });
}


const embeddings = setupEmbeddings();

const vectorStore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: "medical-reports",
  // url: "http://localhost:8000",
  collectionName: "medical-reports",
  host: "localhost",
  port: 8000,
  ssl: false,
});

// direct query to inspect
const results = await vectorStore.similaritySearch("test", 3);
console.log(JSON.stringify(results, null, 2));
