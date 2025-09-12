// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

// const embeddings = new HuggingFaceTransformersEmbeddings({
//   modelName: "Xenova/all-MiniLM-L6-v2",
// });

// const vectorStore = await Chroma.fromExistingCollection(embeddings, {
//   collectionName: "test",
//   url: "http://localhost:8000",
//   tenant: "default_tenant",
//   database: "default_database",
// });

// console.warn('vectorStore', vectorStore)
// console.log("Connected OK to default tenant/database");

import { CloudClient } from "chromadb";

const client = new CloudClient({
  apiKey: 'ck-8VEzLCMBiYYz7rK31NgxaEBQ1u8EiVw9PoMuMzFbpRAt',
  tenant: '10a56656-9b0f-4e17-98d1-a7b7be72da24',
  database: 'EstelDemo'
});


console.warn('cl ', client)