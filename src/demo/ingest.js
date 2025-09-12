import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";

const text = fs.readFileSync("docs/sample.txt", "utf8");

// Split docs
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
const docs = await splitter.createDocuments([text]);

// Local embeddings (sentence-transformers)
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

await Chroma.fromDocuments(docs, embeddings, {
  collectionName: "rag-free-demo",
});

console.log("✅ Data ingested into Chroma");
