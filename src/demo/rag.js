import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Ollama } from "@langchain/community/llms/ollama";
import { RetrievalQAChain } from "langchain/chains";

const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

const vectorStore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: "rag-free-demo",
});

const model = new Ollama({ model: "mistral" });

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.invoke({
  query: "Summarize the document in 3 sentences.",
});

console.log("Answer:", response.text);
