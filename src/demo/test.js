import { ChatOllama } from "@langchain/ollama";

const model = new ChatOllama({
  model: "smollm2",  // Default value.
});

const result = await model.invoke(["human", "Hello, how are you?"]);
console.warn('res', result)