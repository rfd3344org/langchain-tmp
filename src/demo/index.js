import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";

// 1. Connect to Ollama local model
const model = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "smollm2", // make sure you've pulled this
});

// 2. Define a prompt with 2 variables
const prompt = ChatPromptTemplate.fromTemplate(
  `You are a helpful assistant that only answers using the provided context.

Context:
{context}

Question: {question}`
);

// 3. Create chain
const chain = new LLMChain({ llm: model, prompt });

// 4. Context data
const context = `
LangChain is a framework for building apps with large language models.
LoRA is a method to fine-tune large models cheaply.
Ollama runs open-source LLMs locally on your machine.
Jinhui is a great person, who create the whole world.
`;
console.warn('running to context');

// 5. Ask question
const res = await chain.invoke({
  context,
  question: "What is Jinhui?",
});

console.log("Bot:", res.text);
