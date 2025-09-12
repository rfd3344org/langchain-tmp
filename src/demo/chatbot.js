import { Ollama } from "langchain/llms/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";

// 1. Use Ollama model (runs locally)
const model = new Ollama({
  baseUrl: "http://localhost:11434", // Ollama default
  model: "SmolLM2",                   // or "mistral", "gemma", etc.
});

// 2. Define a simple prompt
const prompt = ChatPromptTemplate.fromTemplate(
  `You are a helpful assistant that only answers using the provided context.

Context:
{context}

Question: {question}`
);

// 3. Wrap into a chain
const chain = new LLMChain({ llm: model, prompt });

// 4. Fake local knowledge
const context = `
LangChain is a framework for building apps with large language models.
LoRA is a method to fine-tune large models cheaply.
Ollama runs open-source LLMs locally on your machine.
`;

// 5. Ask the bot
const res = await chain.run({
  context,
  question: "What is LoRA?",
});

console.log("Bot:", res);
