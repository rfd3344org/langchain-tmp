import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

app.get("/api/chat", async (req, res) => {
  // const { prompt } = req.body;

  const prompt = 'explain langchain';

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  // Stream from Ollama
  const ollamaRes = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "smollm2", prompt }),
  });

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });

    // Ollama sends JSON lines per chunk
    const lines = chunk.split("\n").filter(Boolean);
    for (const line of lines) {
      const data = JSON.parse(line);
      if (data.response) {
        res.write(`data: ${data.response}\n\n`);
      }
      if (data.done) {
        res.write("event: end\ndata: \n\n");
        res.end();
        return;
      }
    }
  }
});

app.listen(3000, () => console.log("✅ Server running at http://localhost:3000"));
