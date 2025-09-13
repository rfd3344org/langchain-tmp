const chatDiv = document.getElementById("chat");
const input = document.getElementById("prompt");
const sendBtn = document.getElementById("send");

async function sendMessage() {
  const userText = input.value.trim();
  if (!userText) return;

  // Add user message
  addMessage("user", userText);
  input.value = "";

  // Add AI placeholder
  const aiEl = addMessage("ai", "");

  const cros = 'http://localhost:8080/'
  const streamingUrl = 'http://localhost:11434/api/generate';
  // Fetch response (assuming /api/chat streams back text)
  const response = await fetch(cros + streamingUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: userText,
      model: "smollm2",
    }),
  });


  if (!response.body) {
    aiEl.textContent = "Error: No response stream";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let partial = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    partial += decoder.decode(value, { stream: true });
    aiEl.textContent = partial;
    chatDiv.scrollTop = chatDiv.scrollHeight;
  }
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return div;
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
