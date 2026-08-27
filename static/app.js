const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const uploadStatus = document.getElementById("upload-status");
const chatSection = document.getElementById("chat-section");
const docInfo = document.getElementById("doc-info");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function setUploadStatus(text, kind) {
  uploadStatus.textContent = text;
  uploadStatus.className = "status" + (kind ? " " + kind : "");
}

async function uploadFile(file) {
  if (!file) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus("Please choose a PDF file.", "error");
    return;
  }

  dropZoneText.textContent = `Uploading "${file.name}"...`;
  setUploadStatus("Processing PDF (extracting text, chunking, embedding)...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");

    dropZoneText.textContent = `Loaded "${data.filename}" — choose another PDF to replace it`;
    setUploadStatus(`Ready: ${data.num_pages} pages, ${data.num_chunks} chunks indexed.`, "success");

    docInfo.textContent = `Chatting with: ${data.filename} (${data.num_pages} pages, ${data.num_chunks} chunks)`;
    chatSection.hidden = false;
    messages.innerHTML = "";
    questionInput.focus();
  } catch (err) {
    setUploadStatus(err.message, "error");
  }
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  uploadFile(file);
});

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function renderSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `${sources.length} source${sources.length > 1 ? "s" : ""}`;
  details.appendChild(summary);

  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `[${s.index}] page(s) ${s.pages.join(", ")} — score ${s.score.toFixed(3)}`;
    const body = document.createElement("div");
    body.textContent = s.text.trim().slice(0, 300) + (s.text.length > 300 ? "..." : "");
    item.appendChild(meta);
    item.appendChild(body);
    details.appendChild(item);
  }

  wrap.appendChild(details);
  container.appendChild(wrap);
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;

  const pending = addMessage("assistant pending", "Thinking...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed.");

    pending.textContent = data.answer;
    pending.classList.remove("pending");
    renderSources(pending, data.sources);
  } catch (err) {
    pending.textContent = `Error: ${err.message}`;
    pending.classList.remove("pending");
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});
