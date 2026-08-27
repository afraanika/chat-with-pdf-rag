const authSection = document.getElementById("auth-section");
const mainSection = document.getElementById("main-section");
const userEmailEl = document.getElementById("user-email");
const logoutBtn = document.getElementById("logout-btn");

const tabs = document.querySelectorAll(".tab");
const loginForm = document.getElementById("login-form");
const signupForm = document.getElementById("signup-form");
const authStatus = document.getElementById("auth-status");

const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const chatSection = document.getElementById("chat-section");
const docInfo = document.getElementById("doc-info");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

let documents = [];
let activeDocId = null;
const chatHistory = {}; // docId -> array of {role, text, sources}

function setStatus(el, text, kind) {
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Request failed.");
  return data;
}

// ---- Auth ----

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const isLogin = tab.dataset.tab === "login";
    loginForm.hidden = !isLogin;
    signupForm.hidden = isLogin;
    setStatus(authStatus, "");
  });
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("login-email").value;
  const password = document.getElementById("login-password").value;
  try {
    await api("/api/login", { method: "POST", body: JSON.stringify({ email, password }) });
    await enterApp();
  } catch (err) {
    setStatus(authStatus, err.message, "error");
  }
});

signupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("signup-email").value;
  const password = document.getElementById("signup-password").value;
  try {
    await api("/api/signup", { method: "POST", body: JSON.stringify({ email, password }) });
    await enterApp();
  } catch (err) {
    setStatus(authStatus, err.message, "error");
  }
});

logoutBtn.addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  documents = [];
  activeDocId = null;
  Object.keys(chatHistory).forEach((k) => delete chatHistory[k]);
  mainSection.hidden = true;
  authSection.hidden = false;
});

async function enterApp() {
  const user = await api("/api/me");
  userEmailEl.textContent = user.email;
  authSection.hidden = true;
  mainSection.hidden = false;
  await refreshDocuments();
}

async function checkSession() {
  try {
    await enterApp();
  } catch {
    authSection.hidden = false;
    mainSection.hidden = true;
  }
}

// ---- Documents ----

async function refreshDocuments() {
  documents = await api("/api/documents");
  renderDocList();
  if (documents.length > 0 && !activeDocId) {
    selectDocument(documents[0].id);
  } else if (documents.length === 0) {
    chatSection.hidden = true;
  }
}

function renderDocList() {
  docList.innerHTML = "";
  if (documents.length === 0) {
    const li = document.createElement("li");
    li.className = "doc-empty";
    li.textContent = "No documents yet — upload a PDF above.";
    docList.appendChild(li);
    return;
  }

  for (const doc of documents) {
    const li = document.createElement("li");
    li.className = "doc-item" + (doc.id === activeDocId ? " active" : "");

    const label = document.createElement("span");
    label.textContent = `${doc.filename} (${doc.num_pages}p, ${doc.num_chunks} chunks)`;
    label.addEventListener("click", () => selectDocument(doc.id));

    const del = document.createElement("button");
    del.textContent = "Delete";
    del.className = "secondary small";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      await api(`/api/documents/${doc.id}`, { method: "DELETE" });
      delete chatHistory[doc.id];
      if (activeDocId === doc.id) activeDocId = null;
      await refreshDocuments();
    });

    li.appendChild(label);
    li.appendChild(del);
    docList.appendChild(li);
  }
}

function selectDocument(docId) {
  activeDocId = docId;
  const doc = documents.find((d) => d.id === docId);
  renderDocList();
  chatSection.hidden = false;
  docInfo.textContent = `Chatting with: ${doc.filename} (${doc.num_pages} pages, ${doc.num_chunks} chunks)`;
  renderMessages();
  questionInput.focus();
}

// ---- Upload ----

async function uploadFile(file) {
  if (!file) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    setStatus(uploadStatus, "Please choose a PDF file.", "error");
    return;
  }

  dropZoneText.textContent = `Uploading "${file.name}"...`;
  setStatus(uploadStatus, "Processing PDF (extracting text, chunking, embedding)...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");

    dropZoneText.textContent = "Click to choose a PDF, or drag one here";
    setStatus(uploadStatus, `Ready: ${data.num_pages} pages, ${data.num_chunks} chunks indexed.`, "success");
    chatHistory[data.id] = [];
    await refreshDocuments();
    selectDocument(data.id);
  } catch (err) {
    setStatus(uploadStatus, err.message, "error");
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
  uploadFile(e.dataTransfer.files[0]);
});

// ---- Chat ----

function renderMessages() {
  messages.innerHTML = "";
  const history = chatHistory[activeDocId] || [];
  for (const msg of history) {
    const el = addMessageEl(msg.role, msg.text);
    if (msg.sources) renderSources(el, msg.sources);
  }
  messages.scrollTop = messages.scrollHeight;
}

function addMessageEl(role, text) {
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
  if (!question || !activeDocId) return;

  const history = chatHistory[activeDocId] || (chatHistory[activeDocId] = []);

  history.push({ role: "user", text: question });
  addMessageEl("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;

  const pending = addMessageEl("assistant pending", "Thinking...");

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ document_id: activeDocId, question }),
    });

    pending.textContent = data.answer;
    pending.classList.remove("pending");
    renderSources(pending, data.sources);
    history.push({ role: "assistant", text: data.answer, sources: data.sources });
  } catch (err) {
    pending.textContent = `Error: ${err.message}`;
    pending.classList.remove("pending");
    history.push({ role: "assistant", text: `Error: ${err.message}` });
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

checkSession();
