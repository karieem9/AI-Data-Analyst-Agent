const chatThread = document.getElementById("chatThread");
const welcome = document.getElementById("welcome");
const composerForm = document.getElementById("composerForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

const datasetEmpty = document.getElementById("datasetEmpty");
const datasetInfo = document.getElementById("datasetInfo");
const datasetName = document.getElementById("datasetName");
const datasetMeta = document.getElementById("datasetMeta");
const fileInput = document.getElementById("fileInput");
const fileInputReplace = document.getElementById("fileInputReplace");

const profilePanel = document.getElementById("profilePanel");
const profileList = document.getElementById("profileList");

const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");
const historySearch = document.getElementById("historySearch");

let historyData = [];
let msgCounter = 0;

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function scrollToBottom() {
  chatThread.scrollTop = chatThread.scrollHeight;
}

// ---------- upload ----------

async function handleUpload(file) {
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);

  datasetEmpty.querySelector("p").textContent = "Uploading...";

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");

    datasetEmpty.hidden = true;
    datasetInfo.hidden = false;
    datasetName.textContent = data.filename;
    datasetMeta.textContent = `${data.rows.toLocaleString()} rows, ${data.columns} columns`;

    profilePanel.hidden = false;
    profileList.innerHTML = data.profile.map(col => `
      <div class="profile-row">
        <span class="col-name" title="${escapeHtml(col.column)}">${escapeHtml(col.column)}</span>
        <span class="col-dtype">${escapeHtml(col.dtype)}</span>
      </div>
    `).join("");

    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.placeholder = "What was total revenue last month?";

    historyData = [];
    renderHistoryList();
    chatThread.innerHTML = "";
    chatThread.appendChild(welcome);
    welcome.querySelector("h1").textContent = "Dataset loaded";
    welcome.querySelector("p").textContent = `Ask a question about ${data.filename}.`;
  } catch (err) {
    datasetEmpty.querySelector("p").textContent = err.message;
  }
}

fileInput.addEventListener("change", (e) => handleUpload(e.target.files[0]));
fileInputReplace.addEventListener("change", (e) => handleUpload(e.target.files[0]));

// ---------- chat ----------

function addUserMessage(question) {
  const row = document.createElement("div");
  row.className = "chat-row user";
  row.innerHTML = `<div class="chat-bubble user">${escapeHtml(question)}</div>`;
  chatThread.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  const id = `msg-${++msgCounter}`;
  const row = document.createElement("div");
  row.className = "chat-row agent";
  row.id = id;
  row.innerHTML = `
    <div class="chat-bubble agent">
      <div class="agent-header"><span class="mini-orb"></span><span class="agent-label">Thinking</span></div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  chatThread.appendChild(row);
  scrollToBottom();
  return id;
}

function renderResultBody(data, rowId) {
  if (data.error) {
    return `<div class="agent-error">${escapeHtml(data.error)}</div>`;
  }
  if (data.kind === "metric") {
    return `<div class="metric-value">${escapeHtml(data.metric)}</div>`;
  }
  if (data.kind === "table") {
    const rows = data.table || [];
    if (rows.length === 0) return `<div class="explanation-text">No rows.</div>`;
    const cols = Object.keys(rows[0]);
    const head = cols.map(c => `<th>${escapeHtml(c)}</th>`).join("");
    const body = rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("");
    return `<div class="result-table-wrap"><table class="result-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }
  if (data.kind === "figure") {
    return `<div class="chart-container" id="chart-${rowId}"></div>`;
  }
  return "";
}

function renderAgentMessage(rowId, data) {
  const row = document.getElementById(rowId);
  if (!row) return;

  const explanationHtml = data.explanation
    ? `<div class="explanation-text">${escapeHtml(data.explanation)}</div>`
    : (data.explanation_error ? `<div class="agent-error">${escapeHtml(data.explanation_error)}</div>` : "");

  const retryBadge = data.attempts > 1
    ? `<span class="retry-badge" title="The first attempt failed; this is the corrected version">fixed after retry</span>`
    : "";

  row.innerHTML = `
    <div class="chat-bubble agent">
      <div class="agent-header"><span class="mini-orb"></span><span class="agent-label">Answer</span>${retryBadge}</div>
      ${explanationHtml}
      ${renderResultBody(data, rowId)}
      <details class="code-block">
        <summary>View generated code</summary>
        <pre>${escapeHtml(data.code || "")}</pre>
      </details>
    </div>`;

  if (data.kind === "figure" && data.figure) {
    const el = document.getElementById(`chart-${rowId}`);
    if (el) {
      Plotly.newPlot(el, data.figure.data, {
        ...data.figure.layout,
        margin: { t: 30, r: 20, l: 50, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Plus Jakarta Sans, sans-serif", color: "#142a4c" },
      }, { responsive: true, displayModeBar: false });
    }
  }

  scrollToBottom();
}

async function askQuestion(question) {
  addUserMessage(question);
  const typingId = addTypingIndicator();

  questionInput.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) {
      renderAgentMessage(typingId, { error: data.detail || "Something went wrong." });
      return;
    }
    renderAgentMessage(typingId, data);
    historyData.unshift(data);
    renderHistoryList();
  } catch (err) {
    renderAgentMessage(typingId, { error: "Network error: " + err.message });
  } finally {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionInput.value.trim();
  if (!q) return;
  questionInput.value = "";
  askQuestion(q);
});

// ---------- history ----------

function renderHistoryList(filter = "") {
  const items = historyData.filter(h =>
    h.question.toLowerCase().includes(filter.toLowerCase())
  );

  historyEmpty.hidden = historyData.length > 0;
  if (historyData.length === 0) {
    historyList.innerHTML = "";
    historyList.appendChild(historyEmpty);
    return;
  }

  historyList.innerHTML = items.map((h, i) => `
    <button type="button" class="history-item" data-index="${historyData.indexOf(h)}">
      <div class="h-q">${escapeHtml(h.question)}</div>
      <div class="h-kind">${h.error ? "error" : (h.kind || "answer")}</div>
    </button>
  `).join("") || `<p class="history-empty">No matches.</p>`;
}

historySearch.addEventListener("input", (e) => renderHistoryList(e.target.value));

historyList.addEventListener("click", (e) => {
  const btn = e.target.closest(".history-item");
  if (!btn) return;
  // Simple affordance: re-ask isn't needed, just scroll the thread into view.
  chatThread.scrollTop = 0;
});
