/* Health Assistant — Chat UI logic */
/* global marked */

const API_BASE = '';

// --- State ---
let currentSessionId = null;
let currentUserId = 'user_1';
let ws = null;
let lastMeta = null;

// --- DOM refs ---
const messagesEl = document.getElementById('messages');
const msgInput = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const sessionsList = document.getElementById('sessions-list');
const connBadge = document.getElementById('conn-badge');
const debugPanel = document.getElementById('debug-panel');
const debugContent = document.getElementById('debug-content');
const userIdInput = document.getElementById('user-id-input');
const toggleDebugBtn = document.getElementById('toggle-debug');

// --- Utility ---
function genUUID() {
  return crypto.randomUUID ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function formatTime(iso) {
  try { return new Date(iso).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }); }
  catch { return ''; }
}

function renderMarkdown(text) {
  if (window.marked) {
    return marked.parse(text, { breaks: true, gfm: true });
  }
  // Fallback: escape HTML and convert newlines
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

// --- Connection badge ---
function setConnStatus(status) {
  connBadge.className = 'conn-badge ' + status;
  const labels = { connected: 'Подключено', disconnected: 'Отключено', connecting: 'Подключение...' };
  connBadge.innerHTML = `<span class="dot"></span>${labels[status] || status}`;
}

// --- Messages rendering ---
function appendMessage(role, content, meta) {
  const isUser = role === 'user';
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + role;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = isUser ? 'U' : 'AI';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = isUser ? escapeHtml(content) : renderMarkdown(content);

  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';
  metaEl.textContent = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });

  body.appendChild(bubble);
  body.appendChild(metaEl);
  wrapper.appendChild(avatar);
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  scrollToBottom();

  if (meta) {
    lastMeta = meta;
    if (!debugPanel.classList.contains('hidden')) renderDebug(meta);
  }
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function clearMessages() { messagesEl.innerHTML = ''; }

function showTyping() {
  removeTyping();
  const el = document.createElement('div');
  el.id = 'typing';
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// --- Debug panel ---
function renderDebug(meta) {
  if (!meta) { debugContent.innerHTML = '<span style="color:var(--text-muted)">Нет данных</span>'; return; }

  const safetyClass = meta.safety_level === 'ok' ? 'ok' : 'warn';
  const routeClass = meta.blocked ? 'err' : 'ok';

  debugContent.innerHTML = `
    <div class="debug-grid">
      <div class="debug-item"><div class="key">Intent</div><div class="val">${meta.intent || '—'}</div></div>
      <div class="debug-item"><div class="key">Confidence</div><div class="val">${meta.intent_confidence != null ? (meta.intent_confidence * 100).toFixed(0) + '%' : '—'}</div></div>
      <div class="debug-item"><div class="key">Route</div><div class="val ${routeClass}">${meta.route || '—'}</div></div>
      <div class="debug-item"><div class="key">Safety</div><div class="val ${safetyClass}">${meta.safety_level || '—'}</div></div>
      <div class="debug-item"><div class="key">Fast path</div><div class="val">${meta.fast_path ? 'да' : 'нет'}</div></div>
      <div class="debug-item"><div class="key">Duration</div><div class="val">${meta.duration_ms != null ? meta.duration_ms + ' ms' : '—'}</div></div>
      <div class="debug-item"><div class="key">Tools</div><div class="val">${(meta.tools_called || []).join(', ') || '—'}</div></div>
      <div class="debug-item"><div class="key">Modules</div><div class="val">${(meta.modules_used || []).join(', ') || '—'}</div></div>
      ${meta.errors && meta.errors.length ? `<div class="debug-item" style="grid-column:1/-1"><div class="key">Errors</div><div class="val err">${meta.errors.join('; ')}</div></div>` : ''}
    </div>`;
}

toggleDebugBtn.addEventListener('click', () => {
  debugPanel.classList.toggle('hidden');
  if (!debugPanel.classList.contains('hidden') && lastMeta) renderDebug(lastMeta);
});

// --- WebSocket ---
function connectWS(sessionId, userId) {
  if (ws) { ws.close(); ws = null; }
  setConnStatus('connecting');
  sendBtn.disabled = true;

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/chat/${sessionId}?user_id=${encodeURIComponent(userId)}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    setConnStatus('connected');
    sendBtn.disabled = false;
    msgInput.focus();
  };

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }

    if (data.type === 'history') {
      clearMessages();
      (data.messages || []).forEach(m => appendMessage(m.role, m.content));
      if (data.messages && data.messages.length === 0) showEmptyState();
    } else if (data.type === 'typing') {
      showTyping();
    } else if (data.type === 'message') {
      removeTyping();
      hideEmptyState();
      appendMessage(data.role || 'assistant', data.content, data.meta);
    } else if (data.type === 'error') {
      removeTyping();
      appendMessage('assistant', '⚠️ ' + data.content);
    }
  };

  ws.onerror = () => { setConnStatus('disconnected'); sendBtn.disabled = true; };
  ws.onclose = () => {
    setConnStatus('disconnected');
    sendBtn.disabled = true;
    // Reconnect after 3s
    setTimeout(() => { if (currentSessionId === sessionId) connectWS(sessionId, userId); }, 3000);
  };
}

// --- Send message ---
function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  hideEmptyState();
  appendMessage('user', text);
  ws.send(JSON.stringify({ message: text }));
  msgInput.value = '';
  msgInput.style.height = '';
  msgInput.focus();
}

sendBtn.addEventListener('click', sendMessage);
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 180) + 'px';
});

// --- Sessions ---
async function loadSessions() {
  try {
    const resp = await fetch(`${API_BASE}/api/chat/sessions?user_id=${encodeURIComponent(currentUserId)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderSessionsList(data.items || []);
  } catch (e) { console.error('loadSessions error:', e); }
}

function renderSessionsList(sessions) {
  sessionsList.innerHTML = '';
  if (sessions.length === 0) {
    sessionsList.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px 12px;">Нет сессий</div>';
    return;
  }
  sessions.forEach(s => {
    const el = document.createElement('div');
    el.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
    el.dataset.id = s.id;
    const date = new Date(s.updated_at).toLocaleString('ru-RU', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    el.innerHTML = `<span>Сессия ${s.id.slice(0, 8)}…</span><small>${date}</small>`;
    el.addEventListener('click', () => switchSession(s.id));
    sessionsList.appendChild(el);
  });
}

async function switchSession(sessionId) {
  currentSessionId = sessionId;
  clearMessages();
  lastMeta = null;
  connectWS(sessionId, currentUserId);
  // Update active class
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === sessionId);
  });
}

async function createNewSession() {
  try {
    const resp = await fetch(`${API_BASE}/api/chat/sessions?user_id=${encodeURIComponent(currentUserId)}`, { method: 'POST' });
    if (!resp.ok) return;
    const session = await resp.json();
    await loadSessions();
    await switchSession(session.id);
  } catch (e) { console.error('createNewSession error:', e); }
}

document.getElementById('new-session-btn').addEventListener('click', createNewSession);

userIdInput.addEventListener('change', async () => {
  currentUserId = userIdInput.value.trim() || 'user_1';
  await loadSessions();
  // Create session for new user if none exists
  if (!currentSessionId) await createNewSession();
});

// --- Empty state ---
function showEmptyState() {
  if (document.getElementById('empty-state')) return;
  const el = document.createElement('div');
  el.id = 'empty-state';
  el.className = 'empty-state';
  el.innerHTML = `
    <div style="font-size:48px">🏋️</div>
    <h3>Health Assistant</h3>
    <p>Спросите о ваших тренировках, активности или здоровье. Например: «Сколько я бегал на прошлой неделе?»</p>`;
  messagesEl.appendChild(el);
}

function hideEmptyState() {
  const el = document.getElementById('empty-state');
  if (el) el.remove();
}

// --- Init ---
async function init() {
  currentUserId = userIdInput.value.trim() || 'user_1';
  await loadSessions();

  // Find latest session or create one
  const resp = await fetch(`${API_BASE}/api/chat/sessions?user_id=${encodeURIComponent(currentUserId)}`);
  const data = resp.ok ? await resp.json() : { items: [] };
  const sessions = data.items || [];

  if (sessions.length > 0) {
    await switchSession(sessions[0].id);
  } else {
    await createNewSession();
  }
}

init();
