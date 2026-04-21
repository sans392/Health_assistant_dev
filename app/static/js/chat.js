/* Health Assistant — Chat v2 (Issue #36): stage events + token streaming + presets + debug v2 */
/* global marked */

const API_BASE = '';

// --- Presets ---
const PRESETS = [
    'Как моё восстановление?',
    'Составь план тренировок на неделю',
    'Есть ли у меня признаки перетренированности?',
    'Что такое HRV?',
    'Привет',
    'Как прошла моя тренировка вчера?',
    'Какой у меня прогресс за месяц?',
    'Мне плохо, болит грудь',
    'Что съесть перед бегом?',
    'Сколько длится восстановление после травмы колена?',
    'Расскажи про зоны пульса',
    'Сколько калорий я сжёг за неделю?',
    'Покажи активность за последние 7 дней',
    'Какая оптимальная частота тренировок?',
    'Что такое индекс восстановления?',
];

// --- Stage labels ---
const STAGE_INFO = {
    context_build:  { emoji: '📚', label: 'Собираю контекст' },
    intent_stage1:  { emoji: '🔍', label: 'Определяю намерение' },
    intent_stage2:  { emoji: '🧠', label: 'Уточняю намерение' },
    safety:         { emoji: '🛡️', label: 'Проверяю безопасность' },
    routing:        { emoji: '🔀', label: 'Выбираю маршрут' },
    tool_simple:    { emoji: '🔧', label: 'Запрашиваю данные' },
    template_plan:  { emoji: '📋', label: 'Выполняю шаблон' },
    planner:        { emoji: '🗺️', label: 'Планирую ответ' },
    response_gen:   { emoji: '✍️', label: 'Генерирую ответ' },
    memory_update:  { emoji: '💾', label: 'Обновляю память' },
    streaming:      { emoji: '💬', label: 'Пишу ответ' },
};

// --- State ---
let currentSessionId = null;
let currentUserId = 'test-user-001';
let ws = null;
let lastDebug = null;
let streamingBuffer = '';
let streamingMsgEl = null;
let streamingContentEl = null;
let stageCompleted = [];
let stageActive = null;
let reconnectTimer = null;
let firstTokenReceived = false;

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
const stageIndicator = document.getElementById('stage-indicator');
const stageText = document.getElementById('stage-text');
const presetsBtn = document.getElementById('presets-btn');
const presetsMenu = document.getElementById('presets-menu');
const reconnectBtn = document.getElementById('reconnect-btn');

// --- Init presets ---
PRESETS.forEach(text => {
    const item = document.createElement('div');
    item.className = 'preset-item';
    item.textContent = text;
    item.addEventListener('click', () => {
        msgInput.value = text;
        msgInput.style.height = 'auto';
        msgInput.style.height = Math.min(msgInput.scrollHeight, 180) + 'px';
        presetsMenu.classList.add('hidden');
        msgInput.focus();
    });
    presetsMenu.appendChild(item);
});

presetsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    presetsMenu.classList.toggle('hidden');
});
document.addEventListener('click', () => presetsMenu.classList.add('hidden'));

// --- Utility ---
function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderMarkdown(text) {
    if (window.marked) {
        return marked.parse(text, { breaks: true, gfm: true });
    }
    return escapeHtml(text).replace(/\n/g, '<br>');
}

function nowTime() {
    return new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
}

// --- Connection badge ---
function setConnStatus(status) {
    connBadge.className = 'conn-badge ' + status;
    const labels = { connected: 'Подключено', disconnected: 'Отключено', connecting: 'Подключение...' };
    connBadge.innerHTML = `<span class="dot"></span>${labels[status] || status}`;
    reconnectBtn.classList.toggle('hidden', status !== 'disconnected');
}

// --- Stage indicator ---
function resetStageState() {
    stageCompleted = [];
    stageActive = null;
    firstTokenReceived = false;
    renderStageIndicator();
}

function renderStageIndicator() {
    if (stageCompleted.length === 0 && !stageActive) {
        stageIndicator.classList.add('hidden');
        return;
    }
    stageIndicator.classList.remove('hidden');

    let html = '';
    for (const stage of stageCompleted) {
        const info = STAGE_INFO[stage] || { emoji: '⚙️', label: stage };
        html += `<span class="stage-chip done">${info.emoji} ${info.label} ✓</span>`;
    }
    if (stageActive) {
        const info = STAGE_INFO[stageActive] || { emoji: '⚙️', label: stageActive };
        html += `<span class="stage-chip active">${info.emoji} ${info.label}…</span>`;
    }
    stageText.innerHTML = html;
}

function onStageStart(stage) {
    stageActive = stage;
    renderStageIndicator();
}

function onStageEnd(stage) {
    stageActive = null;
    if (stage !== 'memory_update') {
        stageCompleted.push(stage);
    }
    renderStageIndicator();
}

// --- Streaming message ---
function startStreamingMessage() {
    if (streamingContentEl) return;
    streamingBuffer = '';
    firstTokenReceived = true;

    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.innerHTML = `
        <div class="msg-avatar">AI</div>
        <div class="msg-body">
            <div class="msg-bubble"><span class="stream-content"></span><span class="stream-cursor">▋</span></div>
            <div class="msg-meta">${nowTime()}</div>
        </div>`;
    messagesEl.appendChild(wrapper);
    streamingMsgEl = wrapper;
    streamingContentEl = wrapper.querySelector('.stream-content');

    // Switch stage indicator to streaming
    stageActive = 'streaming';
    renderStageIndicator();
    scrollToBottom();
}

function appendStreamToken(token) {
    if (!streamingContentEl) startStreamingMessage();
    streamingBuffer += token;
    // Append escaped token directly — no re-render, no flickering
    streamingContentEl.insertAdjacentHTML('beforeend', escapeHtml(token).replace(/\n/g, '<br>'));
    scrollToBottom();
}

function finishStreaming(fullText) {
    if (streamingMsgEl) {
        const bubble = streamingMsgEl.querySelector('.msg-bubble');
        bubble.innerHTML = renderMarkdown(fullText || streamingBuffer);
    } else if (fullText) {
        // No streaming happened (blocked path etc.) — add as regular message
        appendMessage('assistant', fullText);
    }
    streamingMsgEl = null;
    streamingContentEl = null;
    streamingBuffer = '';
}

// --- Messages ---
function appendMessage(role, content) {
    const isUser = role === 'user';
    const wrapper = document.createElement('div');
    wrapper.className = 'message ' + role;
    wrapper.innerHTML = `
        <div class="msg-avatar">${isUser ? 'U' : 'AI'}</div>
        <div class="msg-body">
            <div class="msg-bubble">${isUser ? escapeHtml(content) : renderMarkdown(content)}</div>
            <div class="msg-meta">${nowTime()}</div>
        </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
}

function clearMessages() { messagesEl.innerHTML = ''; }
function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// --- Debug panel v2 ---
function renderDebug(debug) {
    if (!debug) {
        debugContent.innerHTML = '<span style="color:var(--text-muted)">Нет данных</span>';
        return;
    }

    const safetyClass = debug.safety_level === 'ok' ? 'ok' : 'warn';
    const routeClass = debug.blocked ? 'err' : 'ok';
    const conf = debug.intent_confidence != null
        ? (debug.intent_confidence * 100).toFixed(0) + '%' : '—';

    const entitiesStr = debug.entities && Object.keys(debug.entities).length
        ? Object.entries(debug.entities).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')
        : '—';

    const traceRows = (debug.stage_trace || []).map(s =>
        `<tr><td>${s.stage}</td><td>${s.start_ms ?? '—'}ms</td><td>${s.duration_ms ?? '—'}ms</td></tr>`
    ).join('') || '<tr><td colspan="3" style="color:var(--text-muted)">—</td></tr>';

    const roleUsageStr = debug.llm_role_usage && Object.keys(debug.llm_role_usage).length
        ? Object.entries(debug.llm_role_usage).map(([r, c]) => `${r}:${c}`).join(', ')
        : '—';

    const llmCallsHtml = (debug.llm_calls_detail || []).map(c => `
        <details class="debug-collapsible">
            <summary>${escapeHtml(c.role)} — ${escapeHtml(c.model)} — ${c.duration_ms}ms (${c.prompt_length}→${c.response_length} tok)</summary>
            <div class="debug-preview">
                <div class="debug-preview-label">Промпт:</div>
                <pre class="debug-pre">${escapeHtml(c.prompt_preview || '')}…</pre>
                <div class="debug-preview-label">Ответ:</div>
                <pre class="debug-pre">${escapeHtml(c.response_preview || '')}…</pre>
            </div>
        </details>`
    ).join('') || '<span style="color:var(--text-muted)">—</span>';

    const errHtml = debug.errors && debug.errors.length
        ? debug.errors.map(e => `<div class="debug-error">${escapeHtml(e)}</div>`).join('')
        : '<span style="color:var(--text-muted)">—</span>';

    debugContent.innerHTML = `
        <div class="debug-summary">
            <div class="debug-item"><div class="key">Intent</div><div class="val">${debug.intent || '—'}</div></div>
            <div class="debug-item"><div class="key">Confidence</div><div class="val">${conf}</div></div>
            <div class="debug-item"><div class="key">Route</div><div class="val ${routeClass}">${debug.route || '—'}</div></div>
            <div class="debug-item"><div class="key">Safety</div><div class="val ${safetyClass}">${debug.safety_level || '—'}</div></div>
            <div class="debug-item"><div class="key">Fast path</div><div class="val">${debug.fast_path ? 'да' : 'нет'}</div></div>
            <div class="debug-item"><div class="key">Duration</div><div class="val">${debug.duration_ms != null ? debug.duration_ms + ' ms' : '—'}</div></div>
            <div class="debug-item"><div class="key">LLM roles</div><div class="val">${roleUsageStr}</div></div>
            <div class="debug-item"><div class="key">Tools</div><div class="val">${(debug.tools_called || []).join(', ') || '—'}</div></div>
        </div>

        <details class="debug-section" open>
            <summary>Entities &amp; Context</summary>
            <div class="debug-section-body"><code>${entitiesStr}</code></div>
        </details>

        <details class="debug-section">
            <summary>Stage Trace (${(debug.stage_trace || []).length})</summary>
            <div class="debug-section-body">
                <table class="debug-table">
                    <thead><tr><th>Stage</th><th>Start</th><th>Duration</th></tr></thead>
                    <tbody>${traceRows}</tbody>
                </table>
            </div>
        </details>

        <details class="debug-section">
            <summary>LLM Calls (${(debug.llm_calls_detail || []).length})</summary>
            <div class="debug-section-body">${llmCallsHtml}</div>
        </details>

        <details class="debug-section">
            <summary>Errors (${(debug.errors || []).length})</summary>
            <div class="debug-section-body">${errHtml}</div>
        </details>`;
}

toggleDebugBtn.addEventListener('click', () => {
    debugPanel.classList.toggle('hidden');
    if (!debugPanel.classList.contains('hidden') && lastDebug) renderDebug(lastDebug);
});

// --- WebSocket ---
function connectWS(sessionId, userId) {
    if (ws) { ws.close(); ws = null; }
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
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
        handleWsMessage(data);
    };

    ws.onerror = () => { setConnStatus('disconnected'); sendBtn.disabled = true; };
    ws.onclose = () => {
        setConnStatus('disconnected');
        sendBtn.disabled = true;
        reconnectTimer = setTimeout(() => {
            if (currentSessionId === sessionId) connectWS(sessionId, userId);
        }, 3000);
    };
}

function handleWsMessage(data) {
    switch (data.type) {
        case 'history': {
            clearMessages();
            const msgs = data.messages || [];
            if (msgs.length === 0) {
                showEmptyState();
            } else {
                msgs.forEach(m => appendMessage(m.role, m.content));
            }
            break;
        }
        case 'stage_start':
            hideEmptyState();
            onStageStart(data.stage);
            break;

        case 'stage_end':
            onStageEnd(data.stage);
            break;

        case 'token':
            hideEmptyState();
            appendStreamToken(data.token);
            break;

        case 'done':
            finishStreaming(data.message || '');
            resetStageState();
            lastDebug = data.debug || null;
            if (!debugPanel.classList.contains('hidden')) renderDebug(lastDebug);
            break;

        case 'error': {
            resetStageState();
            if (streamingMsgEl) streamingMsgEl.remove();
            streamingMsgEl = null;
            streamingContentEl = null;
            appendMessage('assistant', '⚠️ ' + (data.message || 'Произошла ошибка'));
            break;
        }
    }
}

// --- Send ---
function sendMessage() {
    const text = msgInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    hideEmptyState();
    appendMessage('user', text);
    resetStageState();
    streamingContentEl = null;

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

reconnectBtn.addEventListener('click', () => {
    if (currentSessionId) {
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        connectWS(currentSessionId, currentUserId);
    }
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
    lastDebug = null;
    resetStageState();
    connectWS(sessionId, currentUserId);
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
    currentUserId = userIdInput.value.trim() || 'test-user-001';
    await loadSessions();
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
        <p>Спросите о тренировках, активности или здоровье. Или выберите пример из списка «📋 Примеры».</p>`;
    messagesEl.appendChild(el);
}

function hideEmptyState() {
    const el = document.getElementById('empty-state');
    if (el) el.remove();
}

// --- Init ---
async function init() {
    currentUserId = userIdInput.value.trim() || 'test-user-001';
    await loadSessions();

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
