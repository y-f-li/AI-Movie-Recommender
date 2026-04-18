const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const messagesEl = document.getElementById('messages');
const emptyState = document.getElementById('emptyState');
const chatScroll = document.getElementById('chatScroll');
const statusPill = document.getElementById('statusPill');

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderSimpleMarkup(value) {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function autoResize() {
  messageInput.style.height = '0px';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 180) + 'px';
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function addMessage(role, content, extraClass = '') {
  emptyState.style.display = 'none';
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${extraClass}`.trim();

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role === 'user' ? 'You' : 'Assistant';

  const body = document.createElement('div');
  if (role === 'assistant') {
    body.innerHTML = renderSimpleMarkup(content);
  } else {
    body.textContent = content;
  }

  bubble.appendChild(label);
  bubble.appendChild(body);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return body;
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    if (data.ok) {
      statusPill.textContent = 'Backend online';
      statusPill.classList.add('online');
      statusPill.classList.remove('offline');
    } else {
      statusPill.textContent = 'Backend needs dataset';
      statusPill.classList.add('offline');
      statusPill.classList.remove('online');
    }
  } catch (error) {
    statusPill.textContent = 'Backend unavailable';
    statusPill.classList.add('offline');
    statusPill.classList.remove('online');
  }
}

async function sendMessage(message) {
  addMessage('user', message);
  const loadingBody = addMessage('assistant', 'Thinking', 'loading-dots');
  sendButton.disabled = true;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();

    const bubble = loadingBody.parentElement;
    bubble.classList.remove('loading-dots');

    if (!response.ok || !data.ok) {
      loadingBody.textContent = data.error || 'Something went wrong.';
      return;
    }

    loadingBody.innerHTML = renderSimpleMarkup(data.response);
  } catch (error) {
    const bubble = loadingBody.parentElement;
    bubble.classList.remove('loading-dots');
    loadingBody.textContent = 'Request failed. Please check the backend console.';
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
    scrollToBottom();
  }
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;
  messageInput.value = '';
  autoResize();
  await sendMessage(message);
});

messageInput.addEventListener('input', autoResize);
messageInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll('.suggestion-btn').forEach((button) => {
  button.addEventListener('click', async () => {
    const text = button.textContent.trim();
    messageInput.value = '';
    autoResize();
    await sendMessage(text);
  });
});

autoResize();
checkHealth();
