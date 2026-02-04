// RAG Q&A System - Frontend JavaScript

const API_BASE = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupFileUpload();
});

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            updateStatus('Ready', 'success');
        }
    } catch (error) {
        updateStatus('Offline', 'error');
        console.error('Health check failed:', error);
    }
}

// Update status indicator
function updateStatus(text, type) {
    const statusText = document.getElementById('status-text');
    const statusDot = document.querySelector('.status-dot');

    statusText.textContent = text;

    if (type === 'success') {
        statusDot.style.background = '#10b981';
    } else if (type === 'error') {
        statusDot.style.background = '#ef4444';
    } else if (type === 'loading') {
        statusDot.style.background = '#f59e0b';
    }
}

// File upload setup
function setupFileUpload() {
    const fileInput = document.getElementById('file-input');

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        await uploadFile(file);
        fileInput.value = ''; // Reset input
    });
}

// Upload file
async function uploadFile(file) {
    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.textContent = 'Uploading...';
    uploadStatus.className = 'upload-status';

    updateStatus('Uploading...', 'loading');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.textContent = `✅ ${data.message} (${data.processing_time}s)`;
            uploadStatus.className = 'upload-status success';
            updateStatus('Ready', 'success');

            // Add system message
            addMessage('assistant', `Document "${data.filename}" has been uploaded and processed in ${data.processing_time} seconds. You can now ask questions about it!`);
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        uploadStatus.textContent = `❌ ${error.message}`;
        uploadStatus.className = 'upload-status error';
        updateStatus('Ready', 'success');
        console.error('Upload error:', error);
    }
}

// Handle key press in input
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendQuestion();
    }
}

// Ask example question
function askExample(question) {
    document.getElementById('question-input').value = question;
    sendQuestion();
}

// Send question
async function sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();

    if (!question) return;

    // Clear input
    input.value = '';

    // Hide welcome message if present
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message
    addMessage('user', question);

    // Add loading message
    const loadingId = addMessage('assistant', '<div class="loading-dots"><span></span><span></span><span></span></div>', true);

    // Disable send button
    const sendBtn = document.querySelector('.btn-send');
    sendBtn.disabled = true;
    updateStatus('Thinking...', 'loading');

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                top_k: 3
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Remove loading message
            document.getElementById(loadingId).remove();

            // Add answer
            addMessage('assistant', data.answer);

            // Add metadata if available
            if (data.metadata) {
                addMetadata(data.metadata);
            }

            // Add sources if available
            if (data.sources && data.sources.length > 0) {
                addSources(data.sources);
            }

            updateStatus('Ready', 'success');
        } else {
            throw new Error(data.detail || 'Failed to get answer');
        }
    } catch (error) {
        // Remove loading message
        document.getElementById(loadingId).remove();

        // Add error message
        addMessage('assistant', `❌ Error: ${error.message}`);

        updateStatus('Ready', 'success');
        console.error('Chat error:', error);
    } finally {
        sendBtn.disabled = false;
    }
}

// Format markdown-like text to HTML
function formatText(text) {
    // Don't escape $ symbols - they're needed for LaTeX
    let formatted = text
        .replace(/&(?!#)/g, '&amp;')  // Escape & but not &#
        .replace(/<(?![$/])/g, '&lt;')  // Escape < but not for math
        .replace(/(?<![$/])>/g, '&gt;'); // Escape > but not for math

    // Protect LaTeX expressions from further processing
    const latexBlocks = [];
    let latexIndex = 0;

    // Extract display math ($$...$$)
    formatted = formatted.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
        const placeholder = `__LATEX_DISPLAY_${latexIndex}__`;
        latexBlocks[latexIndex] = match;
        latexIndex++;
        return placeholder;
    });

    // Extract inline math ($...$)
    formatted = formatted.replace(/\$([^\$\n]+?)\$/g, (match) => {
        const placeholder = `__LATEX_INLINE_${latexIndex}__`;
        latexBlocks[latexIndex] = match;
        latexIndex++;
        return placeholder;
    });

    // Convert markdown to HTML (but not LaTeX)
    // Headers
    formatted = formatted.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gm, '<h1>$1</h1>');

    // Bold (but not in LaTeX)
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Lists
    formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
    formatted = formatted.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');

    // Wrap consecutive list items in ul
    formatted = formatted.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
        return '<ul>' + match + '</ul>';
    });

    // Line breaks
    formatted = formatted.replace(/\n\n/g, '<br><br>');
    formatted = formatted.replace(/\n/g, '<br>');

    // Restore LaTeX expressions
    latexBlocks.forEach((latex, index) => {
        formatted = formatted.replace(`__LATEX_DISPLAY_${index}__`, latex);
        formatted = formatted.replace(`__LATEX_INLINE_${index}__`, latex);
    });

    return formatted;
}

// Add message to chat
function addMessage(role, content, isLoading = false) {
    const messagesContainer = document.getElementById('chat-messages');

    const messageDiv = document.createElement('div');
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // Format content if it's from assistant and not loading
    if (role === 'assistant' && !isLoading && !content.includes('<div class="loading-dots">')) {
        messageContent.innerHTML = formatText(content);

        // Trigger MathJax to render LaTeX
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([messageContent]).catch((err) => {
                console.error('MathJax rendering error:', err);
            });
        }
    } else {
        messageContent.innerHTML = content;
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return messageId;
}

// Add metadata panel
function addMetadata(metadata) {
    const messagesContainer = document.getElementById('chat-messages');

    const metadataDiv = document.createElement('div');
    metadataDiv.className = 'metadata-panel';

    const metadataHTML = `
        <div class="metadata-header">📊 Response Details</div>
        <div class="metadata-grid">
            <div class="metadata-item">
                <span class="metadata-label">Model:</span>
                <span class="metadata-value">${metadata.model}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Chunks Used:</span>
                <span class="metadata-value">${metadata.chunks_used}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Retrieval Time:</span>
                <span class="metadata-value">${metadata.retrieval_time.toFixed(2)}s</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Generation Time:</span>
                <span class="metadata-value">${metadata.generation_time.toFixed(2)}s</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Total Time:</span>
                <span class="metadata-value">${metadata.total_time.toFixed(2)}s</span>
            </div>
        </div>
    `;

    metadataDiv.innerHTML = metadataHTML;
    messagesContainer.appendChild(metadataDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Add sources panel
function addSources(sources) {
    const messagesContainer = document.getElementById('chat-messages');

    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources-panel';

    let sourcesHTML = '<div class="sources-header">📚 Sources Used</div>';

    sources.forEach((source, index) => {
        const pageInfo = source.page ? `Page ${source.page}` : 'Page N/A';
        const contentType = source.content_type ? `[${source.content_type.toUpperCase()}]` : '';
        const charCount = source.char_count ? `${source.char_count} chars` : '';
        const tokenCount = source.token_count ? `~${source.token_count} tokens` : '';

        // Create clickable page link if file_path exists
        const pageLink = source.file_path && source.page
            ? `<a href="${source.file_path}#page=${source.page}" target="_blank" class="page-link">${pageInfo}</a>`
            : `<span class="page-info">${pageInfo}</span>`;

        sourcesHTML += `
            <div class="source-item">
                <div class="source-number">${index + 1}</div>
                <div class="source-content">
                    <div class="source-text">${source.text}</div>
                    <div class="source-meta">
                        ${pageLink} • ${source.source} ${contentType}
                        ${charCount ? `• ${charCount}` : ''}
                        ${tokenCount ? `• ${tokenCount}` : ''}
                    </div>
                </div>
            </div>
        `;
    });

    sourcesDiv.innerHTML = sourcesHTML;
    messagesContainer.appendChild(sourcesDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
