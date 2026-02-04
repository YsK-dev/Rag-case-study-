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
            uploadStatus.textContent = `✅ ${data.message}`;
            uploadStatus.className = 'upload-status success';
            updateStatus('Ready', 'success');
            
            // Add system message
            addMessage('assistant', `Document "${data.filename}" has been uploaded and processed. You can now ask questions about it!`);
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
    messageContent.innerHTML = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}
