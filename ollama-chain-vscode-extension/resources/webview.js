(function () {
    const vscode = acquireVsCodeApi();

    const modeSelect = document.getElementById('modeSelect');
    const modeDesc = document.getElementById('modeDesc');
    const webSearchCheck = document.getElementById('webSearchCheck');
    const maxIter = document.getElementById('maxIter');
    const iterationsRow = document.getElementById('iterationsRow');
    const chatArea = document.getElementById('chatArea');
    const welcome = document.getElementById('welcome');
    const progressBar = document.getElementById('progressBar');
    const promptInput = document.getElementById('promptInput');
    const sendBtn = document.getElementById('sendBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const clearBtn = document.getElementById('clearBtn');
    const logBtn = document.getElementById('logBtn');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsBody = document.getElementById('settingsBody');
    const settingsArrow = document.getElementById('settingsArrow');

    const MODE_DESCRIPTIONS = {
        cascade: 'Chain ALL models smallest to largest, each refining the answer.',
        auto: 'Router classifies complexity and picks the best strategy automatically.',
        agent: 'Autonomous agent with planning, memory, tools, and dynamic control flow.',
        route: 'Fast model scores complexity (1-5), routes to fast or strong.',
        pipeline: 'Fast model extracts key points, strong model provides deep analysis.',
        verify: 'Fast model drafts, strong model verifies and refines the answer.',
        consensus: 'All models answer independently, strongest merges the best parts.',
        search: 'Always fetches web results first, strongest model synthesizes.',
        fast: 'Direct to the smallest/fastest model for quick answers.',
        strong: 'Direct to the largest/strongest model for best quality.',
    };

    let isRunning = false;
    let currentConfig = {};

    // ── Settings Toggle ──

    settingsToggle.addEventListener('click', function () {
        settingsBody.classList.toggle('open');
        settingsArrow.classList.toggle('open');
    });

    // ── Mode Selection ──

    modeSelect.addEventListener('change', updateModeUI);

    function updateModeUI() {
        const mode = modeSelect.value;
        modeDesc.textContent = MODE_DESCRIPTIONS[mode] || '';
        iterationsRow.style.display = mode === 'agent' ? 'flex' : 'none';
    }

    // ── Send ──

    function getOptions() {
        return {
            mode: modeSelect.value,
            webSearch: webSearchCheck.checked,
            maxIterations: parseInt(maxIter.value, 10) || 15,
            ollamaUrl: currentConfig.ollamaUrl || 'http://localhost:11434',
            cliCommand: currentConfig.cliCommand || 'ollama-chain',
            cliTimeout: currentConfig.cliTimeout || 300,
        };
    }

    function sendPrompt() {
        const prompt = promptInput.value.trim();
        if (!prompt || isRunning) { return; }

        promptInput.value = '';
        promptInput.style.height = 'auto';
        progressBar.innerHTML = '';
        progressBar.classList.remove('visible');

        vscode.postMessage({ type: 'sendPrompt', prompt: prompt, options: getOptions() });
    }

    sendBtn.addEventListener('click', sendPrompt);

    promptInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendPrompt();
        }
    });

    promptInput.addEventListener('input', function () {
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 120) + 'px';
    });

    cancelBtn.addEventListener('click', function () {
        vscode.postMessage({ type: 'cancel' });
    });

    clearBtn.addEventListener('click', function () {
        vscode.postMessage({ type: 'clearChat' });
    });

    logBtn.addEventListener('click', function () {
        vscode.postMessage({ type: 'showOutput' });
    });

    // ── Markdown Rendering ──

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function renderMarkdown(text) {
        var BT = String.fromCharCode(96);
        var BT3 = BT + BT + BT;

        // 1. Extract fenced code blocks before escaping
        var codeBlocks = [];
        var cbRegex = new RegExp(BT3 + '(\\w*)?\\n([\\s\\S]*?)' + BT3, 'g');
        text = text.replace(cbRegex, function (_, lang, code) {
            var idx = codeBlocks.length;
            var id = 'cb-' + Math.random().toString(36).substr(2, 8);
            codeBlocks.push({ lang: lang || '', code: code.trim(), id: id });
            return '\x00CB' + idx + '\x00';
        });

        // 2. Extract inline code before escaping
        var inlineCode = [];
        var icRegex = new RegExp(BT + '([^' + BT + '\\n]+)' + BT, 'g');
        text = text.replace(icRegex, function (_, code) {
            var idx = inlineCode.length;
            inlineCode.push(code);
            return '\x00IC' + idx + '\x00';
        });

        // 3. Escape HTML
        var html = escapeHtml(text);

        // 4. Bold & italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

        // 5. Headers
        html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

        // 6. Bullet lists (- or *)
        html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');

        // 7. Numbered lists
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // 8. Wrap consecutive <li> in <ul>
        html = html.replace(/((?:<li>[\s\S]*?<\/li>\s*)+)/g, '<ul>$1</ul>');

        // 9. Line breaks (but not inside block elements)
        html = html.replace(/\n/g, '<br>');
        html = html.replace(/<br>\s*(<\/?(?:ul|ol|li|h[2-4]|pre|div))/g, '$1');
        html = html.replace(/(<\/(?:ul|ol|h[2-4]|pre|div)>)\s*<br>/g, '$1');

        // 10. Re-insert inline code
        for (var i = 0; i < inlineCode.length; i++) {
            html = html.replace('\x00IC' + i + '\x00', '<code>' + escapeHtml(inlineCode[i]) + '</code>');
        }

        // 11. Re-insert code blocks
        for (var j = 0; j < codeBlocks.length; j++) {
            var cb = codeBlocks[j];
            var langLabel = cb.lang
                ? '<span class="code-lang">' + escapeHtml(cb.lang) + '</span>'
                : '';
            var codeHtml =
                '<div class="code-block-wrap">' +
                langLabel +
                '<pre id="' + cb.id + '"><code>' + escapeHtml(cb.code) + '</code></pre>' +
                '<div class="code-actions">' +
                '<button class="code-action-btn" data-action="copy" data-target="' + cb.id + '">Copy</button>' +
                '<button class="code-action-btn" data-action="insert" data-target="' + cb.id + '">Insert</button>' +
                '</div></div>';
            html = html.replace('\x00CB' + j + '\x00', codeHtml);
        }

        return html;
    }

    // ── Code Action Buttons (event delegation) ──

    chatArea.addEventListener('click', function (e) {
        var btn = e.target.closest('.code-action-btn');
        if (!btn) { return; }

        var action = btn.getAttribute('data-action');
        var targetId = btn.getAttribute('data-target');
        var pre = document.getElementById(targetId);
        if (!pre) { return; }

        var code = pre.textContent || '';

        if (action === 'copy') {
            navigator.clipboard.writeText(code).then(function () {
                btn.textContent = 'Copied!';
                setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
            }).catch(function () {});
        } else if (action === 'insert') {
            vscode.postMessage({ type: 'insertCode', code: code });
        }
    });

    // ── DOM Message Rendering ──

    function addMessageToDOM(msg) {
        if (welcome) { welcome.style.display = 'none'; }

        var el = document.createElement('div');
        el.className = 'message ' + msg.role;

        if (msg.role === 'status') {
            el.innerHTML = '<span class="spinner"></span> ' + escapeHtml(msg.content);
        } else if (msg.role === 'error') {
            el.textContent = msg.content;
        } else {
            var header = document.createElement('div');
            header.className = 'message-header';

            var role = document.createElement('span');
            role.className = 'message-role';
            role.textContent = msg.role === 'user' ? 'You' : 'Assistant';
            header.appendChild(role);

            if (msg.mode && msg.role === 'assistant') {
                var badge = document.createElement('span');
                badge.className = 'message-mode';
                badge.textContent = msg.mode;
                header.appendChild(badge);
            }

            el.appendChild(header);

            var body = document.createElement('div');
            body.className = 'message-body';
            if (msg.role === 'assistant') {
                body.innerHTML = renderMarkdown(msg.content);
            } else {
                body.textContent = msg.content;
            }
            el.appendChild(body);
        }

        chatArea.appendChild(el);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function showWelcome() {
        chatArea.innerHTML = '';
        if (welcome) {
            welcome.style.display = 'block';
            chatArea.appendChild(welcome);
        }
    }

    // ── Messages from Extension ──

    window.addEventListener('message', function (event) {
        var msg = event.data;
        switch (msg.type) {
            case 'addMessage':
                addMessageToDOM(msg.message);
                break;

            case 'syncMessages':
                chatArea.innerHTML = '';
                if (!msg.messages || msg.messages.length === 0) {
                    showWelcome();
                } else {
                    if (welcome) { welcome.style.display = 'none'; }
                    for (var i = 0; i < msg.messages.length; i++) {
                        addMessageToDOM(msg.messages[i]);
                    }
                }
                break;

            case 'clearMessages':
                showWelcome();
                break;

            case 'setRunning':
                isRunning = msg.running;
                sendBtn.disabled = isRunning;
                sendBtn.style.display = isRunning ? 'none' : 'flex';
                cancelBtn.classList.toggle('visible', isRunning);
                if (!isRunning) {
                    progressBar.classList.remove('visible');
                }
                break;

            case 'progress':
                progressBar.classList.add('visible');
                var line = document.createElement('div');
                line.className = 'progress-line';
                line.textContent = msg.line;
                progressBar.appendChild(line);
                while (progressBar.children.length > 8) {
                    progressBar.removeChild(progressBar.firstChild);
                }
                progressBar.scrollTop = progressBar.scrollHeight;
                break;

            case 'config':
                if (msg.config) {
                    currentConfig = msg.config;
                    modeSelect.value = msg.config.mode || 'cascade';
                    webSearchCheck.checked = msg.config.webSearch !== false;
                    maxIter.value = msg.config.maxIterations || 15;
                    updateModeUI();
                }
                break;
        }
    });

    // Request config on load
    vscode.postMessage({ type: 'getConfig' });
    updateModeUI();
    promptInput.focus();
})();
