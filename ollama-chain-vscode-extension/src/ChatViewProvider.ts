import * as vscode from 'vscode';
import { OllamaChainRunner, ChainOptions } from './OllamaChainRunner';

interface ChatMessage {
    role: 'user' | 'assistant' | 'status' | 'error';
    content: string;
    mode?: string;
    timestamp: number;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'ollama-chain.chatView';

    private view?: vscode.WebviewView;
    private messages: ChatMessage[] = [];
    private isRunning = false;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly runner: OllamaChainRunner,
    ) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ): void {
        this.view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri],
        };

        webviewView.webview.html = this.getHtml(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (msg) => {
            switch (msg.type) {
                case 'sendPrompt':
                    await this.handlePrompt(msg.prompt, msg.options);
                    break;
                case 'cancel':
                    this.runner.cancel();
                    this.isRunning = false;
                    this.postMessage({ type: 'setRunning', running: false });
                    this.addMessage('status', 'Cancelled.');
                    break;
                case 'clearChat':
                    this.clearChat();
                    break;
                case 'showOutput':
                    this.runner.showOutput();
                    break;
                case 'insertCode':
                    this.insertCodeAtCursor(msg.code);
                    break;
                case 'getConfig':
                    this.sendConfig();
                    break;
            }
        });

        this.sendConfig();
        this.syncMessages();
    }

    public async sendPromptFromCommand(): Promise<void> {
        const prompt = await vscode.window.showInputBox({
            prompt: 'Enter your prompt for Ollama Chain',
            placeHolder: 'Ask anything...',
        });
        if (prompt) {
            const options = this.runner.getConfig();
            await this.handlePrompt(prompt, options);
        }
    }

    public clearChat(): void {
        this.messages = [];
        this.postMessage({ type: 'clearMessages' });
    }

    public async selectMode(): Promise<void> {
        const modes = [
            { label: 'cascade', description: 'Chain ALL models smallest→largest' },
            { label: 'auto', description: 'Router picks the best strategy' },
            { label: 'agent', description: 'Autonomous agent with planning, memory, and tools' },
            { label: 'route', description: 'Fast model classifies, routes to fast/strong' },
            { label: 'pipeline', description: 'Fast extracts + classifies, strong reasons' },
            { label: 'verify', description: 'Fast drafts, strong verifies' },
            { label: 'consensus', description: 'All models answer, strongest merges' },
            { label: 'search', description: 'Search-first, strongest synthesizes' },
            { label: 'fast', description: 'Direct to fastest model' },
            { label: 'strong', description: 'Direct to strongest model' },
        ];

        const selected = await vscode.window.showQuickPick(modes, {
            placeHolder: 'Select a chain mode',
        });

        if (selected) {
            const cfg = vscode.workspace.getConfiguration('ollamaChain');
            await cfg.update('mode', selected.label, vscode.ConfigurationTarget.Global);
            this.sendConfig();
            vscode.window.showInformationMessage(`Chain mode set to: ${selected.label}`);
        }
    }

    public async listModels(): Promise<void> {
        const models = await this.runner.listModels();
        if (models.length === 0) {
            vscode.window.showWarningMessage(
                'No Ollama models found. Is Ollama running? Try: ollama pull qwen3:14b'
            );
            return;
        }
        const items = models.map((m, i) => ({
            label: `${i + 1}. ${m}`,
            description: i === 0 ? '(fastest)' : i === models.length - 1 ? '(strongest)' : '',
        }));
        vscode.window.showQuickPick(items, {
            placeHolder: `${models.length} model(s) available — cascade order shown`,
        });
    }

    // ── Prompt handling ──

    private async handlePrompt(prompt: string, options: ChainOptions): Promise<void> {
        if (this.isRunning) {
            vscode.window.showWarningMessage('A prompt is already running. Cancel it first.');
            return;
        }

        this.isRunning = true;
        this.addMessage('user', prompt);
        this.postMessage({ type: 'setRunning', running: true });

        const cliAvailable = await this.runner.isCliAvailable(options.cliCommand);

        let result;
        if (cliAvailable) {
            this.addMessage('status', `Running in ${options.mode} mode...`);
            result = await this.runner.runChain(prompt, options, (line) => {
                this.postMessage({ type: 'progress', line });
            });
        } else {
            this.addMessage(
                'status',
                'ollama-chain CLI not found — using direct Ollama API. Install ollama-chain for full chain features.',
            );
            result = await this.runner.callOllamaDirectly(prompt, options);
        }

        this.isRunning = false;
        this.postMessage({ type: 'setRunning', running: false });

        if (result.success && result.output) {
            this.addMessage('assistant', result.output, options.mode);
        } else if (!result.success) {
            this.addMessage('error', result.stderr || 'Unknown error occurred.');
        } else {
            this.addMessage('assistant', '(empty response)');
        }
    }

    // ── Helpers ──

    private addMessage(role: ChatMessage['role'], content: string, mode?: string): void {
        const msg: ChatMessage = { role, content, mode, timestamp: Date.now() };
        this.messages.push(msg);
        this.postMessage({ type: 'addMessage', message: msg });
    }

    private syncMessages(): void {
        this.postMessage({ type: 'syncMessages', messages: this.messages });
    }

    private sendConfig(): void {
        this.postMessage({ type: 'config', config: this.runner.getConfig() });
    }

    private postMessage(msg: unknown): void {
        this.view?.webview.postMessage(msg);
    }

    private async insertCodeAtCursor(code: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor to insert code into.');
            return;
        }
        await editor.edit((editBuilder) => {
            editBuilder.insert(editor.selection.active, code);
        });
    }

    // ── HTML ──

    private getHtml(webview: vscode.Webview): string {
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.extensionUri, 'resources', 'webview.js'),
        );
        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
<style nonce="${nonce}">
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-sideBar-background);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}

.settings-panel {
    border-bottom: 1px solid var(--vscode-panel-border);
    padding: 8px 12px;
    flex-shrink: 0;
}

.settings-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
    padding: 4px 0;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--vscode-descriptionForeground);
}

.settings-toggle:hover { color: var(--vscode-foreground); }
.settings-toggle .arrow { transition: transform 0.2s; font-size: 10px; }
.settings-toggle .arrow.open { transform: rotate(90deg); }

.settings-body {
    overflow: hidden;
    max-height: 0;
    transition: max-height 0.25s ease-out;
}

.settings-body.open { max-height: 400px; }

.setting-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 8px 0;
}

.setting-row label {
    flex-shrink: 0;
    font-size: 12px;
    min-width: 80px;
    color: var(--vscode-descriptionForeground);
}

.setting-row select,
.setting-row input[type="number"] {
    flex: 1;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent);
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 12px;
    font-family: inherit;
    outline: none;
}

.setting-row select:focus,
.setting-row input:focus { border-color: var(--vscode-focusBorder); }

.checkbox-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 8px 0;
}

.checkbox-row input[type="checkbox"] { accent-color: var(--vscode-checkbox-background); }
.checkbox-row label { font-size: 12px; cursor: pointer; }

.mode-desc {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin: 2px 0 6px 0;
    font-style: italic;
}

/* ── Chat ── */

.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.message {
    padding: 8px 12px;
    border-radius: 6px;
    max-width: 100%;
    line-height: 1.5;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.message.user {
    background: var(--vscode-input-background);
    border-left: 3px solid var(--vscode-charts-blue, #3794ff);
}

.message.assistant {
    background: var(--vscode-editor-background);
    border-left: 3px solid var(--vscode-charts-green, #89d185);
}

.message.status {
    background: transparent;
    text-align: center;
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    font-style: italic;
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

.message.error {
    background: var(--vscode-inputValidation-errorBackground, rgba(255,0,0,0.1));
    border-left: 3px solid var(--vscode-errorForeground, #f44);
    color: var(--vscode-errorForeground, #f44);
    font-size: 12px;
}

.message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.message-role {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--vscode-descriptionForeground);
}

.message-mode {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
}

.message-body { font-size: 13px; }

.message-body h2, .message-body h3, .message-body h4 {
    margin: 8px 0 4px;
    font-weight: 600;
}

.message-body h2 { font-size: 15px; }
.message-body h3 { font-size: 14px; }
.message-body h4 { font-size: 13px; }

.code-block-wrap {
    margin: 8px 0;
    border-radius: 4px;
    overflow: hidden;
    background: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.2));
}

.code-lang {
    display: block;
    font-size: 10px;
    padding: 4px 10px 0;
    opacity: 0.6;
    font-family: var(--vscode-editor-font-family, monospace);
}

.message-body pre {
    background: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.2));
    padding: 8px 10px;
    overflow-x: auto;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 12px;
    line-height: 1.4;
    margin: 0;
    white-space: pre;
}

.message-body code:not(pre code) {
    background: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.2));
    padding: 1px 4px;
    border-radius: 3px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 12px;
}

.message-body ul, .message-body ol {
    padding-left: 20px;
    margin: 4px 0;
}

.message-body li { margin: 2px 0; }
.message-body strong { font-weight: 600; }

.code-actions {
    display: flex;
    gap: 4px;
    padding: 4px 8px;
    justify-content: flex-end;
    background: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.2));
    border-top: 1px solid var(--vscode-panel-border);
}

.code-action-btn {
    font-size: 11px;
    padding: 2px 8px;
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: none;
    border-radius: 3px;
    cursor: pointer;
}

.code-action-btn:hover { background: var(--vscode-button-secondaryHoverBackground); }

/* ── Progress ── */

.progress-bar {
    padding: 4px 12px;
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    display: none;
    border-top: 1px solid var(--vscode-panel-border);
    flex-shrink: 0;
    max-height: 60px;
    overflow-y: auto;
}

.progress-bar.visible { display: block; }
.progress-line { padding: 1px 0; font-family: var(--vscode-editor-font-family, monospace); }

/* ── Input ── */

.input-area {
    border-top: 1px solid var(--vscode-panel-border);
    padding: 10px 12px;
    flex-shrink: 0;
}

.input-row { display: flex; gap: 6px; }

.prompt-input {
    flex: 1;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: inherit;
    font-size: 13px;
    resize: none;
    min-height: 36px;
    max-height: 120px;
    line-height: 1.4;
    outline: none;
}

.prompt-input:focus { border-color: var(--vscode-focusBorder); }
.prompt-input::placeholder { color: var(--vscode-input-placeholderForeground); }

.send-btn, .cancel-btn {
    padding: 8px 14px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 4px;
}

.send-btn {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
}

.send-btn:hover { background: var(--vscode-button-hoverBackground); }
.send-btn:disabled { opacity: 0.5; cursor: default; }

.cancel-btn {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    display: none;
}

.cancel-btn:hover { background: var(--vscode-button-secondaryHoverBackground); }
.cancel-btn.visible { display: flex; }

.input-actions {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
}

.input-actions button {
    font-size: 11px;
    padding: 2px 8px;
    background: transparent;
    color: var(--vscode-descriptionForeground);
    border: none;
    cursor: pointer;
    border-radius: 3px;
}

.input-actions button:hover {
    color: var(--vscode-foreground);
    background: var(--vscode-toolbar-hoverBackground);
}

/* ── Spinner ── */

@keyframes spin { to { transform: rotate(360deg); } }

.spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid var(--vscode-descriptionForeground);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}

/* ── Welcome ── */

.welcome {
    text-align: center;
    padding: 30px 20px;
    color: var(--vscode-descriptionForeground);
}

.welcome h2 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--vscode-foreground);
}

.welcome p { font-size: 12px; line-height: 1.6; margin: 4px 0; }

.welcome kbd {
    background: var(--vscode-keybindingLabel-background, var(--vscode-input-background));
    border: 1px solid var(--vscode-keybindingLabel-border, var(--vscode-panel-border));
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 11px;
    font-family: inherit;
}
</style>
</head>
<body>

<div class="settings-panel">
    <div class="settings-toggle" id="settingsToggle">
        <span>Settings</span>
        <span class="arrow" id="settingsArrow">&#9654;</span>
    </div>
    <div class="settings-body" id="settingsBody">
        <div class="setting-row">
            <label for="modeSelect">Mode</label>
            <select id="modeSelect">
                <option value="cascade">cascade</option>
                <option value="auto">auto</option>
                <option value="agent">agent</option>
                <option value="route">route</option>
                <option value="pipeline">pipeline</option>
                <option value="verify">verify</option>
                <option value="consensus">consensus</option>
                <option value="search">search</option>
                <option value="fast">fast</option>
                <option value="strong">strong</option>
            </select>
        </div>
        <div class="mode-desc" id="modeDesc"></div>
        <div class="checkbox-row">
            <input type="checkbox" id="webSearchCheck" checked>
            <label for="webSearchCheck">Web search</label>
        </div>
        <div class="setting-row" id="iterationsRow" style="display:none">
            <label for="maxIter">Iterations</label>
            <input type="number" id="maxIter" min="1" max="50" value="15">
        </div>
    </div>
</div>

<div class="chat-area" id="chatArea">
    <div class="welcome" id="welcome">
        <h2>Ollama Chain</h2>
        <p>Chat with your local LLMs using chain modes.</p>
        <p>Select a mode above and type your prompt below.</p>
        <p style="margin-top:12px">
            <kbd>Enter</kbd> to send &middot; <kbd>Shift+Enter</kbd> for newline
        </p>
    </div>
</div>

<div class="progress-bar" id="progressBar"></div>

<div class="input-area">
    <div class="input-row">
        <textarea class="prompt-input" id="promptInput"
                  placeholder="Ask anything..." rows="1"></textarea>
        <button class="send-btn" id="sendBtn" title="Send">Send</button>
        <button class="cancel-btn" id="cancelBtn" title="Cancel">Stop</button>
    </div>
    <div class="input-actions">
        <button id="clearBtn" title="Clear chat history">Clear</button>
        <button id="logBtn" title="Show chain output log">Log</button>
    </div>
</div>

<script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }
}

function getNonce(): string {
    let text = '';
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return text;
}
