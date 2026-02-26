import * as vscode from 'vscode';
import * as http from 'http';

export interface ChainOptions {
    mode: string;
    webSearch: boolean;
    maxIterations: number;
    apiUrl: string;
    timeout: number;
}

export interface RunResult {
    output: string;
    stderr: string;
    success: boolean;
}

const SSE_MAX_RECONNECTS = 3;
const SSE_RECONNECT_BASE_MS = 2000;
const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 10800; // ~6 hours at 2s intervals — no artificial ceiling
const SSE_IDLE_TIMEOUT_MS = 660_000; // 11 minutes without any SSE data → reconnect (must exceed server-side agent idle limit of 600s)

export class OllamaChainRunner {
    private outputChannel: vscode.OutputChannel;
    private _currentJobId: string | null = null;
    private sseAbort: { abort: () => void } | null = null;

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('Ollama Chain');
    }

    get currentJobId(): string | null {
        return this._currentJobId;
    }

    getConfig(): ChainOptions {
        const cfg = vscode.workspace.getConfiguration('ollamaChain');
        return {
            mode: cfg.get<string>('mode', 'cascade'),
            webSearch: cfg.get<boolean>('webSearch', true),
            maxIterations: cfg.get<number>('maxIterations', 15),
            apiUrl: cfg.get<string>('apiUrl', 'http://localhost:8585'),
            timeout: cfg.get<number>('timeout', 600),
        };
    }

    cancel(): void {
        if (this._currentJobId) {
            const cfg = this.getConfig();
            this.cancelViaApi(cfg.apiUrl, this._currentJobId);
            this._currentJobId = null;
        }
        if (this.sseAbort) {
            this.sseAbort.abort();
            this.sseAbort = null;
        }
    }

    // ── API server ─────────────────────────────────────────────────────

    async isApiAvailable(apiUrl?: string): Promise<boolean> {
        const base = (apiUrl ?? this.getConfig().apiUrl).replace(/\/+$/, '');
        return new Promise<boolean>((resolve) => {
            const req = http.get(`${base}/api/health`, (res) => {
                resolve(res.statusCode === 200);
                res.resume();
            });
            req.on('error', () => resolve(false));
            req.setTimeout(3000, () => { req.destroy(); resolve(false); });
        });
    }

    async runViaApi(
        prompt: string,
        options: ChainOptions,
        onProgress?: (line: string) => void,
    ): Promise<RunResult> {
        const base = options.apiUrl.replace(/\/+$/, '');

        const submitResult = await this.httpPost(`${base}/api/prompt`, {
            prompt,
            mode: options.mode,
            web_search: options.webSearch,
            max_iterations: options.maxIterations,
            timeout: options.timeout,
        });

        if (!submitResult.success) {
            return {
                output: '',
                stderr: submitResult.error || 'Failed to submit prompt to API',
                success: false,
            };
        }

        const jobId: string = submitResult.data.job_id;
        this._currentJobId = jobId;
        this.outputChannel.appendLine(`[api] Job submitted: ${jobId}`);

        if (submitResult.data.position > 0 && onProgress) {
            onProgress(`Queued at position ${submitResult.data.position + 1}`);
        }

        return this.streamSSE(base, jobId, options.timeout, onProgress);
    }

    /**
     * Reconnect to an existing API job that may still be running.
     * Used to recover a session after webview re-creation.
     */
    async recoverJob(
        jobId: string,
        options: ChainOptions,
        onProgress?: (line: string) => void,
    ): Promise<RunResult> {
        this._currentJobId = jobId;
        const base = options.apiUrl.replace(/\/+$/, '');

        const status = await this.getJobStatus(base, jobId);
        if (!status) {
            this._currentJobId = null;
            return { output: '', stderr: 'Job not found', success: false };
        }

        if (status.status === 'completed') {
            this._currentJobId = null;
            return { output: status.result || '', stderr: '', success: true };
        }
        if (status.status === 'timed_out') {
            this._currentJobId = null;
            const partial = status.result || '';
            const err = status.error || 'Job timed out';
            if (partial) {
                return { output: partial, stderr: err, success: true };
            }
            return { output: '', stderr: err, success: false };
        }
        if (status.status === 'failed') {
            this._currentJobId = null;
            return { output: '', stderr: status.error || 'Chain failed', success: false };
        }
        if (status.status === 'cancelled') {
            this._currentJobId = null;
            return { output: '', stderr: 'Cancelled', success: false };
        }

        if (onProgress) {
            onProgress('[session] Reconnecting to active job...');
        }

        return this.streamSSE(base, jobId, options.timeout, onProgress);
    }

    // ── Utilities ──────────────────────────────────────────────────────

    async listModels(): Promise<string[]> {
        const cfg = this.getConfig();
        const base = cfg.apiUrl.replace(/\/+$/, '');

        return new Promise<string[]>((resolve) => {
            const req = http.get(`${base}/api/models`, (res) => {
                let data = '';
                res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        resolve(json.models ?? []);
                    } catch {
                        resolve([]);
                    }
                });
            });
            req.on('error', () => resolve([]));
            req.setTimeout(5000, () => { req.destroy(); resolve([]); });
        });
    }

    async extendTimeout(extraSeconds: number = 300): Promise<{ extended_by: number; remaining: number; timeout: number } | null> {
        if (!this._currentJobId) { return null; }
        const cfg = this.getConfig();
        const base = cfg.apiUrl.replace(/\/+$/, '');
        const result = await this.httpPatch(
            `${base}/api/prompt/${this._currentJobId}/timeout`,
            { extend_by: extraSeconds },
        );
        if (result.success && result.data) {
            this.outputChannel.appendLine(
                `[api] Timeout extended by ${result.data.extended_by}s — ${result.data.remaining}s remaining`,
            );
            return result.data;
        }
        return null;
    }

    showOutput(): void {
        this.outputChannel.show(true);
    }

    dispose(): void {
        this.cancel();
        this.outputChannel.dispose();
    }

    // ── Private helpers ────────────────────────────────────────────────

    private httpRequest(
        url: string,
        method: string,
        data: Record<string, unknown>,
    ): Promise<{ success: boolean; data?: any; error?: string }> {
        return new Promise((resolve) => {
            const body = JSON.stringify(data);
            const parsed = new URL(url);

            const req = http.request(
                {
                    hostname: parsed.hostname,
                    port: parsed.port || undefined,
                    path: parsed.pathname,
                    method,
                    headers: {
                        'Content-Type': 'application/json',
                        'Content-Length': Buffer.byteLength(body),
                    },
                },
                (res) => {
                    let responseData = '';
                    res.on('data', (chunk: Buffer) => { responseData += chunk.toString(); });
                    res.on('end', () => {
                        try {
                            const json = JSON.parse(responseData);
                            resolve({
                                success: res.statusCode === 200 || res.statusCode === 202,
                                data: json,
                            });
                        } catch {
                            resolve({ success: false, error: `Invalid response: ${responseData}` });
                        }
                    });
                },
            );
            req.on('error', (err) => {
                resolve({ success: false, error: err.message });
            });
            req.setTimeout(10000, () => {
                req.destroy();
                resolve({ success: false, error: 'Connection timeout' });
            });
            req.write(body);
            req.end();
        });
    }

    private httpPost(
        url: string,
        data: Record<string, unknown>,
    ): Promise<{ success: boolean; data?: any; error?: string }> {
        return this.httpRequest(url, 'POST', data);
    }

    private httpPatch(
        url: string,
        data: Record<string, unknown>,
    ): Promise<{ success: boolean; data?: any; error?: string }> {
        return this.httpRequest(url, 'PATCH', data);
    }

    private settleFromJobStatus(status: any): RunResult {
        switch (status.status) {
            case 'completed':
                return { output: status.result || '', stderr: '', success: true };
            case 'timed_out': {
                const partial = status.result || '';
                const err = status.error || 'Job timed out';
                return partial
                    ? { output: partial, stderr: err, success: true }
                    : { output: '', stderr: err, success: false };
            }
            case 'failed':
                return { output: '', stderr: status.error || 'Chain execution failed', success: false };
            case 'cancelled':
                return { output: '', stderr: 'Cancelled', success: false };
            default:
                return { output: '', stderr: `Unexpected status: ${status.status}`, success: false };
        }
    }

    private streamSSE(
        apiUrl: string,
        jobId: string,
        timeout: number,
        onProgress?: (line: string) => void,
    ): Promise<RunResult> {
        return new Promise((resolve) => {
            let output = '';
            let errorMsg = '';
            let settled = false;
            let reconnectAttempts = 0;
            let totalAttempts = 0;
            const MAX_TOTAL_ATTEMPTS = 10;

            const settle = (result: RunResult) => {
                if (!settled) {
                    settled = true;
                    this._currentJobId = null;
                    this.sseAbort = null;
                    resolve(result);
                }
            };

            const processSSEParts = (raw: string) => {
                const parts = raw.split('\n\n');
                const remainder = parts.pop() || '';
                for (const part of parts) {
                    if (!part.trim()) { continue; }
                    if (part.startsWith(':')) { continue; }

                    let eventType = '';
                    let data = '';
                    for (const line of part.split('\n')) {
                        if (line.startsWith('event: ')) { eventType = line.slice(7); }
                        else if (line.startsWith('data: ')) { data = line.slice(6); }
                    }
                    if (!eventType || !data) { continue; }

                    try {
                        const payload = JSON.parse(data);
                        switch (eventType) {
                            case 'progress':
                                if (onProgress && payload.line) {
                                    onProgress(payload.line);
                                    this.outputChannel.appendLine(payload.line);
                                }
                                break;
                            case 'queued':
                                if (onProgress) {
                                    onProgress(`Position in queue: ${(payload.position ?? 0) + 1}`);
                                }
                                break;
                            case 'complete':
                                output = payload.result || '';
                                return { terminal: true, remainder };
                            case 'timed_out':
                                if (payload.partial_result) {
                                    output = payload.partial_result;
                                    errorMsg = payload.error || 'Job timed out (partial result available)';
                                } else {
                                    errorMsg = payload.error || 'Job timed out';
                                }
                                return { terminal: true, remainder };
                            case 'error':
                                errorMsg = payload.error || 'Unknown error';
                                return { terminal: true, remainder };
                            case 'cancelled':
                                errorMsg = 'Cancelled';
                                return { terminal: true, remainder };
                        }
                    } catch {
                        // skip malformed SSE events
                    }
                }
                return { terminal: false, remainder };
            };

            const settleFromAccumulated = () => {
                if (output && !errorMsg) {
                    settle({ output, stderr: '', success: true });
                } else if (output && errorMsg) {
                    settle({ output, stderr: errorMsg, success: true });
                } else {
                    settle({ output: '', stderr: errorMsg || 'Unknown error', success: false });
                }
            };

            const tryConnect = () => {
                if (settled) { return; }

                totalAttempts++;
                if (totalAttempts > MAX_TOTAL_ATTEMPTS) {
                    this.outputChannel.appendLine(
                        `[session] Max total attempts (${MAX_TOTAL_ATTEMPTS}) exceeded, polling...`,
                    );
                    this.pollJobResult(apiUrl, jobId, onProgress).then(settle);
                    return;
                }

                const parsed = new URL(`${apiUrl}/api/prompt/${jobId}/stream`);
                let buffer = '';
                let gotTerminalEvent = false;
                let idleTimer: ReturnType<typeof setTimeout> | null = null;

                const resetIdleTimer = (req: http.ClientRequest) => {
                    if (idleTimer) { clearTimeout(idleTimer); }
                    idleTimer = setTimeout(() => {
                        this.outputChannel.appendLine('[session] SSE idle timeout, reconnecting...');
                        req.destroy();
                        if (!settled) {
                            scheduleReconnectOrPoll();
                        }
                    }, SSE_IDLE_TIMEOUT_MS);
                };

                const clearIdle = () => {
                    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
                };

                const req = http.get(
                    {
                        hostname: parsed.hostname,
                        port: parsed.port || undefined,
                        path: parsed.pathname,
                    },
                    (res) => {
                        const contentType = res.headers['content-type'] || '';

                        if (contentType.includes('application/json')) {
                            let jsonBuf = '';
                            res.on('data', (chunk: Buffer) => { jsonBuf += chunk.toString(); });
                            res.on('end', () => {
                                clearIdle();
                                try {
                                    const job = JSON.parse(jsonBuf);
                                    this.outputChannel.appendLine(
                                        `[session] Job ${jobId} already ${job.status}, received via JSON`,
                                    );
                                    settle(this.settleFromJobStatus(job));
                                } catch {
                                    if (!settled) { scheduleReconnectOrPoll(); }
                                }
                            });
                            return;
                        }

                        resetIdleTimer(req);

                        res.on('data', (chunk: Buffer) => {
                            resetIdleTimer(req);
                            buffer += chunk.toString();
                            const result = processSSEParts(buffer);
                            buffer = result.remainder;
                            if (result.terminal) {
                                gotTerminalEvent = true;
                            }
                        });

                        res.on('end', () => {
                            clearIdle();
                            if (!gotTerminalEvent && buffer.trim()) {
                                const result = processSSEParts(buffer + '\n\n');
                                if (result.terminal) {
                                    gotTerminalEvent = true;
                                }
                            }
                            if (gotTerminalEvent) {
                                settleFromAccumulated();
                            } else if (!settled) {
                                scheduleReconnectOrPoll();
                            }
                        });
                    },
                );

                this.sseAbort = {
                    abort: () => {
                        clearIdle();
                        req.destroy();
                        settle({ output: '', stderr: 'Cancelled', success: false });
                    },
                };

                req.on('error', () => {
                    clearIdle();
                    if (!settled) {
                        scheduleReconnectOrPoll();
                    }
                });
            };

            const scheduleReconnectOrPoll = () => {
                if (settled) { return; }

                this.getJobStatus(apiUrl, jobId).then((status) => {
                    if (settled) { return; }

                    if (status && status.status !== 'queued' && status.status !== 'running') {
                        this.outputChannel.appendLine(
                            `[session] Job ${jobId} is ${status.status}, settling from poll`,
                        );
                        settle(this.settleFromJobStatus(status));
                        return;
                    }

                    if (reconnectAttempts < SSE_MAX_RECONNECTS) {
                        reconnectAttempts++;
                        const delay = SSE_RECONNECT_BASE_MS * reconnectAttempts;
                        if (onProgress) {
                            onProgress(`[session] Connection lost, reconnecting (${reconnectAttempts}/${SSE_MAX_RECONNECTS})...`);
                        }
                        setTimeout(tryConnect, delay);
                    } else {
                        if (onProgress) {
                            onProgress('[session] SSE reconnection exhausted, polling for result...');
                        }
                        this.pollJobResult(apiUrl, jobId, onProgress).then(settle);
                    }
                }).catch(() => {
                    if (settled) { return; }
                    if (reconnectAttempts < SSE_MAX_RECONNECTS) {
                        reconnectAttempts++;
                        setTimeout(tryConnect, SSE_RECONNECT_BASE_MS * reconnectAttempts);
                    } else {
                        this.pollJobResult(apiUrl, jobId, onProgress).then(settle);
                    }
                });
            };

            tryConnect();
        });
    }

    private async pollJobResult(
        apiUrl: string,
        jobId: string,
        onProgress?: (line: string) => void,
    ): Promise<RunResult> {
        let lastProgressIdx = 0;

        for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
            const status = await this.getJobStatus(apiUrl, jobId);
            if (!status) {
                return { output: '', stderr: 'Job not found during polling', success: false };
            }

            if (onProgress && status.progress) {
                while (lastProgressIdx < status.progress.length) {
                    onProgress(status.progress[lastProgressIdx]);
                    lastProgressIdx++;
                }
            }

            if (status.status === 'completed') {
                return { output: status.result || '', stderr: '', success: true };
            }
            if (status.status === 'timed_out') {
                const partial = status.result || '';
                const err = status.error || 'Job timed out';
                if (partial) {
                    return { output: partial, stderr: err, success: true };
                }
                return { output: '', stderr: err, success: false };
            }
            if (status.status === 'failed') {
                return { output: '', stderr: status.error || 'Chain execution failed', success: false };
            }
            if (status.status === 'cancelled') {
                return { output: '', stderr: 'Cancelled', success: false };
            }

            await new Promise<void>(r => setTimeout(r, POLL_INTERVAL_MS));
        }

        return { output: '', stderr: 'Polling timed out', success: false };
    }

    private getJobStatus(
        apiUrl: string,
        jobId: string,
    ): Promise<any | null> {
        return new Promise((resolve) => {
            const parsed = new URL(`${apiUrl}/api/prompt/${jobId}`);
            const req = http.get(
                {
                    hostname: parsed.hostname,
                    port: parsed.port || undefined,
                    path: parsed.pathname,
                },
                (res) => {
                    let data = '';
                    res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
                    res.on('end', () => {
                        try {
                            resolve(JSON.parse(data));
                        } catch {
                            resolve(null);
                        }
                    });
                },
            );
            req.on('error', () => resolve(null));
            req.setTimeout(5000, () => { req.destroy(); resolve(null); });
        });
    }

    private cancelViaApi(apiUrl: string, jobId: string): void {
        const base = apiUrl.replace(/\/+$/, '');
        const parsed = new URL(`${base}/api/prompt/${jobId}`);
        const req = http.request(
            {
                hostname: parsed.hostname,
                port: parsed.port || undefined,
                path: parsed.pathname,
                method: 'DELETE',
            },
            () => { /* fire and forget */ },
        );
        req.on('error', () => { /* ignore */ });
        req.end();
    }
}
