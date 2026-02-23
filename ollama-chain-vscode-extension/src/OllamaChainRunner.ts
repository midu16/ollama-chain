import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as http from 'http';

export interface ChainOptions {
    mode: string;
    webSearch: boolean;
    maxIterations: number;
    ollamaUrl: string;
    cliCommand: string;
    cliTimeout: number;
}

export interface RunResult {
    output: string;
    stderr: string;
    success: boolean;
}

export class OllamaChainRunner {
    private outputChannel: vscode.OutputChannel;
    private activeProcess: ChildProcess | null = null;

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('Ollama Chain');
    }

    getConfig(): ChainOptions {
        const cfg = vscode.workspace.getConfiguration('ollamaChain');
        return {
            mode: cfg.get<string>('mode', 'cascade'),
            webSearch: cfg.get<boolean>('webSearch', true),
            maxIterations: cfg.get<number>('maxIterations', 15),
            ollamaUrl: cfg.get<string>('ollamaUrl', 'http://localhost:11434'),
            cliCommand: cfg.get<string>('cliCommand', 'ollama-chain'),
            cliTimeout: cfg.get<number>('cliTimeout', 300),
        };
    }

    cancel(): void {
        if (this.activeProcess) {
            this.activeProcess.kill('SIGTERM');
            this.activeProcess = null;
        }
    }

    async runChain(
        prompt: string,
        options: ChainOptions,
        onProgress?: (line: string) => void,
    ): Promise<RunResult> {
        const args: string[] = [];

        args.push('-m', options.mode);

        if (!options.webSearch) {
            args.push('--no-search');
        }

        if (options.mode === 'agent') {
            args.push('--max-iterations', String(options.maxIterations));
        }

        args.push(prompt);

        this.outputChannel.appendLine(`\n${'='.repeat(60)}`);
        this.outputChannel.appendLine(
            `[${new Date().toLocaleTimeString()}] Running: ${options.cliCommand} -m ${options.mode}`
        );
        this.outputChannel.appendLine(`Prompt: ${prompt.substring(0, 100)}...`);
        this.outputChannel.appendLine('='.repeat(60));

        return new Promise<RunResult>((resolve) => {
            let stdout = '';
            let stderr = '';

            const proc = spawn(options.cliCommand, args, {
                shell: true,
                env: { ...process.env },
                timeout: options.cliTimeout * 1000,
            });
            this.activeProcess = proc;

            proc.stdout?.on('data', (data: Buffer) => {
                const text = data.toString();
                stdout += text;
            });

            proc.stderr?.on('data', (data: Buffer) => {
                const text = data.toString();
                stderr += text;
                this.outputChannel.append(text);
                if (onProgress) {
                    for (const line of text.split('\n').filter((l: string) => l.trim())) {
                        onProgress(line.trim());
                    }
                }
            });

            proc.on('close', (code) => {
                this.activeProcess = null;
                resolve({
                    output: stdout.trim(),
                    stderr: stderr.trim(),
                    success: code === 0,
                });
            });

            proc.on('error', (err) => {
                this.activeProcess = null;
                resolve({
                    output: '',
                    stderr: `Failed to run ollama-chain: ${err.message}`,
                    success: false,
                });
            });
        });
    }

    async callOllamaDirectly(
        prompt: string,
        options: ChainOptions,
    ): Promise<RunResult> {
        const url = new URL('/api/chat', options.ollamaUrl);

        const models = await this.listModels(options.ollamaUrl);
        if (models.length === 0) {
            return {
                output: '',
                stderr: 'No models found. Pull a model first: ollama pull qwen3:14b',
                success: false,
            };
        }

        const model = models[0];
        this.outputChannel.appendLine(`[direct] Using model: ${model}`);

        const body = JSON.stringify({
            model,
            messages: [{ role: 'user', content: prompt }],
            stream: false,
        });

        return new Promise<RunResult>((resolve) => {
            const req = http.request(
                url,
                { method: 'POST', headers: { 'Content-Type': 'application/json' } },
                (res) => {
                    let data = '';
                    res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
                    res.on('end', () => {
                        try {
                            const json = JSON.parse(data);
                            const content = json?.message?.content ?? '';
                            resolve({ output: content, stderr: '', success: true });
                        } catch {
                            resolve({ output: data, stderr: '', success: true });
                        }
                    });
                },
            );
            req.on('error', (err) => {
                resolve({
                    output: '',
                    stderr: `Cannot reach Ollama at ${options.ollamaUrl}: ${err.message}`,
                    success: false,
                });
            });
            req.setTimeout(options.cliTimeout * 1000, () => {
                req.destroy();
                resolve({ output: '', stderr: 'Request timed out', success: false });
            });
            req.write(body);
            req.end();
        });
    }

    async listModels(ollamaUrl?: string): Promise<string[]> {
        const url = ollamaUrl ?? this.getConfig().ollamaUrl;

        return new Promise<string[]>((resolve) => {
            http.get(`${url}/api/tags`, (res) => {
                let data = '';
                res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        const models = (json.models ?? []).map(
                            (m: { name?: string; model?: string }) => m.model ?? m.name ?? 'unknown'
                        );
                        resolve(models);
                    } catch {
                        resolve([]);
                    }
                });
            }).on('error', () => resolve([]));
        });
    }

    async isCliAvailable(cliCommand?: string): Promise<boolean> {
        const cmd = cliCommand ?? this.getConfig().cliCommand;
        return new Promise<boolean>((resolve) => {
            const proc = spawn(cmd, ['--help'], { shell: true, timeout: 5000 });
            proc.on('close', (code) => resolve(code === 0));
            proc.on('error', () => resolve(false));
        });
    }

    showOutput(): void {
        this.outputChannel.show(true);
    }

    dispose(): void {
        this.cancel();
        this.outputChannel.dispose();
    }
}
