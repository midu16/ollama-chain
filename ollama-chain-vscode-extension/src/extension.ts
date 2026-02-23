import * as vscode from 'vscode';
import { OllamaChainRunner } from './OllamaChainRunner';
import { ChatViewProvider } from './ChatViewProvider';

export function activate(context: vscode.ExtensionContext) {
    const runner = new OllamaChainRunner();
    const chatProvider = new ChatViewProvider(context.extensionUri, runner);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ChatViewProvider.viewType,
            chatProvider,
            { webviewOptions: { retainContextWhenHidden: true } },
        ),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('ollama-chain.sendPrompt', () => {
            chatProvider.sendPromptFromCommand();
        }),
        vscode.commands.registerCommand('ollama-chain.clearChat', () => {
            chatProvider.clearChat();
        }),
        vscode.commands.registerCommand('ollama-chain.selectMode', () => {
            chatProvider.selectMode();
        }),
        vscode.commands.registerCommand('ollama-chain.listModels', () => {
            chatProvider.listModels();
        }),
        vscode.commands.registerCommand('ollama-chain.insertCodeBlock', () => {
            vscode.window.showInformationMessage(
                'Use the "Insert" button on a code block in the Ollama Chain chat.'
            );
        }),
    );

    context.subscriptions.push({ dispose: () => runner.dispose() });
}

export function deactivate() {}
