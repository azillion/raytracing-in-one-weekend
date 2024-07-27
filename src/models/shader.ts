import { BunFile } from "bun";
import { watch } from "fs";

export enum ShaderType {
    Vertex,
    Fragment,
    Compute
}

export class Shader {
    private path: string;
    private file: BunFile;
    private shaderCode: string;
    private type: ShaderType;
    private watcher: any;
    private onChangeCallback: (() => void) | null = null;

    constructor(path: string, type: ShaderType, shouldReload: boolean = false) {
        this.path = path;
        this.file = Bun.file(path);
        this.shaderCode = "";
        this.type = type;
        this.loadShader();
        if (shouldReload) {
            this.setupWatcher();
        }
    }

    private async loadShader() {
        try {
            this.shaderCode = await this.file.text();
            console.log(`Shader ${this.path} loaded.`);
        } catch (error) {
            console.error(`Error loading shader ${this.path}:`, error);
        }
    }

    private setupWatcher() {
        this.watcher = watch(this.path);
        this.watcher.on("change", async () => {
            await this.loadShader();
            if (this.onChangeCallback) {
                this.onChangeCallback();
            }
        });
    }

    public getShaderCode(): string {
        return this.shaderCode;
    }

    public getType(): ShaderType {
        return this.type;
    }

    public onchange(callback: () => void) {
        this.onChangeCallback = callback;
    }

    public dispose() {
        this.watcher.close();
        console.log(`Shader ${this.path} disposed.`);
    }
}
