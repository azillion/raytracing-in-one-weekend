export enum ShaderType {
    Vertex,
    Fragment,
    VertexAndFragment,
    Compute
}

export class Shader {
    private path: string;
    private file: any;
    private shaderCode: string;
    private type: ShaderType;

    constructor(path: string, type: ShaderType) {
        this.path = path;
        this.file = Bun.file(path);
        this.shaderCode = "";
        this.type = type;
        this.loadShader();
    }

    private async loadShader() {
        try {
            this.shaderCode = await this.file.text();
            console.log(`Shader ${this.path} loaded.`);
        } catch (error) {
            console.error(`Error loading shader ${this.path}:`, error);
        }
    }

    public getShaderCode(): string {
        return this.shaderCode;
    }

    public getType(): ShaderType {
        return this.type;
    }
}
