import "./style.css";
import triangleShader from "./shaders/test1.wgsl.ts";

export async function initWebGPU(canvas: HTMLCanvasElement) {
    const navigator: any = window.navigator;
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu") as any;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device,
        format: presentationFormat,
    });

    return { device, context, presentationFormat };
}

function createRenderTriangle(device: any, context: any, presentationFormat: any): () => void {
    const module = device.createShaderModule({
        label: "shader",
        code: triangleShader,
    });

    const pipeline = device.createRenderPipeline({
        label: "hard coded red triangle pipeline",
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs",
        },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: presentationFormat }],
        },
    });

    const renderPassDescriptor = {
        label: "our basic canvas render pass",
        colorAttachments: [
            {
                view: undefined,
                clearValue: [0.3, 0.3, 0.3, 1.0],
                loadOp: "clear",
                storeOp: "store",
            },
        ],
    };

    return () => {
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();

        const encoder = device.createCommandEncoder({ label: "our basic render pass command encoder" });
        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }
}

function createComputeShader(device: any): () => void {
    const shader = `
        @group(0) @binding(0) var<storage, read_write> data : array<f32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3u) {
           let i = id.x;
           data[i] = data[i] * 2.0;
        }
        `;
    const module = device.createShaderModule({
        label: "double data compute shader",
        code: shader,
    });

    const pipleline = device.createComputePipeline({
        label: "double data compute pipeline",
        layout: "auto",
        compute: {
            module,
            entryPoint: "main",
        },
    });

    const input = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const buffer = device.createBuffer({
        label: "data buffer",
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // SLOW: Copy the data to the GPU
    device.queue.writeBuffer(buffer, 0, input);

    const resultBuffer = device.createBuffer({
        label: "result buffer",
        size: input.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = device.createBindGroup({
        label: "data bind group",
        layout: pipleline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer,
                },
            },
        ],
    });

    return async () => {
        const encoder = device.createCommandEncoder({ label: "compute command encoder" });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipleline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(input.length);
        pass.end();

        encoder.copyBufferToBuffer(buffer, 0, resultBuffer, 0, resultBuffer.size);
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        // SLOW: Copy the result back to the CPU
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());
        console.log("input:", input);
        console.log("Result:", result.map(f => f));
        resultBuffer.unmap();
    }
}

async function main() {
    const { device, context, presentationFormat } = await initWebGPU(document.getElementById("canvas") as HTMLCanvasElement);

    const renderTriangle = createRenderTriangle(device, context, presentationFormat);
    const renderCompute = createComputeShader(device);

    renderTriangle();
    renderCompute();
}

main().catch(e => {
    console.error(e);
    alert(e);
});
