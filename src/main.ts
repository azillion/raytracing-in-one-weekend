import "./style.css";
import triangleShader from "./shaders/test1.wgsl.ts";

async function main() {
    const navigator = window.navigator as any;
    if (!navigator.gpu) {
        throw Error("WebGPU not supported.");
    }
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
        throw Error("Couldn't request WebGPU adapter.");
    }

    const device = await adapter.requestDevice();
    if (!device) {
        throw Error("Couldn't request WebGPU device.");
    }
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu") as any;
    if (!context) {
        throw Error("WebGPU context not supported.");
    }
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat(adapter);
    context.configure({
        device,
        format: presentationFormat,
    });


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

    function render() {
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();

        const encoder = device.createCommandEncoder({ label: "our basic render pass command encoder" });
        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    render();
}

try {
    main();
} catch (e) {
    alert(e);
}
