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
    const presentationFormat = navigator.gpu?.getPreferredFormat(adapter);
    context.configure({
        device,
        format: presentationFormat,
    });
}

try {
    main();
} catch (e) {
    alert(e);
}
