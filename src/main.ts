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
    console.log(device);
}

try {
    main();
} catch (e) {
    alert(e);
}
