import "./style.css";

export async function initWebGPU(canvas: HTMLCanvasElement) {
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

const constants = `
const PI: f32 = 3.1415926535897932385;
const INFINITY: f32 = 1e38;
const SEED: vec2<f32> = vec2<f32>(69.69, 4.20);
`;

const helpers = `
fn lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a * (1.0 - t) + b * t;
}

fn degreesToRadians(degrees: f32) -> f32 {
    return degrees * PI / 180.0;
}

fn rand(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn randMinMax(seed: vec2<f32>, min: f32, max: f32) -> f32 {
    return min + (max - min) * rand(seed);
}

fn randVec3(seed: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(rand(seed), rand(seed + 1.0), rand(seed + 2.0));
}
`;

const intervalShader = `
struct Interval {
    minI: f32,
    maxI: f32,
}

fn createInterval(min: f32, max: f32) -> Interval {
    return Interval(min, max);
}

fn intervalSize(i: Interval) -> f32 {
    return i.maxI - i.minI;
}

fn intervalContains(i: Interval, x: f32) -> bool {
    return i.minI <= x && x <= i.maxI;
}

fn intervalSurrounds(i: Interval, x: f32) -> bool {
    return i.minI < x && x < i.maxI;
}

fn clampInterval(i: Interval, minI: f32, maxI: f32) -> f32 {
    return min(max(i.minI, minI), maxI);
}

const INTERVAL_EMPTY: Interval = Interval(INFINITY, -INFINITY);
const INTERVAL_UNIVERSE: Interval = Interval(-INFINITY, INFINITY);
`;

const hittableShapesShader = `
struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    hit: bool,
    front_face: bool,
}

struct FaceNormalRecord {
    front_face: bool,
    normal: vec3<f32>,
}

fn setFaceNormal(rec: HitRecord, r: Ray, outwardNormal: vec3<f32>) -> FaceNormalRecord {
    var newRec: FaceNormalRecord;
    let frontFaceDirections = dot(r.direction, outwardNormal);
    if (frontFaceDirections < 0.0) {
        newRec.front_face = true;
        newRec.normal = outwardNormal;
    } else {
        newRec.front_face = false;
        newRec.normal = -outwardNormal;
    }
    return newRec;
}

fn hit_sphere(sphere: Sphere, r: Ray, ray_t: Interval) -> HitRecord {
    var rec: HitRecord;
    rec.hit = false;

    let oc = sphere.center - r.origin;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;

    let discriminant = h * h - a * c;
    if (discriminant < 0.0) {
        return rec;
    }

    let sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    var root = (h - sqrtd) / a;
    if (!intervalContains(ray_t, root)) {
        root = (h + sqrtd) / a;
        if (!intervalContains(ray_t, root)) {
            return rec;
        }
    }

    rec.t = root;
    rec.p = rayAt(r, rec.t);
    rec.normal = (rec.p - sphere.center) / sphere.radius;
    let faceNormalRec = setFaceNormal(rec, r, rec.normal);
    rec.front_face = faceNormalRec.front_face;
    rec.normal = faceNormalRec.normal;
    rec.hit = true;

    return rec;
}

fn hit_spheres(r: Ray, ray_t: Interval) -> HitRecord {
    var closest_so_far = ray_t.maxI;
    var rec: HitRecord;
    rec.hit = false;

    for (var i = 0u; i < 2u; i++) { 
        let sphere_rec = hit_sphere(spheres[i], r, createInterval(ray_t.minI, closest_so_far));
        if (sphere_rec.hit) {
            closest_so_far = sphere_rec.t;
            rec = sphere_rec;
        }
    }

    return rec;
}
`;

const cameraShader = `
struct Camera {
    origin: vec3<f32>,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    samples_per_pixel: u32,
}

fn createCamera(aspect_ratio: f32) -> Camera {
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    let samples_per_pixel: u32 = 100;

    let origin = vec3<f32>(0.0, 0.0, 0.0);
    let horizontal = vec3<f32>(viewport_width, 0.0, 0.0);
    let vertical = vec3<f32>(0.0, viewport_height, 0.0);
    let lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - vec3<f32>(0.0, 0.0, focal_length);

    return Camera(origin, lower_left_corner, horizontal, vertical, samples_per_pixel);
}

fn getRay(camera: Camera, u: f32, v: f32) -> Ray {
    return Ray(
        camera.origin,
        camera.lower_left_corner + u*camera.horizontal + v*camera.vertical - camera.origin
    );
}
`;

function createComputeShader(device: GPUDevice, textureSize: { width: number, height: number }) {
    const module = device.createShaderModule({
        label: "Compute shader",
        code: `
            ${constants}
            ${helpers}
            ${intervalShader}
            ${hittableShapesShader}
            ${cameraShader}
            struct Ray {
                origin: vec3<f32>,
                direction: vec3<f32>
            }

            const spheres = array<Sphere, 2>(
                Sphere(vec3<f32>(0.0, 0.0, -1.0), 0.5),
                Sphere(vec3<f32>(0.0, -100.5, -1.0), 100.0)
            );

            fn createRay(uv: vec2<f32>) -> Ray {
                let aspectRatio = f32(textureDimensions(output).x) / f32(textureDimensions(output).y);
                let origin = vec3<f32>(0.0, 0.0, 0.0);
                let direction = vec3<f32>(uv.x * aspectRatio, uv.y, -1.0);
                return Ray(origin, normalize(direction));
            }

            fn rayColor(ray: Ray) -> vec3<f32> {
                let rec = hit_spheres(ray, createInterval(0.001, INFINITY));
                if (rec.hit) {
                    return 0.5 * (rec.normal + vec3<f32>(1.0, 1.0, 1.0));
                }
                let unit_direction = normalize(ray.direction);
                let a = 0.5 * (unit_direction.y + 1.0);
                return lerp(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), a);
            }

            fn rayAt(ray: Ray, t: f32) -> vec3<f32> {
                return ray.origin + ray.direction * t;
            }

            @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(output);
                let coords = vec2<u32>(id.xy);
                
                if (coords.x >= dims.x || coords.y >= dims.y) {
                    return;
                }

                let aspect_ratio = f32(dims.x) / f32(dims.y);
                let camera = createCamera(aspect_ratio);

                var pixel_color = vec3<f32>(0.0, 0.0, 0.0);
                for (var s = 0u; s < camera.samples_per_pixel; s++) {
                    let u = (f32(coords.x) + rand(SEED + vec2<f32>(f32(s), 0.0))) / f32(dims.x);
                    let v = (f32(coords.y) + rand(SEED + vec2<f32>(0.0, f32(s)))) / f32(dims.y);
                    let ray = getRay(camera, u, v);
                    pixel_color += rayColor(ray);
                }
                
                pixel_color = pixel_color / f32(camera.samples_per_pixel); 
                
                textureStore(output, vec2<i32>(coords), vec4<f32>(pixel_color, 1.0));
            }
        `
    });

    const pipeline = device.createComputePipeline({
        label: "Compute pipeline",
        layout: "auto",
        compute: {
            module,
            entryPoint: "main"
        }
    });

    const texture = device.createTexture({
        size: textureSize,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: texture.createView() }
        ]
    });

    return { pipeline, bindGroup, texture };
}

function createRenderPipeline(device: GPUDevice, format: GPUTextureFormat) {
    const module = device.createShaderModule({
        label: "Render shader",
        code: `
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

           @vertex
           fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
               var pos = array<vec2<f32>, 4>(
                   vec2<f32>(-1.0, -1.0),
                   vec2<f32>(1.0, -1.0),
                   vec2<f32>(-1.0, 1.0),
                   vec2<f32>(1.0, 1.0)
               );
               var uv = array<vec2<f32>, 4>(
                   vec2<f32>(0.0, 0.0),  // Changed from (0.0, 1.0)
                   vec2<f32>(1.0, 0.0),  // Changed from (1.0, 1.0)
                   vec2<f32>(0.0, 1.0),  // Changed from (0.0, 0.0)
                   vec2<f32>(1.0, 1.0)   // Changed from (1.0, 0.0)
               );
               var output: VertexOutput;
               output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
               output.uv = uv[vertexIndex];
               return output;
           } 

            @group(0) @binding(0) var textureSampler: sampler;
            @group(0) @binding(1) var inputTexture: texture_2d<f32>;

            @fragment
            fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(inputTexture, textureSampler, uv);
            }
        `
    });

    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vertexMain"
        },
        fragment: {
            module,
            entryPoint: "fragmentMain",
            targets: [{ format }]
        },
        primitive: {
            topology: "triangle-strip",
            stripIndexFormat: "uint32"
        }
    });
}

async function main() {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    const { device, context, presentationFormat } = await initWebGPU(canvas);

    // for now we will hardcode the canvas size
    const width = 400;
    const aspectRation = 16.0 / 9.0;
    const height = Math.floor(width / aspectRation);
    canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
    canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
    let textureSize = { width: canvas.width, height: canvas.height };

    let computeShader = createComputeShader(device, textureSize);
    let renderPipeline = createRenderPipeline(device, presentationFormat);

    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
    });

    let renderBindGroup: GPUBindGroup;

    function updateBindGroups() {
        renderBindGroup = device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: sampler },
                { binding: 1, resource: computeShader.texture.createView() }
            ]
        });
    }

    updateBindGroups();

    function render() {
        const commandEncoder = device.createCommandEncoder();

        // Run compute shader
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computeShader.pipeline);
        computePass.setBindGroup(0, computeShader.bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(textureSize.width / 8), Math.ceil(textureSize.height / 8));
        computePass.end();

        // Render to canvas
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(4);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const renderTime = performance.now();
            const canvas = entry.target as HTMLCanvasElement;
            canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
            canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));

            // Update context configuration
            context.configure({
                device,
                format: presentationFormat,
                size: [canvas.width, canvas.height]
            });

            // Recreate compute shader with new size
            textureSize = { width: canvas.width, height: canvas.height };
            computeShader = createComputeShader(device, textureSize);

            // Update bind groups
            updateBindGroups();

            // Re-render
            render();
            console.log("Rerender time:", performance.now() - renderTime);
        }
    });

    observer.observe(canvas);

    // Initial render
    render();
}

let renderTime = performance.now();
main().then(() =>
    console.log("Total time:", performance.now() - renderTime)
).catch(e => {
    console.error(e);
    alert(e);
});
