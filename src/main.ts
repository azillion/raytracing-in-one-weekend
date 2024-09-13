import "./style.css";
let renderTime = performance.now();

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

const NUM_SPHERES = 3;

const constants = `
const PI: f32 = 3.1415926535897932385;
const INFINITY: f32 = 1e38;
const SEED: vec2<f32> = vec2<f32>(69.68, 4.20);
const MAX_DEPTH: u32 = 100;
const NUM_SPHERES: u32 = ${NUM_SPHERES};
`;

const helpers = `
fn lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a * (1.0 - t) + b * t;
}

fn degreesToRadians(degrees: f32) -> f32 {
    return degrees * PI / 180.0;
}

fn hash(seed: vec2<u32>) -> u32 {
    var state = seed.x;
    state = state ^ (state << 13u);
    state = state ^ (state >> 17u);
    state = state ^ (state << 5u);
    state = state * 1597334677u;
    state = state ^ seed.y;
    state = state * 1597334677u;
    return state;
}

fn rand(seed: vec2<u32>) -> f32 {
    return f32(hash(seed)) / 4294967295.0;
}

fn randMinMax(seed: vec2<u32>, min: f32, max: f32) -> f32 {
    return min + (max - min) * rand(seed);
}

fn randVec3(seed: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(rand(seed), rand(seed + vec2<u32>(1u, 0u)), rand(seed + vec2<u32>(0u, 1u)));
}

fn randVec3MinMax(seed: vec2<u32>, min: f32, max: f32) -> vec3<f32> {
    return vec3<f32>(randMinMax(seed, min, max), randMinMax(seed + vec2<u32>(1u, 0u), min, max), randMinMax(seed + vec2<u32>(0u, 1u), min, max));
}

fn randInUnitSphere(seed: vec2<u32>) -> vec3<f32> {
    var tempseed = seed;
    loop {
        let p = randVec3MinMax(tempseed, -1.0, 1.0);
        if length(p) < 1.0 {
            return p;
        }
        tempseed = vec2<u32>(hash(tempseed), hash(tempseed + vec2<u32>(1u, 1u)));
    }
}

fn randInUnitDisk(seed: vec2<u32>) -> vec3<f32> {
    var tempseed = seed;
    loop {
        let p = vec3<f32>(randMinMax(tempseed, -1.0, 1.0), randMinMax(tempseed + vec2<u32>(1u, 0u), -1.0, 1.0), 0.0);
        if dot(p, p) < 1.0 {
            return p;
        }
        tempseed = vec2<u32>(hash(tempseed), hash(tempseed + vec2<u32>(1u, 1u)));
    }
}

fn randUnitVector(seed: vec2<u32>) -> vec3<f32> {
    return normalize(randInUnitSphere(seed));
}

fn randomOnHemisphere(normal: vec3<f32>, seed: vec2<u32>) -> vec3<f32> {
    let on_unit_sphere = randUnitVector(seed);
    if (dot(on_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn refract(uv: vec3<f32>, n: vec3<f32>, etai_over_etat: f32) -> vec3<f32> {
    let cos_theta = min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - length(r_out_perp) * length(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

fn nearZero(v: vec3<f32>) -> bool {
    let s = 1e-8;
    return (v.x > -s && v.x < s) && (v.y > -s && v.y < s) && (v.z > -s && v.z < s);
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

const materialShader = `
struct Material {
    albedo: vec3<f32>,
    fuzziness: f32,
    refraction_index: f32,
    mat_type: u32,
}

struct ScatterRecord {
    scattered: Ray,
    attenuation: vec3<f32>,
    is_scattered: bool,
}

fn scatterLambertian(r: Ray, rec: HitRecord, material: Material, seed: vec2<u32>) -> ScatterRecord {
    var scatter_direction = rec.normal + randUnitVector(seed);
    if (nearZero(scatter_direction)) {
        scatter_direction = rec.normal;
    }
    let scattered = Ray(rec.p, scatter_direction);
    let attenuation = material.albedo;
    return ScatterRecord(scattered, attenuation, true);
}

fn scatterMetal(r: Ray, rec: HitRecord, material: Material, seed: vec2<u32>) -> ScatterRecord {
    let reflected = reflect(normalize(r.direction), rec.normal);
    let scattered = Ray(rec.p, reflected + material.fuzziness * randInUnitSphere(seed));
    let attenuation = material.albedo;
    let is_scattered = dot(scattered.direction, rec.normal) > 0.0;
    return ScatterRecord(scattered, attenuation, is_scattered);
}

fn scatterDielectric(r: Ray, rec: HitRecord, material: Material, seed: vec2<u32>) -> ScatterRecord {
    let attenuation = vec3<f32>(1.0, 1.0, 1.0);
    var refraction_ratio: f32;
    if (rec.front_face) {
        refraction_ratio = 1.0 / material.refraction_index;
    } else {
        refraction_ratio = material.refraction_index;
    }

    let unit_direction = normalize(r.direction);
    let cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand(seed)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    return ScatterRecord(Ray(rec.p, direction), attenuation, true);
}
`;

const hittableShapesShader = `
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material: Material,
}

struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    hit: bool,
    front_face: bool,
    material: Material,
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
    rec.material = sphere.material;
    rec.hit = true;

    return rec;
}

fn hit_spheres(r: Ray, world: array<Sphere, NUM_SPHERES>, ray_t: Interval) -> HitRecord {
    var closest_so_far = ray_t.maxI;
    var rec: HitRecord;
    rec.hit = false;

    for (var i = 0u; i < NUM_SPHERES; i++) { 
        let sphere_rec = hit_sphere(world[i], r, createInterval(ray_t.minI, closest_so_far));
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
    vfov: f32,
    lookfrom: vec3<f32>,
    lookat: vec3<f32>,
    vup: vec3<f32>,
    defocus_angle: f32,
    focus_distance: f32,
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
    defocus_disk_u: vec3<f32>,
    defocus_disk_v: vec3<f32>,
}

fn createCamera(aspect_ratio: f32) -> Camera {
    let samples_per_pixel: u32 = 200;
    let vfov = 20.0;
    let lookfrom = vec3<f32>(13.0, 2.0, 3.0);
    let lookat = vec3<f32>(0.0, 0.0, 0.0);
    let vup = vec3<f32>(0.0, 1.0, 0.0);
    let defocus_angle = 0.6;
    let focus_distance = 10.0;

    let theta = degreesToRadians(vfov);
    let h = tan(theta / 2.0);
    let viewport_height = 2.0 * h * focus_distance;
    let viewport_width = aspect_ratio * viewport_height;

    let w = normalize(lookfrom - lookat);
    let u = normalize(cross(vup, w));
    let v = cross(w, u);

    let origin = lookfrom;
    let horizontal = viewport_width * u;
    let vertical = viewport_height * v;
    let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_distance * w;

    let defocus_radius = focus_distance * tan(degreesToRadians(defocus_angle / 2.0));
    let defocus_disk_u = u * defocus_radius;
    let defocus_disk_v = v * defocus_radius;

    return Camera(origin, lower_left_corner, horizontal, vertical, 
                  samples_per_pixel, vfov, lookfrom, lookat, vup, 
                  defocus_angle, focus_distance, u, v, w, defocus_disk_u, defocus_disk_v);
}

fn getRay(camera: Camera, s: f32, t: f32, seed: vec2<u32>) -> Ray {
    var rd: vec3<f32> = camera.origin;  // This should be vec3<f32>(0.0, 0.0, 0.0)
    if (camera.defocus_angle > 0.0) {
        let p = randInUnitDisk(seed);
        rd = (camera.defocus_disk_u * p.x + camera.defocus_disk_v * p.y);
    }
    let offset = camera.u * rd.x + camera.v * rd.y;
    return Ray(
        camera.origin + offset,
        camera.lower_left_corner + s*camera.horizontal + t*camera.vertical - camera.origin - offset
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
            ${materialShader}
            ${hittableShapesShader}
            ${cameraShader}
            struct Ray {
                origin: vec3<f32>,
                direction: vec3<f32>
            }

            const MATERIAL_GROUND: Material = Material(vec3<f32>(0.8, 0.8, 0.0), 0.0, 0.0, 0);
            const MATERIAL_CENTER: Material = Material(vec3<f32>(0.1, 0.2, 0.5), 0.0, 0.0, 0);
            const MATERIAL_LEFT: Material = Material(vec3<f32>(0.8, 0.8, 0.8), 0.0, 1.5, 2);
            const MATERIAL_BUBBLE: Material = Material(vec3<f32>(1.0, 1.0, 1.0), 0.0, 1.0/1.5, 2);
            const MATERIAL_RIGHT: Material = Material(vec3<f32>(0.8, 0.6, 0.2), 1.0, 0.0, 1);

            const R = cos(PI / 4.0);

            // const TEMP_MATERIAL_LEFT: Material = Material(vec3<f32>(0.0, 0.0, 1.0), 0.0, 0.0, 0);
            // const TEMP_MATERIAL_RIGHT: Material = Material(vec3<f32>(1.0, 0.0, 0.0), 0.0, 0.0, 0);

            // const spheres = array<Sphere, 5>(
            //     Sphere(vec3<f32>(0.0, -100.5, -1.0), 100.0, MATERIAL_GROUND),
            //     Sphere(vec3<f32>(0.0, 0.0, -1.2), 0.5, MATERIAL_CENTER),
            //     Sphere(vec3<f32>(-1.0, 0.0, -1.0), 0.5, MATERIAL_LEFT),
            //     Sphere(vec3<f32>(-1.0, 0.0, -1.0), 0.4, MATERIAL_BUBBLE),
            //     Sphere(vec3<f32>(1.0, 0.0, -1.0), 0.5, MATERIAL_RIGHT)
            //     //Sphere(vec3<f32>(-R, 0.0, -1.0), R, TEMP_MATERIAL_LEFT),
            //     //Sphere(vec3<f32>(R, 0.0, -1.0), R, TEMP_MATERIAL_RIGHT)
            // );

            fn createRay(uv: vec2<f32>) -> Ray {
                let aspectRatio = f32(textureDimensions(output).x) / f32(textureDimensions(output).y);
                let origin = vec3<f32>(0.0, 0.0, 0.0);
                let direction = vec3<f32>(uv.x * aspectRatio, uv.y, -1.0);
                return Ray(origin, normalize(direction));
            }

            fn rayColor(initial_ray: Ray, world: array<Sphere, NUM_SPHERES>, seed: vec2<u32>) -> vec3<f32> {
                var ray = initial_ray;
                var color = vec3<f32>(1.0, 1.0, 1.0);
                var current_seed = seed;
                
                for (var depth = 0u; depth < MAX_DEPTH; depth++) {
                    if (depth == MAX_DEPTH - 1) {
                        color *= 0.0;
                    }
                    let rec = hit_spheres(ray, world, createInterval(0.001, INFINITY));
                    if (rec.hit) {
                        current_seed = vec2<u32>(hash(current_seed), depth);

                        //let direction = randomOnHemisphere(rec.normal, current_seed);
                        let direction = rec.normal + randUnitVector(current_seed);

                        var scatterRec: ScatterRecord;

                        if (rec.material.mat_type == 0) {
                            scatterRec = scatterLambertian(ray, rec, rec.material, current_seed);
                        } else if (rec.material.mat_type == 1) {
                            scatterRec = scatterMetal(ray, rec, rec.material, current_seed);
                        } else if (rec.material.mat_type == 2) {
                            scatterRec = scatterDielectric(ray, rec, rec.material, current_seed);
                        } else {
                            // Handle unknown material type
                            return vec3<f32>(1.0, 0.0, 1.0); // Magenta for error
                        }

                        if (scatterRec.is_scattered) {
                            ray = scatterRec.scattered;
                            color *= scatterRec.attenuation;
                        } else {
                            return vec3<f32>(0.0, 0.0, 0.0); // Ray was absorbed
                        }
                    } else {
                        let unit_direction = normalize(ray.direction);
                        let a = 0.5 * (unit_direction.y + 1.0);
                        color *= (1.0 - a) * vec3<f32>(1.0, 1.0, 1.0) + a * vec3<f32>(0.5, 0.7, 1.0);
                        break;
                    }
                }
                
                return color;
            }

            fn rayAt(ray: Ray, t: f32) -> vec3<f32> {
                return ray.origin + ray.direction * t;
            }

            fn createRandomScene() {
                // Ground
                spheres[0] = Sphere(
                    vec3<f32>(0.0, -1000.0, 0.0),
                    1000.0,
                    Material(vec3<f32>(0.5, 0.5, 0.5), 0.0, 0.0, 0) // Lambertian
                );

                // Three large spheres
                spheres[1] = Sphere(vec3<f32>(0.0, 1.0, 0.0), 1.0, Material(vec3<f32>(1.0), 0.0, 1.5, 2)); // Glass
                spheres[2] = Sphere(vec3<f32>(-4.0, 1.0, 0.0), 1.0, Material(vec3<f32>(0.4, 0.2, 0.1), 0.0, 0.0, 0)); // Lambertian
                spheres[3] = Sphere(vec3<f32>(4.0, 1.0, 0.0), 1.0, Material(vec3<f32>(0.7, 0.6, 0.5), 0.0, 0.0, 1)); // Metal

                // Random small spheres
                for (var i = 4u; i < NUM_SPHERES; i++) {
                    let choose_mat = rand(vec2<u32>(i, 0u));
                    let center = vec3<f32>(
                        randMinMax(vec2<u32>(i, 1u), -4.0, 4.0),
                        0.2,
                        randMinMax(vec2<u32>(i, 2u), -4.0, 4.0)
                    );

                    if (length(center - vec3<f32>(4.0, 0.2, 0.0)) > 0.9) {
                        if (choose_mat < 0.8) {
                            // Diffuse
                            let albedo = randVec3(vec2<u32>(i, 3u)) * randVec3(vec2<u32>(i, 4u));
                            spheres[i] = Sphere(center, 0.2, Material(albedo, 0.0, 0.0, 0));
                        } else if (choose_mat < 0.95) {
                            // Metal
                            let albedo = randVec3MinMax(vec2<u32>(i, 5u), 0.5, 1.0);
                            let fuzz = randMinMax(vec2<u32>(i, 6u), 0.0, 0.5);
                            spheres[i] = Sphere(center, 0.2, Material(albedo, fuzz, 0.0, 1));
                        } else {
                            // Glass
                            spheres[i] = Sphere(center, 0.2, Material(vec3<f32>(1.0), 0.0, 1.5, 2));
                        }
                    } else {
                        // If the position is not valid, create a default sphere
                        spheres[i] = Sphere(vec3<f32>(0.0), 0.1, Material(vec3<f32>(1.0), 0.0, 0.0, 0));
                    }
                }
            }


            @group(0) @binding(0) var<storage, read_write> spheres: array<Sphere, NUM_SPHERES>;
            @group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(output);
                let coords = vec2<u32>(id.xy);
                
                if (coords.x >= dims.x || coords.y >= dims.y) {
                    return;
                }

                let aspect_ratio = f32(dims.x) / f32(dims.y);
                let camera = createCamera(aspect_ratio);
                if (id.x == 0u && id.y == 0u) {
                    createRandomScene();
                }

                var pixel_color = vec3<f32>(0.0, 0.0, 0.0);
                for (var s = 0u; s < camera.samples_per_pixel; s++) {
                    let seed = vec2<u32>(coords.x + dims.x * coords.y, s);
                    let u = (f32(coords.x) + rand(seed)) / f32(dims.x);
                    let v = (f32(coords.y) + rand(seed + vec2<u32>(1u, 1u))) / f32(dims.y); 
                    let ray = getRay(camera, u, v, seed);
                    pixel_color += rayColor(ray, spheres, seed);
                }
                
                pixel_color = sqrt(pixel_color / f32(camera.samples_per_pixel)); // Gamma correction

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

    // After creating spheresBuffer
    const sphereData = new Float32Array(NUM_SPHERES * 12); // 8 floats per sphere (3 for center, 1 for radius, 4 for material)

    for (let i = 0; i < NUM_SPHERES; i++) {
        const offset = i * 12;
        sphereData[offset] = 0; // center.x
        sphereData[offset + 1] = 0; // center.y
        sphereData[offset + 2] = 0; // center.z
        sphereData[offset + 3] = 1; // radius
        sphereData[offset + 4] = 1; // material.albedo.r
        sphereData[offset + 5] = 1; // material.albedo.g
        sphereData[offset + 6] = 1; // material.albedo.b
        sphereData[offset + 7] = 0; // material.mat_type
        sphereData[offset + 8] = 0; // material.fuzziness
        sphereData[offset + 9] = 1; // material.refraction_index
        // offset + 10 and offset + 11 are padding and can be left as 0
    }

    const spheresBuffer = device.createBuffer({
        size: sphereData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // After creating spheresBuffer
    // Fill sphereData with your sphere information
    device.queue.writeBuffer(spheresBuffer, 0, sphereData);

    const texture = device.createTexture({
        size: textureSize,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    const bindGroup = device.createBindGroup({
        label: "Compute bind group",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: spheresBuffer } },
            { binding: 1, resource: texture.createView() }
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
    const width = canvas.getBoundingClientRect().width;
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

main().then(() =>
    console.log("Total time:", performance.now() - renderTime)
).catch(e => {
    console.error(e);
    alert(e);
});
