// Enhanced+: what the ray-traced passes share (rt.rs): the parameters, the scene's
// acceleration structure with its geometry records, the ray queries.
enable wgpu_ray_query;

struct RtParams {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    // xyz the camera (render-origin relative), w the frame number
    eye: vec4<f32>,
    // xyz towards the sun, w the tangent of the sun disc's radius as traced (0: no sun)
    sun: vec4<f32>,
    // width, height, 1 / width, 1 / height
    size: vec4<f32>,
    // xy the projection's depth coefficients, z the traced range (m), w the AO radius (m)
    proj: vec4<f32>,
    // x history usable, y the history's least weight, z debug view, w frames accumulated at most
    temporal: vec4<f32>,
    // the player's vehicle now: xyz its origin, w its heading (rad); and a frame ago (in this
    // frame's render-origin coordinates); its box half extents (w: 1 = there is one) and centre
    vehicle_now: vec4<f32>,
    vehicle_prev: vec4<f32>,
    vehicle_box: vec4<f32>,
    vehicle_centre: vec4<f32>,
    // the sky light for the reflections' hits: xyz the mean sky irradiance / pi (pre-exposed),
    // w the probe's scale
    sky: vec4<f32>,
    // xyz the sun's radiance at the hits (pre-exposed), w the probe's mip count
    sun_light: vec4<f32>,
};

struct Record {
    tex: u32,
    trans: u32,
    // bit 0: alpha-tested (stochastic), bit 1: unlit (shows its own colour), bit 2: the
    // coverage is the transmap's alpha, bits 3-4 how its textures are addressed (rt_tex.rs
    // `address_mode`), bit 5: opaque all the same (a transmap whose alpha is not taken),
    // bit 6: a see-through pane
    flags: u32,
    // its first triangle's texture coordinates (rt_tex.rs; 0xffffffff: none yet)
    uvs: u32,
    color: vec4<f32>,
    // the diffuse texture's coordinates: (uv + yz) * x (a texcoordtrans offset, the ground's
    // repeat)
    uvt: vec4<f32>,
};

// One texture's mean colour and its tile in the reflections' atlas (xy its corner, z 1 when
// it is there), 256 bytes apart (written with a dynamic offset, see rt.rs).
struct TexAvg {
    c: vec4<f32>,
    tile: vec4<f32>,
    pad: array<vec4<f32>, 14>,
};

@group(0) @binding(0) var<uniform> p: RtParams;
@group(0) @binding(1) var acc: acceleration_structure;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var<storage, read> records: array<Record>;
@group(0) @binding(4) var<storage, read> tex_avg: array<TexAvg>;

// instance masks: solid shadow casters, solid meshes seen (by the occlusion, the
// reflections), cut-out ones seen, panes
const MASK_SHADOW: u32 = 0x01u;
const MASK_SEEN: u32 = 0x06u;
// see-through panes (their shadow: part of the sun gets through)
const MASK_GLASS: u32 = 0x08u;
// ... and the same panes as the reflections' rays meet them
const MASK_GLASS_SEEN: u32 = 0x10u;
// Ray flags. (Not called RAY_FLAG_*: Direct3D's HLSL has those names built in, and naga
// writes the constants out under their own names - DXC refused the redefinition, the ray
// tracing's pipelines were never made on Windows, and every frame was thrown away.)
const RT_FORCE_OPAQUE: u32 = 0x01u;
const RT_FIRST_HIT: u32 = 0x04u;
const PI: f32 = 3.14159265;
// The roughness up to which a reflection is traced along the mirror direction alone and
// blurred on the screen by its lobe's width (rt_denoise.wgsl `cs_atrous`); above it the
// rays follow the lobe, a new direction every frame, and the frames are averaged. (Rays
// scattered over the lobe of a pane, a polished coat or a wet road - one a pixel, a fresh
// one every frame - left these, where the reflection shows most, as noise that crawled
// whenever anything moved: the history of a lobe that narrow is thrown away at the least
// motion. The blur gives the same soft image of the street, the same every frame.)
const GLOSSY_MIRROR: f32 = 0.25;

fn world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let h = p.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return h.xyz / h.w;
}
fn linear_depth(depth: f32) -> f32 {
    return p.proj.y / max(depth + p.proj.x, 1e-7);
}

// The surface's normal from the depth around it: of the two neighbours on each axis the one
// nearer in depth, so that an edge does not tilt it; turned towards the camera.
fn depth_normal(px: vec2<i32>, world: vec3<f32>, depth: f32) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(t_depth));
    var axes: array<vec3<f32>, 2>;
    for (var i = 0; i < 2; i++) {
        let axis = select(vec2<i32>(1, 0), vec2<i32>(0, 1), i == 1);
        let plus = clamp(px + axis, vec2<i32>(0), size - vec2<i32>(1));
        let minus = clamp(px - axis, vec2<i32>(0), size - vec2<i32>(1));
        let dp = textureLoad(t_depth, plus, 0);
        let dm = textureLoad(t_depth, minus, 0);
        let forward = dp > 0.0 && (dm <= 0.0 || abs(dp - depth) < abs(dm - depth));
        let q = select(minus, plus, forward);
        let d = select(dm, dp, forward);
        axes[i] = (world_pos((vec2<f32>(q) + vec2<f32>(0.5)) * p.size.zw, d) - world) * select(-1.0, 1.0, forward);
    }
    var n = cross(axes[0], axes[1]);
    if (dot(n, n) < 1e-14) {
        n = p.eye.xyz - world;
    }
    n = normalize(n);
    return select(-n, n, dot(n, p.eye.xyz - world) >= 0.0);
}

// Interleaved gradient noise, moved on every frame (a different sample per frame and pixel
// that spreads evenly over a few frames and pixels).
fn noise(px: vec2<f32>, k: f32) -> f32 {
    let f = p.eye.w + k * 17.0;
    let q = px + vec2<f32>(47.0, 17.0) * (f % 64.0) * 0.695;
    return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}

fn hash3(v: vec3<u32>) -> f32 {
    var x = v.x * 1664525u + v.y * 22695477u + v.z * 1013904223u;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x >> 8u) / 16777216.0;
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

// How much of a hit geometry's texels stop a ray: 1 for a solid one, the mean alpha of its
// cut-out texture (or transmap) for an alpha-tested one.
fn coverage(r: Record) -> f32 {
    if ((r.flags & 1u) == 0u || p.temporal.z == 3.0 || p.temporal.z == 5.0) {
        return 1.0;
    }
    if (p.temporal.z == 4.0) {
        return 0.0;
    }
    var a = 1.0;
    if ((r.flags & 4u) != 0u && r.trans != 0xffffffffu) {
        a = tex_avg[r.trans].c.a;
    } else if (r.tex != 0xffffffffu) {
        a = tex_avg[r.tex].c.a;
    }
    // (not the material's alpha: the alpha test compares the texture's own)
    return clamp(a, 0.0, 1.0);
}

struct Hit {
    t: f32,
    record: u32,
    front: bool,
    tries: u32,
};

// One ray query from `start` to `tmax`: its committed intersection. A function of its own,
// called for every ray: naga's Vulkan and Direct3D 12 code keeps a query's initialization
// tracker per function, not per loop iteration, and a query started again in a loop after
// a miss was never traced again there (each later try came back empty).
fn ray_hit(flags: u32, mask: u32, start: f32, tmax: f32, o: vec3<f32>, d: vec3<f32>) -> RayIntersection {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(flags, mask, start, tmax, o, d));
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

// The first hit along a ray within `tmax` (t < 0: none). An alpha-tested hit stops the ray by
// its coverage's chance; else the ray goes on past it (Metal's ray queries take no any-hit
// test of their own: the query is started again from there).
fn trace(o: vec3<f32>, d: vec3<f32>, tmax: f32, mask: u32, seed: vec3<u32>, solid_only: bool) -> Hit {
    var start = 0.0;
    var out: Hit;
    out.t = -1.0;
    for (var i = 0u; i < 6u; i++) {
        out.tries = i + 1u;
        let h = ray_hit(RT_FORCE_OPAQUE, mask, start, tmax, o, d);
        if (h.kind == RAY_QUERY_INTERSECTION_NONE) {
            return out;
        }
        let ri = h.instance_custom_data + h.geometry_index;
        let r = records[ri];
        if ((r.flags & 1u) == 0u || (!solid_only && hash3(seed + vec3<u32>(i * 7919u, ri, 0u)) < coverage(r))) {
            out.t = h.t;
            out.record = ri;
            out.front = h.front_face;
            return out;
        }
        start = h.t + 0.002;
    }
    return out;
}


// Enhanced+: the traced reflections' denoiser (rt.rs), between the rays (rt_reflect.wgsl
// `cs_reflect`) and the composite.
//
// A glossy surface's rays follow its lobe, a new direction every frame: one ray is noise, the
// lobe's average is the reflection. That average is gathered
// - over the frames (`cs_temporal`): each pixel's history is found where its surface was a
//   frame ago - or, for a smooth one, where the mirror image of what it reflects was, which
//   moves with the reflected scene and not with the mirror - and kept as long as the same
//   surface lies there; it is held inside the spread of this frame's rays round the pixel,
//   so that nothing that has moved leaves a trail. A mirror keeps no history: its one ray is
//   exact. The luminance's first two moments are kept along, for how noisy the pixel still is;
// - over the pixels round it (`cs_atrous`, three passes of an a-trous wavelet 1, 2 and 4
//   pixels apart): only from those on the same surface (distance, normal, roughness), as far
//   round as its lobe spreads the reflection on the screen, and less from those whose colour
//   differs by more than the pixel's own noise (its variance) explains.

@group(0) @binding(5) var t_meta: texture_2d<f32>;
@group(0) @binding(6) var t_cur: texture_2d<f32>;
// the temporal pass: last frame's history and its moments; the a-trous passes: this frame's
// rays (their reach in a) and moments (the first pass's variance in w)
@group(0) @binding(7) var t_prev: texture_2d<f32>;
@group(0) @binding(8) var t_prev_m: texture_2d<f32>;
@group(0) @binding(9) var t_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var t_out_m: texture_storage_2d<rgba16float, write>;
// x the a-trous step (pixels), y 1 when the variance is the moments' (the first pass)
@group(0) @binding(11) var<uniform> pass_step: vec4<f32>;

// The rays' grid: one per REFL_DIV x REFL_DIV pixels.
override REFL_DIV: i32 = 1;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return max(vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z), vec3<f32>(0.0));
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n = vec3<f32>(n.xy + select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0)), n.z);
    return normalize(n);
}

// The place on the screen a ray's grid cell stands for (uv of its middle).
fn cell_uv(q: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(q * REFL_DIV) + vec2<f32>(0.5 * f32(REFL_DIV))) * p.size.zw;
}

// A point on the player's vehicle where it was a frame ago (the vehicle moves under the
// camera: its own surfaces stay put on the screen while the world slides past).
fn vehicle_prev(pt: vec3<f32>) -> vec3<f32> {
    if (p.vehicle_box.w < 0.5) {
        return pt;
    }
    let d = pt - p.vehicle_now.xyz;
    let cn = cos(p.vehicle_now.w);
    let sn = sin(p.vehicle_now.w);
    // (into the vehicle's frame: its heading turned back)
    let local = vec3<f32>(d.x * cn + d.y * sn, -d.x * sn + d.y * cn, d.z) - p.vehicle_centre.xyz;
    if (any(abs(local) > p.vehicle_box.xyz + vec3<f32>(0.3))) {
        return pt;
    }
    let l = local + p.vehicle_centre.xyz;
    let cp = cos(p.vehicle_prev.w);
    let sp = sin(p.vehicle_prev.w);
    return p.vehicle_prev.xyz + vec3<f32>(l.x * cp - l.y * sp, l.x * sp + l.y * cp, l.z);
}

fn in_vehicle(pt: vec3<f32>) -> bool {
    return any(vehicle_prev(pt) != pt);
}

// How far round (in rays) a surface's reflection is spread on the screen: the lobe's angle
// (about alpha) over what it reflects `reach` metres behind it, seen from `dist` - nothing for
// a mirror, more for a rougher surface and for what it reflects from further away (a
// reflection touching its object stays sharp where it meets it).
fn lobe_radius(uv: vec2<f32>, rough: f32, reach: f32, dist: f32) -> f32 {
    let a = rough * rough;
    let angle = a * reach / (dist + reach);
    // (the angle one ray's cell spans)
    let v0 = normalize(world_pos(uv, 0.001) - p.eye.xyz);
    let v1 = normalize(world_pos(uv + vec2<f32>(0.0, p.size.w * f32(REFL_DIV)), 0.001) - p.eye.xyz);
    let cell = max(length(v1 - v0), 1e-6);
    return min(angle / cell, 24.0);
}

@compute @workgroup_size(8, 8)
fn cs_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = vec2<i32>(gid.xy);
    let size = vec2<i32>(textureDimensions(t_cur));
    if (any(q >= size)) {
        return;
    }
    let m = textureLoad(t_meta, q, 0);
    let cur = textureLoad(t_cur, q, 0);
    if (m.x <= 0.0) {
        textureStore(t_out, q, vec4<f32>(0.0));
        textureStore(t_out_m, q, vec4<f32>(0.0));
        return;
    }
    let dist = m.x;
    let rough = m.y;
    let uv = cell_uv(q);
    let view = normalize(world_pos(uv, 0.001) - p.eye.xyz);
    let pt = p.eye.xyz + view * dist;
    let clip_w = (p.view_proj * vec4<f32>(pt, 1.0)).w;
    // --- this frame's rays round the pixel, on its own surface: their mean and spread (in
    // YCoCg, where a box round them holds the colours better than in RGB)
    let tol = 0.03 * dist + 0.1;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var lm1 = 0.0;
    var lm2 = 0.0;
    var cnt = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let s = clamp(q + vec2<i32>(i, j), vec2<i32>(0), size - vec2<i32>(1));
            let ms = textureLoad(t_meta, s, 0);
            if (ms.x > 0.0 && abs(ms.x - dist) < tol) {
                let c = textureLoad(t_cur, s, 0).rgb;
                let y = to_ycocg(c);
                m1 += y;
                m2 += y * y;
                let l = luma(c);
                lm1 += l;
                lm2 += l * l;
                cnt += 1.0;
            }
        }
    }
    m1 /= cnt;
    let sigma = sqrt(max(m2 / cnt - m1 * m1, vec3<f32>(0.0)));
    let spatial_var = max(lm2 / cnt - (lm1 / cnt) * (lm1 / cnt), 0.0);
    // --- where the history is: the surface a frame ago and, for a smooth one, the mirror
    // image of what it reflects (as far behind the surface as the ray went)
    let on_vehicle = in_vehicle(pt);
    let prev_pt = vehicle_prev(pt);
    let pc = p.prev_view_proj * vec4<f32>(prev_pt, 1.0);
    var hist = vec4<f32>(0.0);
    var hist_m = vec2<f32>(0.0);
    var found = 0.0;
    if (p.temporal.x > 0.5 && pc.w > 0.05) {
        var puv = vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
        let mirror = (1.0 - smoothstep(0.1, 0.35, rough)) * select(1.0, 0.0, on_vehicle);
        if (mirror > 0.0) {
            let vc = p.prev_view_proj * vec4<f32>(p.eye.xyz + view * (dist + min(cur.a, 2000.0)), 1.0);
            if (vc.w > 0.05) {
                let vuv = vec2<f32>(vc.x / vc.w * 0.5 + 0.5, 0.5 - vc.y / vc.w * 0.5);
                puv = mix(puv, vuv, mirror);
            }
        }
        // (bilinear over the four cells round it, each only where the same surface was)
        let f = puv * vec2<f32>(size) - vec2<f32>(0.5);
        let b = vec2<i32>(floor(f));
        let fr = f - floor(f);
        let ptol = 0.04 * pc.w + 0.15;
        var wsum = 0.0;
        for (var j = 0; j <= 1; j++) {
            for (var i = 0; i <= 1; i++) {
                let s = b + vec2<i32>(i, j);
                if (all(s >= vec2<i32>(0)) && all(s < size)) {
                    let pm = textureLoad(t_prev_m, s, 0);
                    let w = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
                    if (pm.z > 0.0 && abs(pm.z - pc.w) < ptol && w > 1e-3) {
                        hist += textureLoad(t_prev, s, 0) * w;
                        hist_m += pm.xy * w;
                        wsum += w;
                    }
                }
            }
        }
        if (wsum > 0.05) {
            hist /= wsum;
            hist_m /= wsum;
            found = 1.0;
        }
    }
    // --- the blend: as many frames as the lobe needs (none for a mirror), the history held
    // inside the box of this frame's rays (wider for a rough surface, whose rays scatter)
    // (a ray along the mirror direction is the same every frame: nothing to average, and no
    // history to trail behind what moves)
    let most = select(1.0, mix(12.0, 32.0, smoothstep(GLOSSY_MIRROR, 0.5, rough)), rough > GLOSSY_MIRROR);
    let count = min(select(0.0, hist.a, found > 0.5) + 1.0, most);
    let alpha = 1.0 / count;
    var c = cur.rgb;
    var lmom = vec2<f32>(luma(cur.rgb), luma(cur.rgb) * luma(cur.rgb));
    if (found > 0.5 && count > 1.0) {
        let gamma = 1.0 + 2.0 * smoothstep(0.1, 0.6, rough);
        let hy = to_ycocg(hist.rgb);
        // (clipped towards the box's middle, not clamped per channel: no colour shift)
        let dir = hy - m1;
        // (not narrower than a few per cent of the brightness: a grey reflection's chroma
        // hardly spreads, and a box that thin snapped every history back to the noisy mean)
        let ext = max(sigma * gamma, vec3<f32>(0.03 * max(m1.x, 0.0) + 1e-3));
        let t = max(abs(dir.x) / ext.x, max(abs(dir.y) / ext.y, abs(dir.z) / ext.z));
        var held = hy;
        if (t > 1.0) {
            held = m1 + dir / t;
        }
        c = mix(from_ycocg(held), cur.rgb, alpha);
        lmom = mix(hist_m, lmom, alpha);
    }
    // (the noise left: the moments' once a few frames are in, before that the spread of the
    // rays round it)
    let temporal_var = max(lmom.y - lmom.x * lmom.x, 0.0);
    let variance = mix(spatial_var, temporal_var, smoothstep(2.0, 6.0, count));
    textureStore(t_out, q, vec4<f32>(c, count));
    textureStore(t_out_m, q, vec4<f32>(lmom, clip_w, variance));
}

// One pass of the a-trous wavelet (the B3 spline's 5 x 5 taps, `pass_step.x` rays apart).
@compute @workgroup_size(8, 8)
fn cs_atrous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = vec2<i32>(gid.xy);
    let size = vec2<i32>(textureDimensions(t_cur));
    if (any(q >= size)) {
        return;
    }
    let m = textureLoad(t_meta, q, 0);
    let c = textureLoad(t_cur, q, 0);
    var var_c = c.a;
    if (pass_step.y > 0.5) {
        var_c = textureLoad(t_prev_m, q, 0).w;
    }
    if (m.x <= 0.0) {
        textureStore(t_out, q, vec4<f32>(0.0));
        return;
    }
    let dist = m.x;
    let rough = m.y;
    let n = oct_decode(m.zw);
    let step = max(i32(pass_step.x), 1);
    // (a mirror, or a lobe narrower than this pass's step: the pixel as it is)
    let radius = lobe_radius(cell_uv(q), rough, max(textureLoad(t_prev, q, 0).a, 0.0), dist);
    if (rough < 0.03 || radius < 0.5 * f32(step)) {
        textureStore(t_out, q, vec4<f32>(c.rgb, var_c));
        return;
    }
    let lc = luma(c.rgb);
    let sigma_l = 4.0 * sqrt(max(var_c, 0.0)) + 1e-3;
    let k = array<f32, 3>(0.375, 0.25, 0.0625);
    var sum = vec3<f32>(0.0);
    var vsum = 0.0;
    var wsum = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let s = q + vec2<i32>(i, j) * step;
            if (any(s < vec2<i32>(0)) || any(s >= size)) {
                continue;
            }
            let ms = textureLoad(t_meta, s, 0);
            if (ms.x <= 0.0) {
                continue;
            }
            let cs = textureLoad(t_cur, s, 0);
            var vs = cs.a;
            if (pass_step.y > 0.5) {
                vs = textureLoad(t_prev_m, s, 0).w;
            }
            let r = length(vec2<f32>(f32(i), f32(j))) * f32(step);
            let w_space = k[abs(i)] * k[abs(j)] * exp(-r * r / (2.0 * radius * radius));
            let w_depth = exp(-abs(ms.x - dist) / (0.01 * dist * f32(step) + 0.05));
            let w_normal = pow(max(dot(oct_decode(ms.zw), n), 0.0), 32.0);
            let w_rough = exp(-abs(ms.y - rough) * 12.0);
            let w_luma = exp(-abs(luma(cs.rgb) - lc) / sigma_l);
            // (the mirror rays are blurred as wide as the lobe whatever their colours: they
            // are not noisy, and the colour's edges are the reflection's own)
            let w = w_space * w_depth * w_normal * w_rough * select(w_luma, 1.0, (i == 0 && j == 0) || rough <= GLOSSY_MIRROR);
            sum += cs.rgb * w;
            vsum += vs * w * w;
            wsum += w;
        }
    }
    let out = sum / max(wsum, 1e-5);
    textureStore(t_out, q, vec4<f32>(out, vsum / max(wsum * wsum, 1e-8)));
}
