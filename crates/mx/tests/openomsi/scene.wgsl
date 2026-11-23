fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}

struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    world_origin: vec4<f32>, // world-space origin removed from render coordinates
    sun_dir: vec4<f32>,      // xyz direction to the sun, w intensity
    ambient: vec4<f32>,      // rgb ambient (envir light C)
    fog: vec4<f32>,          // rgb, density
    sun_color: vec4<f32>,    // rgb (envir light A), w night factor 0..1 (nightmap strength)
    sky_color: vec4<f32>,    // rgb secondary light from above (envir light B), w 1 = vanilla (as OMSI 2), 0 = Vanilla+
    light_grid: vec4<f32>,   // x,y grid origin (render-origin relative), z cell size, w cells per side
    sky: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    clouds: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    light_view_proj_far: mat4x4<f32>,
    shadow: vec4<f32>,       // x enabled, y texel size, z near cascade half range
    post: vec4<f32>,         // x enhanced, y time, zw sun ndc
    inside_a: vec4<f32>,     // player vehicle box: origin xyz, sin(heading)
    inside_b: vec4<f32>,     // cos(heading), half extents xyz
    inside_c: vec4<f32>,     // box centre offset xyz, w = 1 when there is a box, 2 with the eye in it
    flags: vec4<f32>,        // x detail texturing, y enhanced graphics, z never set (see fs_main's end), w close cascade half range
    light_view_proj_close: mat4x4<f32>,
    wind: vec4<f32>,         // the player's vehicle's velocity (m/s, world): the airstream on its glass
    // Enhanced: the lamps' shadow maps (the tiles under the far map; zero where nothing is
    // drawn) and where the tiles lie: x a tile's side, y tiles to a row, z the far map's
    // texture height, w the number of tiles (`LampAtlas::uniform`)
    lamp_view_proj: array<mat4x4<f32>, 56>,
    lamp_atlas: vec4<f32>,
    // Windy trees: xy the weather's wind (m/s, world; 0 with the setting off), zw how far
    // the air has carried the gusts since the start (m, modulo PATTERN_PERIOD)
    tree_wind: vec4<f32>,
    // an articulated bus's rear section (as inside_*): read by the snowfall only - a test of
    // it here, inlined into every use of inside_vehicle, brought Apple's M4 shader compiler
    // down (see 0.2.21's Metal note)
    inside2_a: vec4<f32>,
    inside2_b: vec4<f32>,
    inside2_c: vec4<f32>,
    // Enhanced: the field each light cookie layer covers (xy: the tangents of its half
    // angles across and up; z: its first record in the light buffer), see enhanced.wgsl
    // `cookie_light`
    cookies: array<vec4<f32>, 8>,
    // Enhanced: the lights of the player's saloon that have a shadow map (their places in the
    // light buffer, 4 a vector, -1 for none): see `cabin_visibility`
    cabin_lamps: array<vec4<f32>, 4>,
    // x: the textures' mip bias (FSR 1: negative, see `CameraUniform::lod`)
    lod: vec4<f32>,           // x mip bias, y ground-prepass flag, z road sunlight glare, w air temperature (degC)
    // the classic picture's headlight (`CameraUniform::spot`, see `classic_spot`)
    spot: array<vec4<f32>, 4>,
};

// 1 when the point lies inside the player's vehicle (its [boundingbox], shrunk a little so
// that the outer skin, the glass and the roof stay outside): weather stays out of the cab.
// The side/front/back shrink was a bare 0.12 m, which left an interior fixture close to the
// wall (a window sill, the handbrake by the windscreen) just outside the box, wearing the
// up-facing wet/snow coat meant for the body panel beside it (see the `outside` weight
// below). A previous fix WIDENED this margin to 0.6 m, reasoning by analogy with the falling-
// particle exclusion box in `rain.rs` - but that box works the other way round (its margin
// is added, growing the excluded region outward, away from the body) from this one (whose
// margin is subtracted, shrinking the "inside" region inward, toward the body): widening it
// here shrank the cabin's own "inside" zone and threw the *whole* cabin outside it, coating
// the entire interior in snow instead of just the ledge. Narrowed to 0.03 m instead - just
// enough that the true outer skin and glass stay outside, without giving up real interior
// floor space near the wall.
// A normal of no length (a mesh saved with zero normals) is upright instead of NaN:
// normalize(0) is NaN on Vulkan and Metal and the pixel went black.
fn safe_normal(v: vec3<f32>) -> vec3<f32> {
    let d = dot(v, v);
    if (d > 1e-20) {
        return v * inverseSqrt(d);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn inside_vehicle(world: vec3<f32>) -> f32 {
    if (camera.inside_c.w < 0.5) {
        return 0.0;
    }
    let d = world - camera.inside_a.xyz;
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let x = d.x * ch - d.y * sh - camera.inside_c.x;
    let y = d.x * sh + d.y * ch - camera.inside_c.y;
    let z = d.z - camera.inside_c.z;
    let h = camera.inside_b.yzw;
    let in_x = abs(x) < h.x - 0.03;
    let in_y = abs(y) < h.y - 0.03;
    let in_z = z > -h.z - 0.6 && z < h.z - 0.03;
    return select(0.0, 1.0, in_x && in_y && in_z);
}

// 1 when the point belongs to the player's vehicle: its `[boundingbox]` grown a little,
// so that the skin and the glass the cab test (`inside_vehicle`) leaves out are in.
fn near_player_vehicle(world: vec3<f32>) -> f32 {
    if (camera.inside_c.w < 0.5) {
        return 0.0;
    }
    let d = world - camera.inside_a.xyz;
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let x = d.x * ch - d.y * sh - camera.inside_c.x;
    let y = d.x * sh + d.y * ch - camera.inside_c.y;
    let z = d.z - camera.inside_c.z;
    let h = camera.inside_b.yzw + vec3<f32>(0.25);
    return select(0.0, 1.0, abs(x) < h.x && abs(y) < h.y && abs(z) < h.z);
}

// How much of the way from the camera to `world` lies outside the player's vehicle: there is
// no fog in the cab. From the driver's seat the whole saloon (seats 2 m away, the doors 5 m
// away) took the fog of that distance and in thick ground fog the interior went milky, the
// door panes nearly opaque - outside the bus there was nothing of the kind. The box is the
// one `inside_vehicle` uses; the part of the ray inside it is taken off the fogged distance.
fn fog_distance(world: vec3<f32>) -> f32 {
    let full = distance(world, camera.cam_pos.xyz);
    if (camera.inside_c.w < 0.5) {
        return full;
    }
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let to_local = fn_box_local(camera.cam_pos.xyz);
    let p1 = fn_box_local(world);
    let d = p1 - to_local;
    let h = camera.inside_b.yzw;
    // slab test of the segment against the box (x, y the plan, z height)
    var t0 = 0.0;
    var t1 = 1.0;
    for (var k = 0; k < 3; k = k + 1) {
        let o = to_local[k];
        let dk = d[k];
        let hk = h[k];
        if (abs(dk) < 1e-6) {
            if (abs(o) > hk) {
                return full;
            }
        } else {
            let a = (-hk - o) / dk;
            let b = (hk - o) / dk;
            t0 = max(t0, min(a, b));
            t1 = min(t1, max(a, b));
        }
    }
    if (t1 <= t0) {
        return full;
    }
    return max(full * (1.0 - (t1 - t0)), 0.0);
}

// A point in the player's vehicle box frame (centred, x across, y along, z up).
fn fn_box_local(world: vec3<f32>) -> vec3<f32> {
    let d = world - camera.inside_a.xyz;
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    return vec3<f32>(d.x * ch - d.y * sh - camera.inside_c.x, d.x * sh + d.y * ch - camera.inside_c.y, d.z - camera.inside_c.z);
}

fn own_vehicle_blocks(p: vec3<f32>, l: vec3<f32>) -> bool {
    if (camera.inside_c.w < 1.5) {
        return false;
    }
    let a = fn_box_local(l);
    let d = fn_box_local(p) - a;
    let h = camera.inside_b.yzw - vec3<f32>(0.15);
    if (all(abs(a) < h)) {
        return false;
    }
    var t0 = 0.0;
    var t1 = 1.0;
    for (var k = 0; k < 3; k = k + 1) {
        let o = a[k];
        let dk = d[k];
        let hk = h[k];
        if (abs(dk) < 1e-6) {
            if (abs(o) >= hk) {
                return false;
            }
        } else {
            let lo = (-hk - o) / dk;
            let hi = (hk - o) / dk;
            t0 = max(t0, min(lo, hi));
            t1 = min(t1, max(lo, hi));
        }
    }
    return t1 > t0;
}

// 1 where the weather reaches the point. The cab test above reaches 0.6 m below the
// vehicle's box so that the floor stays dry, and that takes in the road under the bus as
// well: the snow on it vanished in a bus-shaped patch of bare asphalt (which read as the
// bus's shadow wiping the snow off). The ground - terrain, roads, painted ground, flagged
// by `terrain` or the instance's surface flag - is never inside a vehicle.
fn weather_outside(world: vec3<f32>, terrain: bool, surface: f32) -> f32 {
    if (terrain || surface > 0.5) {
        return 1.0;
    }
    return 1.0 - inside_vehicle(world);
}

// As `weather_outside`, knowing the surface's normal: the bus's own outer skin is outside
// even where it lies inside the box. A side wall leans in towards the roof (and bulges over
// the wheel arches), so the box's side plane cuts through the panel: above the cut the paint
// took the cab's light and stayed dry, below it the sky's - the wall looked half glossy,
// half matt, along a sharp line. A face within 0.35 m of a side, the front, the back or the
// roof of the box and turned out through it is skin, not cabin.
fn weather_outside_n(world: vec3<f32>, n: vec3<f32>, terrain: bool, surface: f32) -> f32 {
    if (terrain || surface > 0.5) {
        return 1.0;
    }
    // a vehicle's part (lib.rs `Instance::roof`): under its roof it is dry, whichever
    // vehicle it is - the one the camera is in, another player's, a timetable bus
    // (what faces up only: taken for every face below the roof, the whole outer skin of
    // every car and bus below the roof's edge was cab - lit by the dim cab light in the
    // enhanced picture, dry in the rain - with a hard seam round the body 0.3 m under the
    // roof where the sky's light began, #805)
    if (surface < -500.0) {
        // (the code carries the roof's snow in its ten thousands, see scene/upload.rs)
        let code = -surface;
        let roof = code - 10000.0 * floor(code / 10000.0) - 5000.0;
        if (world.z < roof - 0.3 && n.z > 0.5) {
            return 0.0;
        }
    }
    if (inside_vehicle(world) < 0.5) {
        return 1.0;
    }
    let d = world - camera.inside_a.xyz;
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let x = d.x * ch - d.y * sh - camera.inside_c.x;
    let y = d.x * sh + d.y * ch - camera.inside_c.y;
    let z = d.z - camera.inside_c.z;
    let nx = n.x * ch - n.y * sh;
    let ny = n.x * sh + n.y * ch;
    let h = camera.inside_b.yzw;
    let skin_x = h.x - abs(x) < 0.35 && nx * sign(x) > 0.5;
    let skin_y = h.y - abs(y) < 0.35 && ny * sign(y) > 0.5;
    let skin_z = h.z - z < 0.35 && n.z > 0.5;
    return select(0.0, 1.0, skin_x || skin_y || skin_z);
}
@group(0) @binding(0) var<uniform> camera: Camera;
// Whether this pipeline draws alpha-tested materials.
// Opaque pipelines set the discard overrides false: a fragment
// function that can discard turns the GPU's early depth test off - on Apple's GPUs the
// hidden surface removal as well - for everything it draws, and the heavy shading then
// ran for every covered layer of the city, not once per pixel.
override ALPHA_TEST: bool = true;
// Multisampled cutout pipelines can turn filtered alpha directly into sample coverage.
override ALPHA_TO_COVERAGE: bool = false;
override DETAILED_TREES: bool = false;
@group(0) @binding(5) var t_shadow: texture_depth_2d;
@group(0) @binding(6) var s_shadow: sampler_comparison;
@group(0) @binding(7) var t_shadow_far: texture_depth_2d;
@group(0) @binding(8) var t_ao: texture_2d<f32>;
@group(0) @binding(9) var s_ao: sampler;

// Enhanced+: the traced lighting at a pixel (rt.wgsl, full size; camera.clouds.w 2): x the
// ambient occlusion, z the sun's visibility (< 0 where it was not traced), w 1 where the
// texel lies at this surface's depth (the pixel's own or a neighbour's; 0: the surface is
// not in the depth prepass - a pane, a blended layer - and the shadow map stands in).
fn rt_at(frag: vec2<f32>, world: vec3<f32>) -> vec4<f32> {
    if (camera.clouds.w < 1.5) {
        return vec4<f32>(1.0, 0.0, -1.0, 0.0);
    }
    let size = vec2<i32>(textureDimensions(t_ao));
    let z = (camera.view_proj * vec4<f32>(world, 1.0)).w;
    let tol = 0.03 + 0.01 * z;
    let px = vec2<i32>(frag);
    var best = vec4<f32>(1.0, 0.0, -1.0, 0.0);
    var best_d = tol;
    for (var k = 0; k < 5; k = k + 1) {
        var o = vec2<i32>(0, 0);
        switch k {
            case 1: { o = vec2<i32>(1, 0); }
            case 2: { o = vec2<i32>(-1, 0); }
            case 3: { o = vec2<i32>(0, 1); }
            case 4: { o = vec2<i32>(0, -1); }
            default: {}
        }
        let s = textureLoad(t_ao, clamp(px + o, vec2<i32>(0), size - vec2<i32>(1)), 0);
        let d = abs(s.g - z);
        if (s.g > 0.0 && d < best_d) {
            best_d = d;
            best = vec4<f32>(s.r, s.g, s.b, 1.0);
        }
        if (k == 0 && d < tol * 0.3) {
            break;
        }
    }
    return best;
}

// The ambient occlusion at a pixel of the full picture. It is worked out at half size, and
// a plain bilinear lookup blended the occlusion of the ground behind an edge with that of
// the object in front: every wheel, pole and kerb stood in a pale outline on the shaded
// ground under the bus. Of the four half-size texels around the pixel only those at the
// pixel's own depth count (a depth-aware upsample); none of them: the closest in depth.
// The light the surfaces round a point send back to it (irradiance, rgb), packed by
// ssao.wgsl `bounce_pack` in the AO texture's b and a.
fn bounce_unpack(t: vec4<f32>) -> vec3<f32> {
    let y = t.b;
    let iu = floor(t.a / 32.0 + 0.001);
    let iv = t.a - iu * 32.0;
    let r = iu / 31.0 * 2.0 * y;
    let b = iv / 31.0 * 2.0 * y;
    let g = max((y - 0.2126 * r - 0.0722 * b) / 0.7152, 0.0);
    return vec3<f32>(r, g, b);
}

// The occlusion (x) and the bounced light (yzw) at a pixel, upsampled by depth as `ao_at`.
fn ao_gi_at(frag: vec2<f32>, world: vec3<f32>, traced: vec4<f32>) -> vec4<f32> {
    if (camera.clouds.w > 1.5) {
        return vec4<f32>(select(traced.x, 1.0, traced.w < 0.5), 0.0, 0.0, 0.0);
    }
    let size = vec2<i32>(textureDimensions(t_ao));
    let z = (camera.view_proj * vec4<f32>(world, 1.0)).w;
    let f = frag * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(f));
    let fr = f - floor(f);
    let tol = 0.04 + 0.015 * z;
    var sum = 0.0;
    var wsum = 0.0;
    var best = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    var best_d = 1e9;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            let c = clamp(base + vec2<i32>(i, j), vec2<i32>(0), size - vec2<i32>(1));
            let s = textureLoad(t_ao, c, 0);
            let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
            let dz = abs(s.g - z);
            let w = bw * (1.0 - smoothstep(0.0, tol, dz)) + 1e-4 * bw;
            sum = sum + s.r * w;
            wsum = wsum + w;
            if (dz < best_d) {
                best_d = dz;
                best = s;
            }
        }
    }
    // no AO sample at this surface's depth: the surface is not in the depth prepass (a
    // door or panel drawn in front of what the AO saw) - the AO behind it must not show
    if (best_d > tol * 1.5) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    // (the bounced light is smooth already - the blur pass averaged it - so the tap at this
    // surface's depth is enough)
    return vec4<f32>(select(best.r, sum / wsum, wsum > 0.02), bounce_unpack(best));
}

fn ao_traced_at(frag: vec2<f32>, world: vec3<f32>, traced: vec4<f32>) -> f32 {
    if (camera.clouds.w > 1.5) {
        return select(traced.x, 1.0, traced.w < 0.5);
    }
    return ao_upsampled(frag, world);
}

fn bounce_at(frag: vec2<f32>, world: vec3<f32>) -> vec3<f32> {
    if (camera.clouds.w > 1.5) {
        return vec3<f32>(0.0);
    }
    let size = vec2<i32>(textureDimensions(t_ao));
    let z = (camera.view_proj * vec4<f32>(world, 1.0)).w;
    let base = vec2<i32>(floor(frag * 0.5 - vec2<f32>(0.5)));
    var best = base;
    var best_d = 1e9;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            let c = clamp(base + vec2<i32>(i, j), vec2<i32>(0), size - vec2<i32>(1));
            let dz = abs(textureLoad(t_ao, c, 0).g - z);
            if (dz < best_d) {
                best_d = dz;
                best = c;
            }
        }
    }
    if (best_d > (0.04 + 0.015 * z) * 1.5) {
        return vec3<f32>(0.0);
    }
    return bounce_unpack(textureLoad(t_ao, best, 0));
}

fn ao_at(frag: vec2<f32>, world: vec3<f32>) -> f32 {
    if (camera.clouds.w > 1.5) {
        let t = rt_at(frag, world);
        return select(t.x, 1.0, t.w < 0.5);
    }
    return ao_upsampled(frag, world);
}

fn ao_upsampled(frag: vec2<f32>, world: vec3<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(t_ao));
    let z = (camera.view_proj * vec4<f32>(world, 1.0)).w;
    let f = frag * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(f));
    let fr = f - floor(f);
    let tol = 0.04 + 0.015 * z;
    var sum = 0.0;
    var wsum = 0.0;
    var best = 1.0;
    var best_d = 1e9;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            let c = clamp(base + vec2<i32>(i, j), vec2<i32>(0), size - vec2<i32>(1));
            let s = textureLoad(t_ao, c, 0);
            let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
            let dz = abs(s.g - z);
            let w = bw * (1.0 - smoothstep(0.0, tol, dz)) + 1e-4 * bw;
            sum = sum + s.r * w;
            wsum = wsum + w;
            if (dz < best_d) {
                best_d = dz;
                best = s.r;
            }
        }
    }
    // no AO sample at this surface's depth: the surface is not in the depth prepass (a
    // door or panel drawn in front of what the AO saw) - the AO behind it must not show
    if (best_d > tol * 1.5) {
        return 1.0;
    }
    return select(best, sum / wsum, wsum > 0.02);
}
// the tile light maps (`.map.LM.bmp`) of the 5x5 tiles around the camera, north up, and where
// they lie: x, y the south-west corner (render-origin relative), z the side, w 1 when set
@group(0) @binding(18) var t_lmap: texture_2d<f32>;
@group(0) @binding(19) var<uniform> lmap: vec4<f32>;

// The tile light map's light at a world point (black outside the loaded square).
fn light_map_at(p: vec3<f32>) -> vec3<f32> {
    if (lmap.w < 0.5) {
        return vec3<f32>(0.0);
    }
    let uv = vec2<f32>((p.x - lmap.x) / lmap.z, 1.0 - (p.y - lmap.y) / lmap.z);
    let c = textureSampleLevel(t_lmap, s_ao, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    let inside = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(vec3<f32>(0.0), c, inside);
}

// A material lit at night by the tile light map (`[LightMapMapping]`, the splines): the
// unlit flag carries 0.35.
fn light_map_mapped(m: vec4<f32>) -> bool {
    return m.y > 0.3 && m.y < 0.45;
}
// (the model matrices as their columns, four vec4 each: some phone GPUs (Mali, Adreno) read an
// array of matrices from a storage buffer wrongly, and every mesh came out flat and far away)
@group(0) @binding(1) var<storage, read> models: array<vec4<f32>>;
// The affine row carries the draw order level and its octahedral geometric normal.
// Neither belongs to the transform used by shading, shadows or motion vectors.
fn model_matrix(e: u32) -> mat4x4<f32> {
    let k = e * 4u;
    return mat4x4<f32>(vec4<f32>(models[k].xyz, 0.0), vec4<f32>(models[k + 1u].xyz, 0.0), vec4<f32>(models[k + 2u].xyz, 0.0), models[k + 3u]);
}
// x: alpha multiplier, y: visible (0/1), zw: uv offset
@group(0) @binding(2) var<storage, read> inst_params: array<vec4<f32>>;
// the frame's draw list: the per-draw entry of each drawn instance. Draws of the same mesh
// and material are batched, so the instance index points into this list, not at an entry.
@group(0) @binding(10) var<storage, read> draw_list: array<u32>;
struct PointLight {
    pos: vec4<f32>,   // xyz relative to the render origin, w radius (0: an enhanced-only light)
    color: vec4<f32>, // rgb, w intensity
    dir: vec4<f32>,   // spot direction, w cosine of the outer cone (< -1.5: a point light)
    extra: vec4<f32>, // enhanced path: cosine of the inner cone, core radius, beam gain, radius
};
@group(0) @binding(3) var<storage, read> lights: array<PointLight>;
// per cell CELL_CAP light indices, 0xffffffff = empty
@group(0) @binding(4) var<storage, read> grid: array<u32>;
const CELL_CAP: u32 = 32u;
// The cutout of `[matl_alpha] 1`: Omsi.exe sets ALPHAREF 0x80 with ALPHAFUNC GREATER, so a
// texel of alpha 128 is thrown away. Painters' window layers (a bus's glass unwrapped on its
// own texture to draw on) mark the clear glass with exactly 128: cut below one half, those
// texels stayed and every window was the layer's black.
const ALPHA_REF: f32 = 128.5 / 255.0;

struct MaterialParams {
    color: vec4<f32>,
    // x: alpha mode (0 opaque, 1 test, 2 blend), y: unlit flag, z: has transmap, w: transmap uses alpha
    params: vec4<f32>,
    // x: terrain material (uv in tile space), y: detail texture repeats per tile (0 = none),
    // z: ground texture repeats per tile, w: has nightmap
    extra: vec4<f32>,
    // x: has lightmap, y: envmap factor, z: moisture, w: has [matl_envmap_mask]
    params2: vec4<f32>,
    // emissive colour (the o3d material's or a [matl_allcolor]'s)
    emissive: vec4<f32>,
    // specular colour and power of the D3D material (power in w; 0 = no highlight)
    specular: vec4<f32>,
    // x: [matl_bumpmap] factor, y: has a bump map, z/w: noZwrite/noZcheck
    bump: vec4<f32>,
    // the PBR set beside the diffuse texture: x normal map, y occlusion, z roughness,
    // w metalness (1 = the map has it; see mx_tex2d[mx_mat.pbr_normal] / mx_tex2d[mx_mat.pbr_orm])
    pbr: vec4<f32>,
    // x: one of the bus's own screens (the enhanced glow and FXAA leave it alone)
    flags: vec4<f32>,
    // rgb: the D3D material's ambient colour, which takes the ambient light (C)
    ambient: vec4<f32>,
    // Windy trees: x 1 for foliage that sways, y the height (mesh units) of its pivot - where
    // the crown leaves the trunk, nothing below it moves - z the crown's top, w how much the
    // tree gives to the wind (1 a broadleaf, less a conifer)
    sway: vec4<f32>,
    // Window wetness mask in mesh X/Z: origin, inverse size; zero disables it.
    wipe_bounds: vec4<f32>,
    // x: [matl_glow] strength (0 = the keyword is not there; the material is its own light -
    // its `mx_tex2d[mx_mat.light]` is a greyscale mask of how much shines where, its colour is its own, see
    // the enhanced pass), y an LED panel's share of the `Led glow` ([led_glow_effect],
    // 1 without it), z/w free
    glow: vec4<f32>,
};
@group(1) @binding(2) var<uniform> material: MaterialParams;
// (the material's textures and samplers by their places in the heap, see `materials_from_heap`)
struct MxMaterial { row: u32, diffuse: u32, sampler: u32, trans: u32, night: u32, light: u32, env: u32, envmask: u32, bump: u32, pbr_normal: u32, pbr_orm: u32, tile_sampler: u32 };
var<immediate> mx_mat: MxMaterial;

// Paint/cut masks and night light maps cover one tile. Wrapping at its edge blends in
// the opposite edge of the SAME tile, opening grass seams even when adjacent masks
// agree. The diffuse/detail textures still repeat through mx_samplers[mx_mat.sampler].
fn sample_transmap(uv: vec2<f32>) -> vec4<f32> {
    return sample_transmap_bias(uv, camera.lod.x);
}

// (`bias`: FSR 1's mip bias for the picture, `camera.lod`; 0 for the shadow and rain maps,
// drawn at their own size)
fn sample_transmap_bias(uv: vec2<f32>, bias: f32) -> vec4<f32> {
    if (material.extra.x > 0.5 || material.wipe_bounds.z != 0.0) {
        return textureSampleBias(mx_tex2d[mx_mat.trans], mx_samplers[mx_mat.tile_sampler], uv, bias);
    }
    return textureSampleBias(mx_tex2d[mx_mat.trans], mx_samplers[mx_mat.sampler], uv, bias);
}

fn sample_nightmap(uv: vec2<f32>) -> vec4<f32> {
    if (material.extra.x > 0.5) {
        return textureSampleBias(mx_tex2d[mx_mat.night], mx_samplers[mx_mat.tile_sampler], uv, camera.lod.x);
    }
    return textureSampleBias(mx_tex2d[mx_mat.night], mx_samplers[mx_mat.sampler], uv, camera.lod.x);
}

// Some vehicle lamps use a clear diffuse placeholder and put the entire lit
// picture, including its silhouette, in the switched night texture. A transmap
// still owns coverage when the model explicitly supplies one.
fn night_coverage(alpha: f32, night_alpha: f32) -> f32 {
    return select(alpha, night_alpha,
        (u32(material.params2.w + 0.5) & 16u) != 0u && material.params.z < 0.5);
}

// The reflection mask of a [matl_envmap] material: the alpha of its [matl_envmap_mask]
// texture when it has one, else the diffuse texture's alpha - which reads 1 for a texture
// without an alpha channel (a 24-bit bitmap, DXT1, a JPEG), so its factor alone decides.
fn has_env_mask() -> bool {
    return (u32(material.params2.w + 0.5) & 1u) != 0u;
}

// [matl_transmap] was given (its file there or not: see MaterialExtra::transmap_declared)
fn has_transmap_declared() -> bool {
    return (u32(material.params2.w + 0.5) & 2u) != 0u;
}

// [matl_texadress_border]: where the diffuse texture's coordinates leave [0, 1] (a roller
// blind's band scrolled away by [texcoordtransX/Y]) Direct3D reads the border colour, not
// the texture's edge (the sampler only clamps). Its rgb comes packed in flags.z.
fn diffuse_border(tex: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let p = u32(material.flags.z + 0.5);
    let border = vec4<f32>(
        f32((p >> 16u) & 0xffu) / 255.0,
        f32((p >> 8u) & 0xffu) / 255.0,
        f32(p & 0xffu) / 255.0,
        material.flags.w,
    );
    let outside = any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0));
    return select(tex, border, material.flags.y > 0.5 && material.flags.y < 1.5 && outside);
}

// [matl_texadress_mirroronce]: Direct3D mirrors the coordinates once about 0 and clamps
// them beyond; the clamping sampler does the rest.
fn tex_address(uv: vec2<f32>) -> vec2<f32> {
    return select(uv, abs(uv), material.flags.y > 1.5);
}

fn reflection_mask(uv: vec2<f32>, diffuse_a: f32) -> f32 {
    let mask = textureSampleBias(mx_tex2d[mx_mat.envmask], mx_samplers[mx_mat.sampler], uv, camera.lod.x).a;
    return select(diffuse_a, mask, has_env_mask());
}

// The D3DCOLOR Omsi.exe packs for its environment stage (0x7ff8ed): each channel the
// [matl_envmap] factor x 255 x its light (at most 1), truncated, and OR-ed together shifted
// into place without saturation - a factor over 1 spills into the neighbouring channel
// exactly as there (a factor of 10 at full light reads almost white).
fn omsi_texture_factor(factor: f32, light: vec3<f32>) -> vec3<f32> {
    let m = min(max(light, vec3<f32>(0.0)), vec3<f32>(1.0)) * max(factor, 0.0) * 255.0;
    let r = u32(m.r);
    let g = u32(m.g);
    let b = u32(m.b);
    let packed = (0xffu << 24u) | (r << 16u) | (g << 8u) | b;
    return vec3<f32>(f32((packed >> 16u) & 0xffu), f32((packed >> 8u) & 0xffu), f32(packed & 0xffu)) / 255.0;
}

// [matl_bumpmap]: Direct3D's bump-mapped environment stage moves the sphere-map lookup by
// a du/dv map times the factor. Every bump map in the stock and mod content is a grey
// height map, so the du/dv map has to be its slope; it is taken here as the forward
// differences of the height, a byte of height difference counting as a signed-byte step
// (1/127, hence the 2). The stock bodies' noise (factor 0.05-0.1) makes their reflections a
// little wavy, and the panel lines of the O530 Facelift's map (factor 1) break its
// reflection at every seam.
fn bump_offset(uv: vec2<f32>) -> vec2<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(mx_tex2d[mx_mat.bump]));
    let h = textureSampleBias(mx_tex2d[mx_mat.bump], mx_samplers[mx_mat.sampler], uv, camera.lod.x).a;
    let hx = textureSampleBias(mx_tex2d[mx_mat.bump], mx_samplers[mx_mat.sampler], uv + vec2<f32>(texel.x, 0.0), camera.lod.x).a;
    let hy = textureSampleBias(mx_tex2d[mx_mat.bump], mx_samplers[mx_mat.sampler], uv + vec2<f32>(0.0, texel.y), camera.lod.x).a;
    return clamp(vec2<f32>(h - hx, h - hy) * 2.0, vec2<f32>(-1.0), vec2<f32>(1.0)) * material.bump.x;
}

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) inst: u32,
};
struct VsOut {
    // invariant: the depth prepass and the main pass must compute the same depth to the
    // last bit, the main pass tests against the depth the prepass left
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    @location(4) @interpolate(flat) params2: vec4<f32>,
    // the D3D material's highlight, lit at the vertex as Omsi.exe's fixed function lights
    // it: from the sun (light A) and from the light above (light B)
    @location(5) spec_sun: vec3<f32>,
    @location(6) spec_sky: vec3<f32>,
    // where a [matl_envmap] reads its sphere map, made at the vertex (see `sphere_map_uv`)
    @location(7) env_uv: vec2<f32>,
    @location(8) wipe_uv: vec3<f32>,
    // The classic headlight is lit at the vertices, as OMSI's fixed function light is.
    @location(9) spot_diffuse: vec3<f32>,
    @location(10) spot_ambient: vec3<f32>,
    // the vertex as its mesh has it: a vehicle's snow lies on the vehicle, not on the
    // world it drives through (road_snow.wgsl `vehicle_snow`)
    @location(11) local: vec3<f32>,
};
// What the fragment shaders take: VsOut without the invariant on the position. The
// invariant belongs to the vertex output; on a fragment input naga's GLSL writer turns it
// into `invariant gl_FragCoord`, which desktop GL drivers and GLES reject.
struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    @location(4) @interpolate(flat) params2: vec4<f32>,
    @location(5) spec_sun: vec3<f32>,
    @location(6) spec_sky: vec3<f32>,
    @location(7) env_uv: vec2<f32>,
    @location(8) wipe_uv: vec3<f32>,
    @location(9) spot_diffuse: vec3<f32>,
    @location(10) spot_ambient: vec3<f32>,
    @location(11) local: vec3<f32>,
};

// Direct3D's D3DTSS_TCI_SPHEREMAP, by which Omsi.exe's environment stage reads the sphere
// map of a [matl_envmap] (0x7ff670): at the vertex, from the reflection R in camera space
// (x right, y up, z ahead), u = Rx/m + 0.5 and v = Ry/m + 0.5 with m = 2|R - (0, 0, 1)| -
// as Windows draws it (wine's d3d9 test_generated_texcoords, measured there). What a pane
// facing the camera mirrors is the map's middle, at half the scale of the u = Rx/2 + 0.5
// taken before (which showed the whole street of the picture across a bus's flank); the
// map's rim is the reflection running on away from the camera, at a grazing angle, where
// the picture swings round between two vertices - the seam the original shows there.
// `eye` is the camera's position, or its mirror image under the road for the player
// vehicle seen in a puddle; the camera's own axes lay the map out either way.
// (The headset: laid out by the bus's heading, level, not by each eye's view.)
fn sphere_map_uv(world: vec3<f32>, n: vec3<f32>, eye: vec3<f32>) -> vec2<f32> {
    let r = reflect(normalize(world - eye), n);
    let vr_env = camera.cam_up.w > 0.5;
    let right = select(camera.cam_right.xyz, vec3<f32>(cos(camera.cam_right.w), -sin(camera.cam_right.w), 0.0), vr_env);
    let up = select(camera.cam_up.xyz, vec3<f32>(0.0, 0.0, 1.0), vr_env);
    let rc = vec3<f32>(dot(r, right), dot(r, up), dot(r, cross(up, right)));
    let m = 2.0 * length(rc - vec3<f32>(0.0, 0.0, 1.0));
    return rc.xy / max(m, 1e-6) + 0.5;
}

// Direct3D's specular term at a vertex (Omsi.exe switches it on in FormActivate, 0x8254e0):
// Omsi.exe's sun (light 0, 0x7089f0: directional, specular = light A) and the light from
// straight above (light 1: specular = light B), each (N.H)^power with the local viewer's
// half vector, only where the light falls on the face, times the material's specular colour
// and clamped at 1 - then interpolated across the face. Computed per pixel instead, every
// small flat part (a gear selector, a switch, a dashboard screen) caught a sharp spot of the
// sun in its middle that the original, lighting only at the corners, never shows.
fn vertex_specular(wp: vec3<f32>, n: vec3<f32>) -> array<vec3<f32>, 2> {
    var out = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    if (material.specular.w <= 0.0 || material.params.y >= 0.5) {
        return out;
    }
    let v = normalize(camera.cam_pos.xyz - wp);
    let p = material.specular.w;
    let l = camera.sun_dir.xyz;
    if (dot(n, l) > 0.0) {
        out[0] = camera.sun_color.rgb * camera.sun_dir.w * pow(max(dot(n, normalize(v + l)), 0.0), p);
    }
    let up = vec3<f32>(0.0, 0.0, 1.0);
    if (n.z > 0.0) {
        out[1] = camera.sky_color.rgb * pow(max(dot(n, normalize(v + up)), 0.0), p);
    }
    let total = out[0] + out[1];
    let k = min(vec3<f32>(1.0), total) / max(total, vec3<f32>(1e-6));
    out[0] = min(vec3<f32>(1.0), out[0] * k * material.specular.rgb);
    out[1] = min(vec3<f32>(1.0), out[1] * k * material.specular.rgb);
    return out;
}

// Windy trees: how far the wind moves a vertex of a tree's foliage (world, m). `local` is
// the vertex in the mesh, `m` its model matrix, whose origin is the tree's foot.
//
// The tree is bent, not shifted: nothing under its pivot (`material.sway.y`, where the crown
// leaves the trunk) moves, and above it the crown bends over as a cantilever does, by the
// square of the height above the pivot, sinking a little so that its branches keep their
// length. The crown moves as one heavy body - a tree is a lightly damped oscillator, and
// what it does is smooth: no part of the picture moves on its own (each corner of a card
// wobbling in its own phase made the crown jelly).
//
// What bends it is the drag of the air: it grows with the wind speed to the power 1.5 rather
// than 2, as a crown streamlines itself (Vogel's reconfiguration exponent), from a few
// millimetres at 1 m/s to some 3.5 % of the tree's height at 20 m/s, a storm. The gusts are
// a frozen pattern of faster and slower air carried along at the mean wind (Taylor): three
// smooth waves 54 .. 149 m long, a turbulence intensity of 0.3 as over a town, so that
// neighbouring trees bow together as a gust passes and the ones further on follow - and,
// being smooth, the gusts never stop the tree dead as the corners of a value noise did.
// Around that lean the tree swings at its own natural frequency (0.2 .. 1 Hz, lower the
// taller it is), its amplitude and phase wandering slowly as a resonance driven by
// turbulence does, and a little across the wind; the crown's top lags its middle a little.
// With no wind nothing moves.
fn tree_sway(local: vec3<f32>, m: mat4x4<f32>) -> vec3<f32> {
    let sw = material.sway;
    let wind = camera.tree_wind.xy;
    let speed = length(wind);
    if (sw.x < 0.5 || speed < 0.05) {
        return vec3<f32>(0.0);
    }
    let h = clamp((local.z - sw.y) / max(sw.z - sw.y, 1e-4), 0.0, 1.0);
    if (h <= 0.0) {
        return vec3<f32>(0.0);
    }
    let scale = length(m[2].xyz);
    let crown = max((sw.z - sw.y) * scale, 0.1);
    let height = max(sw.z * scale, crown);
    let foot = (m * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let foot_map = world_pattern_xy(foot);
    let t = camera.post.y;
    // each tree its own phase
    let seed = hash_cell(floor(foot_map * 2.0), 2000.0);
    let phase = seed * 6.2831853;
    // the gusts at the tree: whole waves per PATTERN_PERIOD (the map position and the
    // drift arrive modulo it), 149, 93 and 54 m long - a gust lasts some seconds; their sum has a standard deviation of
    // 0.51 (0.6, 0.35, 0.2 over sqrt 2), scaled to the turbulence intensity 0.3
    let gp = (foot_map - camera.tree_wind.zw) * (6.2831853 / PATTERN_PERIOD);
    let waves = 0.6 * sin(dot(gp, vec2<f32>(6.0, 3.0)) + 1.3)
        + 0.35 * sin(dot(gp, vec2<f32>(-4.0, 10.0)) + 4.1)
        + 0.2 * sin(dot(gp, vec2<f32>(17.0, -7.0)) + 2.2);
    let gust = max(1.0 + 0.59 * waves, 0.0);
    let along = wind / speed;
    let across = vec2<f32>(-along.y, along.x);
    // the lean under the gusting wind, and the tree's swing about it
    let give = 0.035 * height * sw.w;
    let lean = give * pow(speed * gust / 20.0, 1.5);
    // (in strong wind the swing is of the order of the mean lean itself)
    let swing_size = 0.7 * give * pow(speed / 20.0, 1.5);
    let f0 = clamp(1.6 / sqrt(height), 0.2, 1.0);
    let w0 = 6.2831853 * f0 * t + phase + 0.8 * sin(0.17 * t + 9.0 * seed);
    let envelope = 0.65 + 0.35 * sin(0.23 * t + 5.0 * seed);
    let swing = swing_size * envelope;
    // (the top lags the middle by a fraction of a cycle: the crown bends, not only leans)
    let lag = 0.35 * h;
    let move_xy = along * (lean + swing * sin(w0 - lag)) + across * (0.35 * swing * sin(0.93 * w0 + 1.7 - lag));
    var d = vec3<f32>(move_xy * h * h, 0.0);
    // the crown's big boughs: lobes about the crown's size swaying at about twice the tree's
    // frequency, each with its own slowly wandering phase, harder in a gust - neighbouring
    // parts of the picture move nearly together, so the crown stirs rather than wobbles
    let lm = (m * vec4<f32>(local, 0.0)).xyz;
    let bp = dot(lm, vec3<f32>(0.45, 0.35, 0.3) * (4.0 / crown)) + phase;
    let fb = clamp(2.2 * f0, 0.8, 2.0);
    let wb = 6.2831853 * fb * t + bp + 0.6 * sin(0.31 * t + bp);
    let bough = 0.25 * give * pow(speed * gust / 20.0, 1.5) * h;
    d = d + vec3<f32>(along * sin(wb) + across * 0.5 * sin(1.17 * wb + 2.0), 0.25 * sin(0.9 * wb + 1.0)) * bough;
    // the bent crown keeps its length: it sinks as it leans
    let above = max((local.z - sw.y) * scale, 0.3);
    d.z = d.z - 0.5 * dot(d.xy, d.xy) / above;
    return d;
}

// Where a scene vertex lands on the screen: the main pass's and the depth passes' vertex
// shaders (`vs_main`, `vs_depth`) both take it from here, so that their depths agree to the
// last bit (the main pass tests against the depth the prepass left).
fn scene_clip(e: u32, wp: vec4<f32>, nw: vec3<f32>) -> vec4<f32> {
    // Road surfaces (splines, the objects lying on them, a shadow blob) are pulled towards
    // the eye along the line of sight - the picture does not move, only the depth - so that
    // they win over the flush ground beside them. Code 0.9 keeps the surface shading without
    // the pull; painted ground (0.75) stays put.
    let surf = inst_params[e * 2u + 1u].w;
    var cp = wp.xyz;
    if (surf > 0.9) {
        let to = wp.xyz - camera.cam_pos.xyz;
        let d = length(to);
        // The same share of the way to the eye at every vertex (0.3 %, a few millimetres of
        // height seen from the cab): the pulled road is still the plane it was, so a spline
        // laid a millimetre over another - the rails or markings over a road - stays over it
        // however the two are cut into triangles. A fixed 2 cm more along each vertex's own
        // line of sight bowed the road's long triangles over the short ones of what lies on
        // it, by millimetres near the eye, and the rails went under the road there (#1196).
        // (surface objects, 1.25, a little more than the splines under them)
        // (a vehicle's shadow blob, 2: drawn without a depth test in the original, right
        // over the road it lies on - with the road's own pull alone the two fought for
        // every pixel and the road mostly won; a few centimetres more bring it through the
        // road and still leave it behind the body and the wheels standing on it)
        // The share stops growing at 5 cm. The pull is a depth, not a place - the picture
        // does not move - but the pole of a traffic light that a mod's map tagged as a
        // surface is pulled with the road under it, and a share still growing at distance
        // (a third of a metre at 90 m) carried the pole through the signal head it stands
        // beside. Capped, the pole stays behind what it stands by, and the depth a road
        // needs over flush ground is a few millimetres at most (a floor raised 3 cm above
        // the road is still seen - `surface_bias_keeps_raised_floors_visible_at_a_distance`).
        // (a road's painted lines, 1.4, a little more again: a crossing object laid over
        // the road covered the lines it overlaps, which in Omsi.exe lie a millimetre over
        // the road and win the depth test against it - #1694)
        let decal = surf > 1.1 && surf < 1.35;
        let paint = surf > 1.35 && surf < 1.5;
        let share = select(select(select(0.003, 0.004, decal), 0.0045, paint), 0.005, surf > 1.5);
        let fixed = select(select(select(0.0, 0.01, decal), 0.015, paint), 0.05, surf > 1.5);
        // The cap on the share is for upright faces, the poles. A flat surface keeps its
        // share all the way: pulled by the same part of the way to the eye at every vertex
        // the road is still one plane, however long its triangles, and a marking on it keeps
        // its place. Capped at 5 cm the pull of a triangle from the eye to the horizon
        // stopped growing with the distance and turned the road's plane into a bowl: at
        // 40 m from a low eye the road came in front of the lines on it, and they were
        // gone, and back as the bus came on (#3085). (Half a metre of it, at 170 m and
        // beyond: millimetres of height from a driver's eye.)
        let flat = abs(safe_normal(nw).z) > 0.5;
        let pull = min(min(share * d, select(0.05, 0.5, flat)) + fixed, d * 0.3);
        cp = wp.xyz - to / max(d, 1e-3) * pull;
    }
    // Omsi.exe draws a model mesh by mesh and subset by subset with LESSEQUAL into a 24-bit
    // depth buffer: of two of its surfaces in one plane - a dashboard and the instruments
    // printed on it - the later one is seen. Here their depths differed by rounding alone
    // and every pixel was a coin toss. A slot lying on an earlier surface of its object
    // (`draw_order_level`, 1 to 3; everything else 0) is brought nearer by two steps, each
    // the same over a whole plane however it is cut into triangles (a step worked out per
    // vertex from its own distance bends the surface): the reversed depth times 1 + 1e-5 a
    // level, above the vertices' rounding; and 2e-5 a level over the plane's distance from
    // the eye, against the tilt the rasterizer's snapping of each corner to a fraction of a
    // pixel gives a triangle's depth, the more the flatter it is seen. Seen from above the
    // plane that lifts it by 3e-5 a level times the distance: 0.9 mm at 10 m at level 3.
    // (The plane's distance is at least 5 % of the vertex's own: the edge of a curved mesh,
    // whose tangent plane runs through the eye, is not pulled metres forward.)
    let clip = camera.view_proj * vec4<f32>(cp, 1.0);
    let level_code = u32(round(models[e * 4u].w));
    var level = f32(level_code & 3u);
    if (level <= 0.0) {
        return clip;
    }
    if ((level_code & 4096u) != 0u) {
        // Keep the depth step within half the measured gap.
        let cap = f32((level_code >> 2u) & 1023u) * 1e-6;
        let distance = length(cp - camera.cam_pos.xyz);
        level = min(level, cap / max(3e-5 * distance, 1e-9));
        if (level <= 0.0) {
            return clip;
        }
    }
    // (the depth of the eye's own point: n f / (f - n), about the near plane)
    let at_eye = (camera.view_proj * vec4<f32>(camera.cam_pos.xyz, 1.0)).z;
    // Use geometric plane normals; lighting normals may be smoothed or authored independently.
    let encoded = vec2<f32>(models[e * 4u + 1u].w, models[e * 4u + 2u].w);
    var depth_n = nw;
    if (all(abs(encoded) <= vec2<f32>(1.0))) {
        var n = vec3<f32>(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
        let t = clamp(-n.z, 0.0, 1.0);
        n.x += select(t, -t, n.x >= 0.0);
        n.y += select(t, -t, n.y >= 0.0);
        depth_n = n;
    }
    let plane = max(abs(dot(safe_normal(depth_n), cp - camera.cam_pos.xyz)), 0.05 * length(cp - camera.cam_pos.xyz));
    let slant = level * 2e-5 * at_eye * clip.w / plane;
    return vec4<f32>(clip.x, clip.y, clip.z * (1.0 + level * 1e-5) + slant, clip.w);
}

fn is_tree_card() -> bool {
    return material.params.y > 0.1 && material.params.y < 0.2;
}

fn detailed_trees() -> bool {
    return DETAILED_TREES;
}

fn tree_crown_normal(local: vec3<f32>, m: mat4x4<f32>, up: vec3<f32>) -> vec3<f32> {
    let pivot = select(0.28, material.sway.y, material.sway.x > 0.5);
    let centre = mix(pivot, select(1.0, material.sway.z, material.sway.x > 0.5), 0.45);
    let out = (m * vec4<f32>(local.x, local.y, local.z - centre, 0.0)).xyz;
    let crown = smoothstep(pivot * 0.5, pivot + 0.1, local.z);
    return safe_normal(mix(up, safe_normal(out), 0.65 * crown));
}

fn tree_card_fade(world: vec3<f32>) -> f32 {
    if (!detailed_trees() || (!is_tree_card() && material.sway.x < 0.5)) {
        return 1.0;
    }
    let c = cross(dpdx(world), dpdy(world));
    let v = (camera.cam_pos.xyz - world).xy;
    if (dot(c, c) < 1e-20 || dot(v, v) < 1e-6) {
        return 1.0;
    }
    let n = normalize(c);
    let upright = smoothstep(0.6, 0.9, length(n.xy));
    let facing = abs(dot(normalize(n.xy + vec2<f32>(1e-6, 0.0)), normalize(v)));
    return mix(1.0, smoothstep(0.3, 0.62, facing), upright);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    let wp = m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0);
    var out: VsOut;
    out.clip = scene_clip(e, wp, (m * vec4<f32>(in.normal, 0.0)).xyz);
    out.world = wp.xyz;
    out.local = in.pos;
    out.normal = safe_normal((m * vec4<f32>(in.normal, 0.0)).xyz);
    if (detailed_trees() && (is_tree_card() || (material.sway.x > 0.5 && material.sway.z - material.sway.y > 2.5))) {
        out.normal = tree_crown_normal(in.pos, m, out.normal);
    }
    let sp = vertex_specular(wp.xyz, out.normal);
    out.spec_sun = sp[0];
    out.spec_sky = sp[1];
    // Ray blocking is disabled for exterior cameras; never light our own body or cabin.
    // The vehicle surface code keeps roads and scenery in the same box illuminated.
    out.spot_diffuse = vec3<f32>(0.0);
    out.spot_ambient = vec3<f32>(0.0);
    let own_vehicle = inst_params[e * 2u + 1u].w < -500.0 && near_player_vehicle(wp.xyz) > 0.5;
    if (!own_vehicle) {
        let spot = classic_spot(wp.xyz, out.normal);
        out.spot_diffuse = spot[0];
        out.spot_ambient = spot[1];
    }
    out.env_uv = sphere_map_uv(wp.xyz, out.normal, camera.cam_pos.xyz);
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    out.params2 = inst_params[e * 2u + 1u];
    let side_pane = material.wipe_bounds.z < 0.0;
    let pane = select(in.pos.xz, in.pos.yz, side_pane);
    out.wipe_uv = vec3<f32>((pane - material.wipe_bounds.xy) * abs(material.wipe_bounds.zw), select(in.pos.y, in.pos.x, side_pane));
    if (pr.y < 0.5) {
        // invisible: collapse the triangle
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

// What the depth-only passes (the depth prepass, the multisampled prefill) take from a
// vertex: its place and what a cut-out or a transmap needs to decide its coverage. The
// highlights, the sphere map and the normal of `vs_main` are not worked out for them.
struct DepthOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    // the instance's surface code (`inst_params` w, see `scene_clip`)
    @location(4) surf: f32,
    @location(5) world: vec3<f32>,
};
struct DepthIn {
    @builtin(position) clip: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    @location(4) surf: f32,
    @location(5) world: vec3<f32>,
};

struct ShadowOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) params: vec4<f32>,
};
struct ShadowIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) params: vec4<f32>,
};

@vertex
fn vs_depth(in: VsIn) -> DepthOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    let wp = m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0);
    var out: DepthOut;
    out.clip = scene_clip(e, wp, (m * vec4<f32>(in.normal, 0.0)).xyz);
    out.world = wp.xyz;
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    out.surf = inst_params[e * 2u + 1u].w;
    if (pr.y < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

struct VehicleBox { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
struct VehicleReflection { plane: vec4<f32>, parts: array<VehicleBox, 4> };
@group(2) @binding(0) var<uniform> vehicle_reflection: VehicleReflection;

@vertex
fn vs_puddle_vehicle(in: VsIn) -> VsOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    var out: VsOut;
    out.world = (m * vec4<f32>(in.pos, 1.0)).xyz;
    out.normal = safe_normal((m * vec4<f32>(in.normal, 0.0)).xyz);
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    out.params2 = inst_params[e * 2u + 1u];
    out.spec_sun = vec3<f32>(0.0);
    out.spec_sky = vec3<f32>(0.0);
    out.spot_diffuse = vec3<f32>(0.0);
    out.spot_ambient = vec3<f32>(0.0);
    let plane = vehicle_reflection.plane;
    // the sphere map from the reflected eye fs_puddle_vehicle shades the vehicle from
    out.env_uv = sphere_map_uv(out.world, out.normal, camera.cam_pos.xyz - 2.0 * plane.xyz * (dot(plane.xyz, camera.cam_pos.xyz) - plane.w));
    let reflected = out.world - 2.0 * plane.xyz * (dot(plane.xyz, out.world) - plane.w);
    out.clip = camera.view_proj * vec4<f32>(reflected, 1.0);
    if (out.params.y < 0.5) { out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); }
    return out;
}

// Close an open legacy underbody with one depth-tested face in the same reflected
// view. Wheels and authored panels occlude it normally; no screen-space bands.
@vertex
fn vs_puddle_chassis(@builtin(vertex_index) i: u32, @builtin(instance_index) part: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let box = vehicle_reflection.parts[part];
    let local = corners[i] * max(box.b.yz - vec2<f32>(0.12, 0.18), vec2<f32>(0.0)) + box.c.xy;
    let sh = box.a.w;
    let ch = box.b.x;
    let xy = box.a.xy + vec2<f32>(local.x * ch + local.y * sh, -local.x * sh + local.y * ch);
    let plane = vehicle_reflection.plane;
    let road_z = (plane.w - dot(plane.xy, xy)) / plane.z;
    let z = max(road_z + 0.18, box.a.z + box.c.z - box.b.w + 0.04);
    let world = vec3<f32>(xy, z);
    let reflected = world - 2.0 * plane.xyz * (dot(plane.xyz, world) - plane.w);
    return camera.view_proj * vec4<f32>(reflected, 1.0);
}

// Shadow map passes: depth from the sun, alpha-tested materials cut out by their texture.
// A surface that casts - a spline standing clear of the ground, a bridge deck - casts from
// half a metre further away from the sun: what lies right under it (the embankment object
// under a railway, the ground a ramp touches down on) is no darker for it, while the ground
// metres below a deck gets its shadow.
fn shadow_caster_pos(e: u32, wp: vec4<f32>) -> vec4<f32> {
    if (abs(inst_params[e * 2u + 1u].w - 1.0) < 0.01) {
        return vec4<f32>(wp.xyz - camera.sun_dir.xyz * 0.5, wp.w);
    }
    return wp;
}

@vertex
fn vs_shadow(in: VsIn) -> ShadowOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    let wp = shadow_caster_pos(e, m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0));
    var out: ShadowOut;
    out.clip = camera.light_view_proj * wp;
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    if (pr.y < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_shadow_close(in: VsIn) -> ShadowOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    let wp = shadow_caster_pos(e, m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0));
    var out: ShadowOut;
    out.clip = camera.light_view_proj_close * wp;
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    if (pr.y < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_shadow_far(in: VsIn) -> ShadowOut {
    let e = draw_list[in.inst];
    let m = model_matrix(e);
    let wp = shadow_caster_pos(e, m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0));
    var out: ShadowOut;
    out.clip = camera.light_view_proj_far * wp;
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    if (pr.y < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

// Enhanced: a lamp's shadow map (one of the tiles under the far map). The draw list names
// the tile above the entry's own bits (`LAMP_ENTRY_BITS` in lamp_slots.rs), and the tile's
// matrix looks down from a street lamp's head or along the road from a vehicle's headlamps.
const LAMP_ENTRY_BITS: u32 = 26u;

@vertex
fn vs_shadow_lamp(in: VsIn) -> ShadowOut {
    let raw = draw_list[in.inst];
    let e = raw & ((1u << LAMP_ENTRY_BITS) - 1u);
    let k = raw >> LAMP_ENTRY_BITS;
    let m = model_matrix(e);
    let wp = m * vec4<f32>(in.pos, 1.0) + vec4<f32>(tree_sway(in.pos, m), 0.0);
    var out: ShadowOut;
    out.clip = camera.lamp_view_proj[k] * wp;
    let pr = inst_params[e * 2u];
    out.uv = in.uv + pr.zw;
    out.params = pr;
    if (pr.y < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

@fragment
fn fs_shadow(in: ShadowIn) {
}

// Reflection rays need the window surface as well as the opaque interior behind it.
// This writes a private hit-depth texture after colour compositing; ordinary depth and
// AO still leave glass transparent. Reject absent/fully faded layers and texture holes.
@fragment
fn fs_puddle_glass_depth(in: FsIn) {
    // From the cabin we see the street through our own pane. Treating that pane as
    // an SSR hit terminates every ray at the window instead of the outside vehicle.
    if (inside_vehicle(camera.cam_pos.xyz) * near_player_vehicle(in.world) > 0.5) {
        discard;
    }
    var a = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], tex_address(in.uv), camera.lod.x), in.uv).a;
    if (material.params.z > 0.5) {
        let tm = sample_transmap(tex_address(in.uv - in.params.zw));
        a = select(1.0, tm.a, material.params.w > 0.5);
    }
    if (a * material.color.a * in.params.x < 0.002) {
        discard;
    }
}

// (the shadow and rain maps, drawn at their own size: their textures' mips as they are)
@fragment
fn fs_shadow_test(in: ShadowIn) {
    cutout_coverage(in.uv, in.params, 0.0, 1.0);
}

// the depth-only passes' (`vs_depth`) fragment shaders: plain, cut out, a transmap's
// opaque part
@fragment
fn fs_depth(in: DepthIn) {
}

// (the picture's depth: its textures read with the colour pass's mip bias, `camera.lod` -
// a cut-out's edge the prepass took from a blurrier mip than the colour pass wrote depth
// where the colour then discarded, and the sky showed through)
@fragment
fn fs_depth_test(in: DepthIn) {
    cutout_coverage(in.uv, in.params, camera.lod.x, tree_card_fade(in.world));
}

// (a transmap's opaque part, see `transmap_coverage`)
@fragment
fn fs_depth_transmap(in: DepthIn) {
    transmap_coverage(in.uv, in.params, camera.lod.x);
}

//MS @group(3) @binding(0) var t_pane_depth: texture_depth_multisampled_2d;

//MS @fragment
//MS fn fs_depth_masked(in: DepthIn, @builtin(sample_index) sample: u32) {
//MS     if (in.clip.z <= textureLoad(t_pane_depth, vec2<i32>(in.clip.xy), i32(sample))) {
//MS         discard;
//MS     }
//MS }

// The multisampled prefill's cut-outs (foliage): their depth only where the colour pass
// is sure to cover every sample - the alpha-to-coverage of an alpha of 1 (`shade_enhanced`:
// the texture's alpha past the cutoff's smoothed edge, a plain material at full opacity,
// no pane, LED panel, transmap, ground or shadow blob). The edges, which alpha-to-coverage
// covers in part, are left to the colour pass: a sample laid here that the colour then
// left uncovered would hide what lies behind it (the sky showed through the houses).
@fragment
fn fs_depth_cover(in: DepthIn) {
    let mode = material.params.x;
    if (mode < 0.5 || mode > 1.5 || material.params.z > 0.5 || material.extra.x > 0.5
        || material.wipe_bounds.z != 0.0 || abs(material.emissive.w) > 1.5
        || material.color.a < 1.0 || in.params.x < 1.0 || in.surf > 1.5) {
        discard;
    }
    let duv = tex_address(in.uv);
    let a = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, camera.lod.x), duv).a * tree_card_fade(in.world);
    let aa = max(fwidth(a) * 0.5, 1.0 / 255.0);
    if (a < ALPHA_REF + aa + 4.0 / 255.0) {
        discard;
    }
}

// A cut-out's texel: kept where its alpha reaches the cutoff (`bias`: the mip bias its
// pass reads the textures with, the colour pass's for the picture's depth).
fn cutout_coverage(uv: vec2<f32>, params: vec4<f32>, bias: f32, fade: f32) {
    var duv = tex_address(uv);
    if (material.extra.x > 0.5) {
        duv = uv * material.extra.z;
    }
    // Use the light-pass footprint for cutout coverage. Forcing mip 0 here aliases dense
    // foliage and alpha layers into a checkerboard when the light projection moves by a
    // texel; the mip-aware sample keeps the caster coverage coherent with its resolution.
    // For a material promoted to the shadow pass from Blend, diffuse alpha is commonly a
    // paint/gloss value rather than coverage (vehicle bodies use values such as 0.03). Only
    // alpha-test materials use diffuse alpha as a cutout; blended bodies are solid unless a
    // real transmap supplies coverage.
    var a = select(diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, bias), duv).a, 1.0, material.params.x > 1.5 && material.params.z < 0.5);
    if (material.params.z > 0.5) {
        // (the transmap stays where it is: [texcoordtransX/Y] only moves the diffuse stage)
        let tm = sample_transmap_bias(tex_address(uv - params.zw), bias);
        a = select(1.0, tm.a, material.params.w > 0.5);
    }
    if (a * fade < ALPHA_REF) {
        discard;
    }
}

// Roads with feathered alpha borders stay blended while their overlaps compose.
// Once the surface phases are complete, only fully covered diffuse pixels occlude
// later scenery. The transparent borders must not become invisible depth walls.
@fragment
fn fs_surface_depth(in: FsIn) -> @location(0) vec4<f32> {
    let a = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], tex_address(in.uv), camera.lod.x), in.uv).a
        * material.color.a * in.params.x;
    if (a < 0.999) {
        discard;
    }
    return vec4<f32>(0.0);
}

// A vehicle body and its windows are often one mesh/material. OMSI represents that
// material as alpha-blended because the transmap contains the window opacity, but the
// alpha-tested body pixels still have to occlude vehicles behind them. The normal depth prepass
// deliberately skips all blended materials, so those pixels would otherwise have no depth
// until the colour pass (where blend order can expose the bus through the car). Keep only
// the effectively opaque part of a transmap in a separate depth-only pass; window pixels
// remain out of the prepass and are composited normally.
fn transmap_coverage(uv: vec2<f32>, params: vec4<f32>, bias: f32) {
    if (material.params.z < 0.5) {
        discard;
    }
    let tm = sample_transmap_bias(tex_address(uv - params.zw), bias);
    let a = select(1.0, tm.a, material.params.w > 0.5) * params.x;
    // Only what the colour pass will cover completely may hide what lies behind it: a
    // texel that is merely more opaque than not (the dimmer and anti-aliased dots of a
    // display's text layer, whose transmap is its script texture) wrote depth here, the
    // display's backplate behind it was then rejected, and the half-transparent text was
    // blended over the sky - holes in the display.
    if (a < 0.99) {
        discard;
    }
}

// 1 = lit by the sun, 0 = in shadow. Three cascades: the close one (the bus and what is
// next to it), the near one around the camera and the far one covering the rest of the
// visible street.

// Depth bias of the sun shadow, in metres along the sun's ray. The maps' depth runs over
// the light box's 2199 m, and the comparison used to take off 0.0015 (near) and 0.004 (far)
// of that: 3.3 m and 8.8 m. Nothing closer to the ground than that along the ray cast a
// shadow - not a car (1.4 m), not a bus (3 m; only its roof's shadow at a low sun,
// which lay detached and offset from the bus) - and the grass and kerbs had none either.
// The receiver plane below takes care of what the big constant stood in for.
const SHADOW_DEPTH_RANGE: f32 = 2199.0;
const SHADOW_BIAS_NEAR: f32 = 0.06;
const SHADOW_BIAS_FAR: f32 = 0.3;
const SHADOW_BIAS_CLOSE: f32 = 0.025;

// How the receiver's depth in the shadow map changes across the map (per unit of uv), from
// its normal: the PCF taps around the pixel compare against the surface's own depth at the
// tap instead of the centre's, so a surface at a slant to the sun does not shadow itself
// within the filter radius (acne), which a big constant bias used to hide. `lvp` is the
// cascade's light matrix (orthographic, so the map from the world is affine).
fn shadow_receiver_slope(lvp: mat4x4<f32>, n: vec3<f32>) -> vec2<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t1 = normalize(cross(n, a));
    let t2 = cross(n, t1);
    let p1 = (lvp * vec4<f32>(t1, 0.0)).xyz;
    let p2 = (lvp * vec4<f32>(t2, 0.0)).xyz;
    // (uv = 0.5 x + 0.5, 0.5 - 0.5 y)
    let d1 = vec3<f32>(p1.x * 0.5, -p1.y * 0.5, p1.z);
    let d2 = vec3<f32>(p2.x * 0.5, -p2.y * 0.5, p2.z);
    let det = d1.x * d2.y - d2.x * d1.y;
    if (abs(det) < 1e-12) {
        return vec2<f32>(0.0);
    }
    let g = vec2<f32>(d1.z * d2.y - d2.z * d1.y, d1.x * d2.z - d2.x * d1.z) / det;
    // a surface seen edge-on by the sun: keep the slope finite (it is in shadow of itself
    // there anyway, by its normal)
    return clamp(g, vec2<f32>(-2.0), vec2<f32>(2.0));
}

// The filter's half width in texels. A tent of 2 texels covers 4 x 4 texels, two gathers a
// side - four in all, fewer texture reads than the Poisson kernel's five (sixteen on an
// edge) - and spreads an edge about as the old kernel did in the near and close cascades
// (0.82 of a texel's standard deviation against 0.9 and 1.0; a fifth sharper in the far one).
const SHADOW_TENT: f32 = 2.0;
// Percentage-closer filtering with a tent of `SHADOW_TENT` texels, every texel of the map
// under it compared and weighted by the tent at its centre: four gathers of 2 x 2 texels,
// the whole 4 x 4 footprint read. The sparse Poisson kernel this replaces (16 bilinear taps
// 1.6 to 2.2 texels apart, and only its 5 outermost-and-middle taps when those agreed)
// sampled the map coarser than its texels: a lit gap between leaves, a few texels wide, came
// out as a copy of the kernel - an X of five magnified texels - and a feature lying between
// the taps vanished and reappeared as the receiver moved by a texel, which made the shadows
// shimmer. Read in full, the result is a continuous function of the receiver's position.
// `widen` (Enhanced: a veil of high cloud enlarging the sun) spreads the four gathers that
// many texels apart, a tent as much wider, without reading more.
// `uv`: the receiver in the cascade's own 0..1; `size`: the cascade's texels a side;
// `origin`: where its first texel lies in the texture; `slope`: its receiver plane (per uv).
// The shadow maps' size, texels a side, from the uniforms (`camera.shadow.y` is a texel): the
// near map an atlas of two cascades side by side, the far one taller by the lamps' tiles under
// it (`camera.lamp_atlas.z` is its texture's height). (Asked of the textures with
// textureDimensions in every inlined use of the sun's shadow, they brought Apple's M4 shader
// compiler down on every scene pipeline.)
fn shadow_map_size() -> f32 {
    return 1.0 / max(camera.shadow.y, 1e-6);
}

fn shadow_tent_place(uv: vec2<f32>, size: f32, widen: f32) -> vec4<f32> {
    let k = max(1.0, floor(widen + 0.5));
    // (texel i of the cascade has its centre at p = i; the first texel the tent reaches)
    let p = uv * size - vec2<f32>(0.5);
    let lo = floor(p - vec2<f32>(SHADOW_TENT * k)) + vec2<f32>(1.0);
    return vec4<f32>(p, lo);
}

// The tent's weights of a gather's four texels starting at `t0` (gather order: (0, 1),
// (1, 1), (1, 0), (0, 0) of the quad).
fn shadow_tent_weights(t0: vec2<f32>, p: vec2<f32>, r: f32) -> vec4<f32> {
    let w0 = max(vec2<f32>(0.0), vec2<f32>(1.0) - abs(t0 - p) / r);
    let w1 = max(vec2<f32>(0.0), vec2<f32>(1.0) - abs(t0 + vec2<f32>(1.0) - p) / r);
    return vec4<f32>(w0.x * w1.y, w1.x * w1.y, w1.x * w0.y, w0.x * w0.y);
}

fn shadow_tent_near(uv: vec2<f32>, size: f32, origin: vec2<f32>, z: f32, slope: vec2<f32>, widen: f32, bias: f32) -> f32 {
    let dims = vec2<f32>(2.0, 1.0) * shadow_map_size();
    let pl = shadow_tent_place(uv, size, widen);
    let k = max(1.0, floor(widen + 0.5));
    let r = SHADOW_TENT * k;
    var sum = 0.0;
    var wsum = 0.0;
    for (var q = 0; q < 4; q = q + 1) {
        let t0 = pl.zw + vec2<f32>(f32(q & 1), f32(q >> 1u)) * 2.0 * k;
        let a = (origin + t0 + vec2<f32>(1.0)) / dims;
        let zr = z + dot(slope, (t0 + vec2<f32>(0.5) - pl.xy) / size) - bias;
        let w = shadow_tent_weights(t0, pl.xy, r);
        sum = sum + dot(textureGatherCompare(t_shadow, s_shadow, a, zr), w);
        wsum = wsum + w.x + w.y + w.z + w.w;
    }
    return sum / max(wsum, 1e-6);
}

fn shadow_tent_far(uv: vec2<f32>, size: f32, z: f32, slope: vec2<f32>, widen: f32, bias: f32) -> f32 {
    let dims = vec2<f32>(shadow_map_size(), camera.lamp_atlas.z);
    let pl = shadow_tent_place(uv, size, widen);
    let k = max(1.0, floor(widen + 0.5));
    let r = SHADOW_TENT * k;
    var sum = 0.0;
    var wsum = 0.0;
    for (var q = 0; q < 4; q = q + 1) {
        let t0 = pl.zw + vec2<f32>(f32(q & 1), f32(q >> 1u)) * 2.0 * k;
        let a = (t0 + vec2<f32>(1.0)) / dims;
        let zr = z + dot(slope, (t0 + vec2<f32>(0.5) - pl.xy) / size) - bias;
        let w = shadow_tent_weights(t0, pl.xy, r);
        sum = sum + dot(textureGatherCompare(t_shadow_far, s_shadow, a, zr), w);
        wsum = wsum + w.x + w.y + w.z + w.w;
    }
    return sum / max(wsum, 1e-6);
}

// The near map is an atlas two cascades wide: the near cascade in its left half, the close
// one (`camera.light_view_proj_close`) from the top left of its right half, drawn no bigger
// than 2048 texels (`SHADOW_CLOSE_MAX` in lib.rs). `widen`: how much wider than the
// cascade's own the filter is (Enhanced: a veil of high cloud enlarging the sun).
fn shadow_pcf_near(uv: vec2<f32>, z: f32, slope: vec2<f32>, widen: f32) -> f32 {
    let size = shadow_map_size();
    return shadow_tent_near(uv, size, vec2<f32>(0.0), z, slope, widen, SHADOW_BIAS_NEAR / SHADOW_DEPTH_RANGE);
}

fn shadow_pcf_close(uv: vec2<f32>, z: f32, slope: vec2<f32>) -> f32 {
    // (camera.post.w: the close map's size over the near map's)
    let half = shadow_map_size();
    let size = half * select(1.0, camera.post.w, camera.post.w > 0.0);
    return shadow_tent_near(uv, size, vec2<f32>(half, 0.0), z, slope, 1.0, SHADOW_BIAS_CLOSE / SHADOW_DEPTH_RANGE);
}

fn shadow_push_close(n: vec3<f32>, ndl: f32) -> vec3<f32> {
    return n * (0.015 + 0.03 * (1.0 - ndl));
}

// The close cascade's value and weight at a point (weight 0 outside it): it fades into the
// near cascade between 60 and 90 % of its range from the camera.
fn shadow_close(world: vec3<f32>, n: vec3<f32>, ndl: f32, thin: bool) -> vec2<f32> {
    let range = camera.flags.w;
    if (range <= 0.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let radial = distance(world, camera.cam_pos.xyz);
    let w = 1.0 - smoothstep(range * 0.6, range * 0.9, radial);
    if (w <= 0.001) {
        return vec2<f32>(1.0, 0.0);
    }
    let lp = camera.light_view_proj_close * vec4<f32>(world + shadow_push_close(n, ndl), 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
    if (edge >= 0.48 || lp.z < 0.0 || lp.z > 1.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let slope = select(shadow_receiver_slope(camera.light_view_proj_close, n), vec2<f32>(0.0), thin);
    return vec2<f32>(shadow_pcf_close(uv, lp.z, slope), w);
}

// The far map takes the top of its texture (its texels are the texture's first rows); the
// lamps' tiles lie under it (`LampAtlas` in lights.rs).
fn shadow_pcf_far(uv: vec2<f32>, z: f32, slope: vec2<f32>, widen: f32) -> f32 {
    let size = shadow_map_size();
    return shadow_tent_far(uv, size, z, slope, widen, SHADOW_BIAS_FAR / SHADOW_DEPTH_RANGE);
}

// The shadow lookup point: pushed off the surface along its normal by about a texel of the
// cascade (0.14 m near, 0.68 m far), more where the surface turns away from the sun.
fn shadow_push_near(n: vec3<f32>, ndl: f32) -> vec3<f32> {
    return n * (0.06 + 0.12 * (1.0 - ndl));
}

fn shadow_push_far(n: vec3<f32>, ndl: f32) -> vec3<f32> {
    return n * (0.3 + 0.6 * (1.0 - ndl));
}

// `thin`: an alpha-tested card (leaves, fences), whose normal says nothing about its plane
// (OMSI points a tree's normals up): no receiver plane, the lookup moves half a metre
// towards the sun instead, which keeps a crown from speckling itself.
fn sun_shadow(world_in: vec3<f32>, n: vec3<f32>, thin: bool) -> f32 {
    if (camera.shadow.x < 0.5) {
        return 1.0;
    }
    let world = world_in + select(vec3<f32>(0.0), camera.sun_dir.xyz * 0.5, thin);
    let ndl = clamp(dot(n, camera.sun_dir.xyz), 0.0, 1.0);
    let close = shadow_close(world, n, ndl, thin);
    if (close.y >= 0.999) {
        return close.x;
    }
    let near_lp = camera.light_view_proj * vec4<f32>(world + shadow_push_near(n, ndl), 1.0);
    let near_uv = vec2<f32>(near_lp.x * 0.5 + 0.5, 0.5 - near_lp.y * 0.5);
    let inside = max(abs(near_uv.x - 0.5), abs(near_uv.y - 0.5));
    if (inside < 0.48 && near_lp.z >= 0.0 && near_lp.z <= 1.0) {
        let nv = shadow_pcf_near(near_uv, near_lp.z, select(shadow_receiver_slope(camera.light_view_proj, n), vec2<f32>(0.0), thin), 1.0);
        return mix(nv, close.x, close.y);
    }
    let lp = camera.light_view_proj_far * vec4<f32>(world + shadow_push_far(n, ndl), 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || lp.z < 0.0 || lp.z > 1.0) {
        return 1.0;
    }
    let sum = shadow_pcf_far(uv, lp.z, select(shadow_receiver_slope(camera.light_view_proj_far, n), vec2<f32>(0.0), thin), 1.0);
    let edge = clamp((0.5 - max(abs(uv.x - 0.5), abs(uv.y - 0.5))) * 12.0, 0.0, 1.0);
    return mix(1.0, sum, edge);
}

// Sum of the point lights registered in the grid cell of `p`. A light is registered in
// every cell its range touches, so the point's own cell holds every light that reaches it;
// looking into the neighbouring cells as well counted a light up to four times, and how
// often depended on where the cell edges fell - lamp pools brightened and dimmed as the
// camera moved.
// `map_k`: how much of the map's lamps a surface takes (0 on a light-mapped road in the
// classic picture, whose lamps are in its light map); a vehicle's own lights (dir.x 1, see
// lib.rs `gpu_light`) always shine - the headlights lit no road at all in vanilla.
fn point_lights(p: vec3<f32>, n: vec3<f32>, map_k: f32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let cell = camera.light_grid.z;
    let side = u32(camera.light_grid.w);
    if (cell <= 0.0 || side == 0u) {
        return sum;
    }
    let f = (p.xy - camera.light_grid.xy) / cell;
    let x = i32(floor(f.x));
    let y = i32(floor(f.y));
    if (x >= 0 && y >= 0 && x < i32(side) && y < i32(side)) {
        let base = (u32(y) * side + u32(x)) * CELL_CAP;
        for (var j = 0u; j < CELL_CAP; j = j + 1u) {
            let li = grid[base + j];
            if (li == 0xffffffffu) {
                break;
            }
            let l = lights[li];
            let d = l.pos.xyz - p;
            let dist = length(d);
            if (dist >= l.pos.w) {
                continue;
            }
            if (!(l.dir.w < -1.5 && l.dir.z < -0.5) && own_vehicle_blocks(p, l.pos.xyz)) {
                continue;
            }
            if (l.dir.w >= -1.5 && l.extra.z != 0.0) {
                // a Direct3D spot of the classic picture (lights.rs `push_classic_spot`): a
                // vehicle's [spotlight] (its range negative) is the player's alone, lit by
                // `classic_spot` at the vertices; another - a [spotlight_2], a cookie's
                // cone - is lit so here, its ambient light with the diffuse
                if (l.extra.z < 0.0) {
                    continue;
                }
                let s = d3d_spot(p, n, l.pos.xyz, l.dir, l.extra.x, l.extra.z, l.color.rgb * l.color.w);
                sum = sum + s[0] + s[1];
                continue;
            }
            // full intensity within 1/8 of the range, then inverse-square, cut off at the range;
            // the strength was set by eye while a street lamp was still counted about three
            // times (see above), so it is three times what it was, to keep the lamp pools
            let r0 = l.pos.w * 0.125;
            let att = min(1.0, (r0 * r0) / max(dist * dist, 0.01)) * clamp(1.0 - dist / l.pos.w, 0.0, 1.0) * 3.75;
            let ndl = max(dot(n, d / max(dist, 0.01)), 0.15);
            var k = select(map_k, 1.0, l.dir.x > 0.5 && l.dir.w < -1.5);
            if (l.dir.w >= -1.5) {
                // a spot (a vehicle's [spotlight], as Direct3D lights with it): full inside
                // the inner cone, fading to nothing at the outer one; nothing behind it
                let c = dot(-d / max(dist, 0.01), l.dir.xyz);
                k = smoothstep(l.dir.w, max(l.extra.x, l.dir.w + 1e-3), c);
            }
            sum = sum + l.color.rgb * l.color.w * att * ndl * k;
        }
    }
    return sum;
}

// [interiorlight]: the light of a vehicle's saloon lamps on a mesh that names them in its
// [illumination_interior], 1 = a lamp's full light on a seat under it. `code` is the
// instance's lamp code (lib.rs `Instance::interior_lamps`): the first of its lamp slots in
// `lights` times 64 plus how many; below 1 it is a plain brightness (a passenger standing in
// a lit bus). Each lamp is OMSI's Direct3D point light: attenuation
// 1 / (d² / core²) with `core` the lamp's [interiorlight] range, no cut-off short of 100 m,
// times N·L; the sum saturates, as Direct3D's vertex colour does.
fn interior_lamps(p: vec3<f32>, n: vec3<f32>, code: f32) -> vec3<f32> {
    if (code < 1.0) {
        return vec3<f32>(1.0, 0.96, 0.84) * clamp(code, 0.0, 1.0);
    }
    let c = u32(code + 0.5);
    // (LAMP_CODE_STRIDE: 64 - up to 63 lamps a mesh)
    let first = c >> 6u;
    let count = c & 63u;
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let l = lights[first + i];
        let r = l.extra.w;
        let d = l.pos.xyz - p;
        let dist2 = max(dot(d, d), 1e-4);
        if (l.color.w <= 0.0 || dist2 >= r * r) {
            continue;
        }
        let core = max(l.extra.y, 0.01);
        let att = core * core / dist2;
        let ndl = max(dot(n, d * inverseSqrt(dist2)), 0.0);
        sum = sum + l.color.rgb * l.color.w * att * ndl;
    }
    return min(sum, vec3<f32>(1.0));
}


// A Direct3D spot light of the classic picture at `p` (normal `n`), as Omsi.exe sets its
// up for a vehicle's [spotlight] (0x708acc, see lights.rs `push_classic_spot`): `source`
// is the spotlight's own world position; attenuation 1 / (0.2 + d / R + 40 d² / R²)
// (R the declared range), the cone (`dir`: direction and the cosine of half the outer cone;
// `inner` the inner one's) with a falloff of 2, 0.3 x `color` as diffuse light (times N.L)
// and 0.1 x as ambient light, which lights the material's ambient colour whichever way the
// surface faces. [0] the diffuse light, [1] the ambient light.
fn d3d_spot(p: vec3<f32>, n: vec3<f32>, source: vec3<f32>, dir: vec4<f32>, inner: f32, range: f32, color: vec3<f32>) -> array<vec3<f32>, 2> {
    let to_light = source - p;
    let dl = max(length(to_light), 0.01);
    let att = 1.0 / (0.2 + dl / range + 40.0 * dl * dl / (range * range));
    let rho = dot(-to_light / dl, dir.xyz);
    let cone = pow(clamp((rho - dir.w) / (max(inner, dir.w + 1e-4) - dir.w), 0.0, 1.0), 2.0);
    let lit = color * att * cone;
    return array<vec3<f32>, 2>(lit * 0.3 * max(dot(n, to_light / dl), 0.0), lit * 0.1);
}

// The classic picture's headlight at a vertex: Omsi.exe's one Direct3D spot light, the
// player's [spotlight] (`camera.spot`, see setup.rs `classic_spot`), lit at the vertices as
// its fixed function lights them. Across a road, whose vertices lie at its edges, the pool
// is spread as OMSI 2 spreads it: lit per pixel it was brightest on the crown of the road
// in front of the bus and gone far sooner. `vs_main` excludes the player's own vehicle.
// [0] is the vehicle-face origin for occlusion; [3] is the declared source for lighting.
// Nothing beyond 4 x its range, as Direct3D cuts it off.
fn classic_spot(p: vec3<f32>, n: vec3<f32>) -> array<vec3<f32>, 2> {
    let s = camera.spot;
    let none = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    if (s[0].w <= 0.0 || distance(s[3].xyz, p) >= 4.0 * s[0].w || own_vehicle_blocks(p, s[0].xyz)) {
        return none;
    }
    return d3d_spot(p, n, s[3].xyz, s[1], s[2].w, s[0].w, s[2].rgb);
}
// Value noise in three octaves: the "fractal" detail (the `detail_textures` setting) laid
// over the ground and the road surfaces close by, so that a blurred texture keeps some grain.
//
// The lattice cell is hashed as an integer. The old `fract(sin(dot(q, …)) * 43758.5)`
// hash took the cell's map coordinate - two and a half million at Spandau's 892 km - into
// a sine: in 32-bit floats its argument is a multiple of 64 there, and what came out was
// not noise but a ramp that repeated along straight lines, so every road wore thin
// diagonal streaks at an exact spacing. Cells are wrapped at `PATTERN_PERIOD` (the pattern
// coordinate is taken modulo that on the CPU, see `world_pattern_xy`), which keeps the
// hash's input small and the pattern seamless across the wrap.
const PATTERN_PERIOD: f32 = 1000.0;

fn hash_cell(c: vec2<f32>, cells: f32) -> f32 {
    // the cell index modulo the lattice's cells per period, as an exact integer
    let w = c - cells * floor(c / cells);
    var v = vec2<u32>(w) * 1664525u + vec2<u32>(1013904223u);
    // (a PCG-style mix: two rounds of multiply, cross-add and xor-shift)
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return f32(v.x >> 8u) / 16777215.0;
}

// `freq`: lattice cells per metre of `p` (a whole number of cells per `PATTERN_PERIOD`).
fn vnoise_f(p: vec2<f32>, freq: f32, offset: vec2<f32>) -> f32 {
    let q = p * freq + offset;
    let cells = round(PATTERN_PERIOD * freq);
    let i = floor(q);
    let f = q - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash_cell(i, cells);
    let b = hash_cell(i + vec2<f32>(1.0, 0.0), cells);
    let c = hash_cell(i + vec2<f32>(0.0, 1.0), cells);
    let d = hash_cell(i + vec2<f32>(1.0, 1.0), cells);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}


fn detail_noise(p: vec2<f32>, near: f32) -> f32 {
    // Anti-alias: point-sampling a lattice this fine aliases wherever one pixel's
    // footprint spans more than about one cell of it - looking down a street at a
    // grazing angle, or just driving away, constantly changes that footprint, and the
    // alias pattern with it, which reads as the "fractal" texture re-rendering itself as
    // the view angle changes. `fwidth` gives the footprint in the same units as `p`; an
    // octave fades to its own mean (0.5, the hash is uniform 0..1) once a pixel can no
    // longer resolve it, which is what an infinitely dense filter of it would give anyway.
    let w = max(fwidth(p).x, fwidth(p).y);
    let seen = select(0.0, 1.0, near > 0.0);
    let fade0 = (1.0 - smoothstep(0.3, 1.2, w * 2.9)) * seen;
    let fade1 = (1.0 - smoothstep(0.3, 1.2, w * 0.73)) * seen;
    let fade2 = (1.0 - smoothstep(0.3, 1.2, w * 0.19)) * seen;
    var n0 = 0.5;
    var n1 = 0.5;
    var n2 = 0.5;
    if (fade0 > 0.0) {
        n0 = mix(0.5, vnoise_f(p, 2.9, vec2<f32>(0.0)), fade0);
    }
    if (fade1 > 0.0) {
        n1 = mix(0.5, vnoise_f(p, 0.73, vec2<f32>(0.0)), fade1);
    }
    if (fade2 > 0.0) {
        n2 = mix(0.5, vnoise_f(p, 0.19, vec2<f32>(0.0)), fade2);
    }
    return n0 * 0.5 + n1 * 0.3 + n2 * 0.2;
}

fn world_pattern_xy(p: vec3<f32>) -> vec2<f32> {
    // Geometry is relative to the floating render origin for f32 precision. Procedural
    // patterns are not: reconstruct their stable map-space coordinate so crossing the
    // renderer's 100 m origin cell cannot re-seed the asphalt/ground pattern. The origin
    // arrives modulo PATTERN_PERIOD (lib.rs), so this stays a small, exact number: the map
    // coordinate itself (892 km at Spandau) has a 6 cm step in 32 bits, a fifth of the
    // finest noise cell.
    return p.xy + camera.world_origin.xy;
}

// Whether every component is a finite number, tested on the bits: Metal's fast maths takes
// `x != x` and comparisons with a NaN as it likes (a NaN colour happened to come out as
// something on the Mac), Direct3D 12 keeps the NaN and draws it black.
fn all_finite(v: vec3<f32>) -> bool {
    let e = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
    return all(e != vec3<u32>(0x7f800000u));
}

fn finite_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    return select(fallback, v, all_finite(v));
}

// The side of a tracked pane's wetness mask (`window_wipers::SIZE`). A constant, not the
// texture's own size: with `textureDimensions(mx_tex2d[mx_mat.trans])` in these functions Apple's Metal
// compiler crashed on the scene shader (the pipelines failed and the renderer fell back to
// 1x MSAA without snowfall and lamp shadows).
const WIPE_MASK_SIZE: i32 = 128;

fn window_wetness(in: FsIn) -> f32 {
    if (material.wipe_bounds.z != 0.0) {
        // The mask packs the pane depth into R/G (high and low byte) and the wetness into
        // B (the pane's unwiped wetness) and A (the local wetness). A filtered sample mixes
        // the depth bytes of neighbouring texels into a depth that belongs to neither, so
        // the pane test flickered between A and B from texel to texel: a staircase along
        // every texel row where the raked glass changes depth. Read the four texels
        // unfiltered, test each by its own depth, and blend the results bilinearly.
        let dims = vec2<i32>(WIPE_MASK_SIZE);
        let inside = all(in.wipe_uv.xy >= vec2<f32>(0.0)) && all(in.wipe_uv.xy <= vec2<f32>(1.0));
        let p = in.wipe_uv.xy * vec2<f32>(dims) - 0.5;
        let base = vec2<i32>(floor(p));
        let f = p - floor(p);
        var wet = 0.0;
        for (var k = 0; k < 4; k++) {
            let o = vec2<i32>(k & 1, k >> 1);
            let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
            let texel = textureLoad(mx_tex2d[mx_mat.trans], clamp(base + o, vec2<i32>(0), dims - 1), 0);
            let depth = dot(texel.rg, vec2<f32>(256.0, 1.0)) * (64.0 / 257.0) - 32.0;
            let local = (texel.a * 255.0 - 8.0) * (2.0 / 247.0);
            wet += w * select(texel.b, local, inside && abs(depth - in.wipe_uv.z) <= 0.1);
        }
        return wet;
    }
    return in.params.x;
}

// The snow variant of a tracked pane (`wipe_bounds.w` < 0): the snow the mask holds while it
// snows, drawn instead of water drops (scene/08_rain_glass.wgsl `glass_snow_with`). The mask
// value is the cover: 0 clear glass, up to 1 a pane closed by snow, above 1 the bank a blade
// pushed to the end of its sweep, below 0 glass a blade has just cleared.
fn is_snow_pane() -> bool {
    return material.wipe_bounds.z != 0.0 && material.wipe_bounds.w < 0.0;
}

// Smooth value noise (0..1) from the cell hash: the corners' values blended with a
// smoothstep, so it has no seams at the cell edges.
fn snow_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = rain_hash(i).x;
    let b = rain_hash(i + vec2<f32>(1.0, 0.0)).x;
    let c = rain_hash(i + vec2<f32>(0.0, 1.0)).x;
    let d = rain_hash(i + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// By day in a downpour some of the houses' windows are lit, as people switch the light on when
// it gets that dark (`camera.wind.w` - 1: the share, `Lighting::rain_lights`, 0 with the
// setting off): the night maps of a few rooms here and there, chosen by a cell of the
// object's own frame some four metres wide and three high (`local`: the same rooms of a
// house stay lit, and a bus driving by does not flicker as it crosses cells of the world),
// each fading in at its own point of the rain. What a night map shows: the night's own
// strength or, where such a room is lit, the full light.
fn rain_lit_windows(local: vec3<f32>) -> f32 {
    let night = clamp(camera.sun_color.w, 0.0, 1.0);
    let share = clamp(camera.wind.w - 1.0, 0.0, 1.0);
    if (share <= 0.0) {
        return night;
    }
    let cell = floor(vec3<f32>(local.xy / 4.0, local.z / 3.0));
    let h = rain_hash(cell.xy + vec2<f32>(cell.z * 17.3, cell.z * 5.1));
    // (some a third of the rooms at the heaviest rain)
    let lit = smoothstep(h.x, h.x + 0.08, share * 0.38);
    return max(night, lit * (0.75 + 0.25 * h.y));
}
fn rain_hash(p: vec2<f32>) -> vec2<f32> {
    // Integer mixing keeps neighbouring cells independent without the precision loss and
    // repeated sine evaluations of a floating-point hash.
    let cell = bitcast<vec2<u32>>(vec2<i32>(floor(p)));
    var a = cell.x * 0x9e3779b9u ^ cell.y * 0x85ebca6bu;
    a = (a ^ (a >> 16u)) * 0x7feb352du;
    a = (a ^ (a >> 15u)) * 0x846ca68bu;
    a = a ^ (a >> 16u);
    var b = a ^ 0x68bc21ebu;
    b = (b ^ (b >> 16u)) * 0x7feb352du;
    b = (b ^ (b >> 15u)) * 0x846ca68bu;
    b = b ^ (b >> 16u);
    return vec2<f32>(f32(a), f32(b)) * (1.0 / 4294967296.0);
}

// Smooth noise for water-film refraction and residual wiper streaks (0..1).
fn rain_patches(q: vec2<f32>) -> vec2<f32> {
    let c = floor(q);
    let f = q - c;
    let w = f * f * (3.0 - 2.0 * f);
    let a = rain_hash(c);
    let b = rain_hash(c + vec2<f32>(1.0, 0.0));
    let d = rain_hash(c + vec2<f32>(0.0, 1.0));
    let e = rain_hash(c + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, w.x), mix(d, e, w.x), w.y);
}

// A grid of its own for each layer of drops: turned by its own angle and shifted, so that
// no two layers line up and none runs along the glass's edges.
fn rain_turn(q: vec2<f32>, a: f32) -> vec2<f32> {
    let cs = vec2<f32>(cos(a), sin(a));
    return vec2<f32>(q.x * cs.x - q.y * cs.y, q.x * cs.y + q.y * cs.x);
}

// What a drop shows in the classic picture: the sky from the weather's colours, the
// horizon in the fog's, the ground darker.
fn rain_env_vanilla(d: vec3<f32>) -> vec3<f32> {
    let zenith = camera.sky_color.rgb + camera.ambient.rgb * 0.5;
    let horizon = max(camera.fog.rgb, camera.ambient.rgb);
    let ground = camera.ambient.rgb * 0.4 + camera.sun_color.rgb * camera.sun_dir.w * 0.06;
    let sky = mix(horizon, zenith, smoothstep(0.0, 0.5, d.z));
    let c = mix(ground, sky, smoothstep(-0.06, 0.04, d.z));
    return select(c, srgb_decode(c), camera.sky_color.w > 0.5);
}

// --- The rain on a pane as all the pictures draw it (the enhanced film, enhanced/07_glass.wgsl
// `glass_film`, and the classic one, scene/09_vanilla.wgsl `glass_film_classic`): the drops laid
// out in the vehicle's own frame (`FsIn.local`), so they stay on the glass however the body
// rolls and pitches and on a door that swings open. The modellers' texture coordinates of a
// film are no measure of the glass: on the windscreen of the NL202 they run half as fast
// again one way as the other and at sixty degrees to each other, from triangle to triangle
// anew, and every drop drawn in them came out a slanting streak (the classic pictures' old
// `rain_glass` did, and was replaced by this). So a pane is laid flat by the axes of the
// vehicle (`pane_proj`).

// The glass at a pixel: the position and the flat normal in the vehicle's frame, the
// normal on the rain's side in the world, the pixel's footprint on the glass (m, its longer
// side and the side of a square of its area), and the screen derivatives of both positions.
struct PaneBase {
    l: vec3<f32>,
    nl: vec3<f32>,
    out: vec3<f32>,
    px: f32,
    px_area: f32,
    dlx: vec3<f32>,
    dly: vec3<f32>,
    dwx: vec3<f32>,
    dwy: vec3<f32>,
    ok: bool,
};

fn pane_base(world: vec3<f32>, local: vec3<f32>, n: vec3<f32>, outside_in: bool) -> PaneBase {
    var b: PaneBase;
    b.l = local;
    b.dlx = dpdx(local);
    b.dly = dpdy(local);
    b.dwx = dpdx(world);
    b.dwy = dpdy(world);
    // the film faces the eye unless the eye is in the bus behind it (the triangle's own
    // normal: a film mesh's vertex normals are not to be relied on, many have none, and
    // the drops' size followed their noise)
    let eye = normalize(camera.cam_pos.xyz - world);
    let gn = cross(b.dwx, b.dwy);
    var out = select(safe_normal(n), gn / max(length(gn), 1e-20), dot(gn, gn) > 1e-24);
    out = select(-out, out, dot(out, eye) >= 0.0);
    out = select(out, -out, outside_in);
    b.out = out;
    b.px = max(max(length(b.dwx), length(b.dwy)), 1e-5);
    b.px_area = max(sqrt(length(cross(b.dwx, b.dwy))), 1e-5);
    let c = cross(b.dlx, b.dly);
    let cl = length(c);
    b.ok = cl > 1e-12;
    b.nl = select(vec3<f32>(0.0, 1.0, 0.0), c / max(cl, 1e-20), b.ok);
    return b;
}

// A direction on the glass in the vehicle's frame, in the world (through the two positions'
// screen derivatives).
fn pane_to_world(b: PaneBase, t: vec3<f32>) -> vec3<f32> {
    let g = vec3<f32>(dot(b.dlx, b.dlx), dot(b.dlx, b.dly), dot(b.dly, b.dly));
    let det = g.x * g.z - g.y * g.y;
    if (abs(det) < 1e-24) {
        return vec3<f32>(0.0);
    }
    let r = vec2<f32>(dot(b.dlx, t), dot(b.dly, t));
    let a = (g.z * r.x - g.y * r.y) / det;
    let c = (g.x * r.y - g.y * r.x) / det;
    return a * b.dwx + c * b.dwy;
}

// The pane laid flat along two of the vehicle's axes (`axis`: 0 across and up, the
// windscreen and the rear window; 1 along and up, the side windows; 2 the plan, a roof
// light): where the pixel is (m: x across, y down), how an offset there is taken back onto
// the glass (`m`: metres along `side_w` and `down_w`, the glass's own across and down in
// the world), and the projection's screen derivatives (for the wiper's ridge).
struct PaneProj {
    q: vec2<f32>,
    m: mat2x2<f32>,
    side_w: vec3<f32>,
    down_w: vec3<f32>,
    dqx: vec2<f32>,
    dqy: vec2<f32>,
    steep: bool,
};

fn pane_proj(b: PaneBase, axis: i32) -> PaneProj {
    var p: PaneProj;
    var ea = vec3<f32>(1.0, 0.0, 0.0);
    var eb = vec3<f32>(0.0, 0.0, -1.0);
    var ec = vec3<f32>(0.0, 1.0, 0.0);
    if (axis == 1) {
        ea = vec3<f32>(0.0, 1.0, 0.0);
        ec = vec3<f32>(1.0, 0.0, 0.0);
    } else if (axis == 2) {
        eb = vec3<f32>(0.0, 1.0, 0.0);
        ec = vec3<f32>(0.0, 0.0, 1.0);
    }
    let nl = b.nl;
    let nc = dot(nl, ec);
    let ncs = select(nc, select(-1e-3, 1e-3, nc >= 0.0), abs(nc) < 1e-3);
    p.q = vec2<f32>(dot(b.l, ea), dot(b.l, eb));
    p.dqx = vec2<f32>(dot(b.dlx, ea), dot(b.dlx, eb));
    p.dqy = vec2<f32>(dot(b.dly, ea), dot(b.dly, eb));
    // the glass under a step of the projection's two axes
    let t1 = ea - ec * (dot(nl, ea) / ncs);
    let t2 = eb - ec * (dot(nl, eb) / ncs);
    // the glass's own down (the steepest way on it) and across, in the vehicle's frame
    p.steep = axis != 2;
    var dl = vec3<f32>(0.0, 0.0, -1.0) + nl * nl.z;
    dl = select(normalize(t2), dl / max(length(dl), 1e-6), dot(dl, dl) > 0.02 && p.steep);
    var sl = t1 - dl * dot(t1, dl);
    sl = sl / max(length(sl), 1e-6);
    p.m = mat2x2<f32>(vec2<f32>(dot(t1, sl), dot(t1, dl)), vec2<f32>(dot(t2, sl), dot(t2, dl)));
    // the same two ways in the world, square to the normal on the rain's side
    var sw = pane_to_world(b, sl);
    sw = sw - b.out * dot(sw, b.out);
    var dw = pane_to_world(b, dl);
    dw = dw - b.out * dot(dw, b.out);
    p.side_w = sw / max(length(sw), 1e-6);
    dw = dw - p.side_w * dot(dw, p.side_w);
    p.down_w = dw / max(length(dw), 1e-6);
    return p;
}

// The water on a pane at a pixel: the surface's normal (world, out of the water), how much
// of the pixel a drop or a bead covers and how far out on it the pixel lies (0 its middle,
// 1 its rim), the drop's radius in pixels, the pane's normal on the rain's side, the fine
// mist of droplets under a millimetre, and how much of the view through the pane the
// connected water scatters into a soft blur (`veil`).
struct GlassWater {
    n: vec3<f32>,
    cover: f32,
    out: vec3<f32>,
    mist: f32,
    veil: f32,
    rim: f32,
    size_px: f32,
};

// One drop of radius `r` (m) at `d` (m on the glass, across / down): the slope of its dome
// (the normal's part along the glass), its cover of the pixel (`px`: the pixel's size on
// the glass) and how far out on it the pixel lies. A shallow cap, as water spreading on
// glass makes; a drop smaller than the pixel keeps its share of it, its slopes flattened
// to their average.
fn glass_dome(d: vec2<f32>, r: f32, px: f32) -> vec4<f32> {
    let rr = length(d) / max(r, 1e-6);
    let aa = clamp(px / max(r, 1e-6), 0.02, 1.0);
    let cover = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, rr);
    let seen = smoothstep(0.0, 0.8, r / px);
    let cap = mix(0.15, 0.55, smoothstep(0.7, 2.5, r / px));
    let s = d / max(r, 1e-6) * cap * (0.6 + 0.4 * rr * rr);
    return vec4<f32>(s, cover * seen, min(rr, 1.0));
}

// What covers the pixel, gathered over the layers: the cover, the slope, the rim and the
// drop's size in pixels of the drop on top.
struct DropAcc {
    cover: f32,
    slope: vec2<f32>,
    rim: f32,
    size_px: f32,
    track: f32,
    // the cover of the drops too small for the pixel to see one by one, as their average
    // (taken into the mist, so that they read as an even haze, not as random specks)
    avg: f32,
};

fn drop_take(a: ptr<function, DropAcc>, d: vec4<f32>, r: f32, px: f32) {
    if (d.z > (*a).cover) {
        (*a).cover = d.z;
        (*a).slope = d.xy;
        (*a).rim = d.w;
        (*a).size_px = r / px;
    }
}

// One layer of the drops that sit on the glass: a jittered grid of cells `cell` metres
// wide (of the projection), each holding a drop now and then (`occ` the share of cells
// that hold one). A drop lands, spreads to its size within a moment and sits; after its
// own while it is gone again, taken by a runner or merged into a neighbour, and the next
// lands somewhere else in the cell - all fading, nothing popping.
fn glass_bead(a: ptr<function, DropAcc>, p: PaneProj, layer: f32, cell: f32, occ: f32, r0: f32, size: f32, t: f32, px: f32) {
    // (a layer whose drops are smaller than the pixel, or cells no wider than it - a mirror's
    // small picture, the glass seen from afar - is taken at its average cover: sampled one
    // random cell a pixel, it drew white noise)
    let fine = max(1.0 - smoothstep(0.7, 1.5, r0 * size / px), 1.0 - smoothstep(1.0, 2.5, cell / px));
    if (fine > 0.0) {
        let rr = 1.6 * r0 * size / cell;
        (*a).avg = (*a).avg + fine * occ * min(3.14159 * rr * rr, 1.0);
        if (fine > 0.99) {
            return;
        }
    }
    let turn = 0.61 + layer * 1.37;
    let g2 = rain_turn(p.q, turn) / cell + vec2<f32>(layer * 17.3, layer * 5.1);
    let c = floor(g2);
    let timing = rain_hash(c + vec2<f32>(3.7, 9.2 + layer));
    let life = (9.0 + layer * 4.0) * (1.0 + timing.y);
    let cycle = t / life + timing.x;
    let ph = fract(cycle);
    let arrival = c + floor(cycle) * vec2<f32>(17.0, 43.0);
    let h = rain_hash(arrival);
    // (a share of cells rising with the wetness: a drop fades in over a small rise of it)
    let shown = clamp((occ - h.x) * 20.0, 0.0, 1.0) * smoothstep(0.0, 0.03, ph) * (1.0 - smoothstep(0.9, 1.0, ph)) * (1.0 - fine);
    if (shown <= 0.0) {
        return;
    }
    // (`r0` the layer's smallest drop, the largest twice that: rain on glass sits as drops of
    // half a millimetre to four across - drawn up to a share of the cell they made drops of a
    // centimetre and more, few and far between)
    let radius = r0 * (1.0 + h.y * h.y) * mix(0.6, 1.0, smoothstep(0.0, 0.2, ph)) * size;
    // (anywhere in the cell the largest still fits in - on the glass it is never smaller
    // than in the projection - and the same place whatever its size: the size follows the
    // wetness, which changes across the glass)
    let room = min(r0 * 3.0 / cell, 0.45);
    let centre = c + vec2<f32>(room) + rain_hash(arrival + 9.1) * (1.0 - 2.0 * room);
    let dq = rain_turn((g2 - centre) * cell, -turn);
    if (length(dq) > radius * 1.3) {
        return;
    }
    var d = p.m * dq;
    // No drop is a circle: its rim wanders where the glass held it as it spread, and its
    // weight draws it a little fuller below than above.
    let h3 = rain_hash(arrival + 23.9);
    let dl = max(length(d), 1e-7);
    let u = d / dl;
    let c2 = u.x * u.x - u.y * u.y;
    let s2 = 2.0 * u.x * u.y;
    let c3 = u.x * (4.0 * u.x * u.x - 3.0);
    let s3 = u.y * (3.0 - 4.0 * u.y * u.y);
    let p2 = h3 * 2.0 - 1.0;
    let wave = (s2 * p2.x + c2 * p2.y) * 0.7 + 0.45 * (s3 * p2.y - c3 * p2.x);
    d = d / (1.0 + (0.03 + 0.05 * h.y) * wave);
    d.y = d.y * select(1.06, 0.94, d.y > 0.0);
    let dome = glass_dome(d, radius, px);
    drop_take(a, vec4<f32>(dome.xy, dome.z * shown, dome.w), radius, px);
}

// A runner's sideways wander as it goes down (m): a smooth random path of its own.
fn glass_meander(s: f32, seed: vec2<f32>, amp: f32) -> f32 {
    let w = rain_patches(vec2<f32>(s * 9.0 + seed.x, seed.y));
    return amp * ((w.x - 0.5) * 1.5 + (w.y - 0.5) * 0.5);
}

// The runners in a lattice of lanes (`qr`: the projection turned so they run down +y, `mr`
// taking its offsets onto the glass): now and then a drop grown heavy breaks loose and
// slides a hand's breadth or two down, in fits and starts, wandering a little from side to
// side, and stops again. It sweeps a narrow clear track through the small drops, which
// fill in again as the rain falls on it; behind it lies a short wet trail that drains
// within a second, and a few beads it left. One chance a cell of 4.5 cm x 1.2 m; a pixel
// looks at its own cell and the one above, whose drop may have slid into it.
fn glass_runners(a: ptr<function, DropAcc>, qr: vec2<f32>, mr: mat2x2<f32>, seed: f32, wet: f32, heavy: f32, blow: f32, size: f32, t: f32, px: f32) {
    // (lanes of 4.5 cm and cells 1.2 m tall, few of them running at once: with lanes of 2 cm
    // and cells of 0.8 m nearly every lane ran in a downpour, and their tracks cut the glass
    // into wide clear stripes, #4153)
    let lane_w = 0.045;
    let seg_h = 1.2;
    let lane = floor(qr.x / lane_w);
    let seg0 = floor(qr.y / seg_h);
    for (var k = 0; k < 2; k = k + 1) {
        let seg = seg0 - f32(k);
        let key = vec2<f32>(lane * 1.7 + seed * 101.0, seg * 2.3 + seed * 37.0);
        let h1 = rain_hash(key + vec2<f32>(3.0, 11.0));
        if (h1.x >= wet * (0.15 + 0.15 * heavy + 0.1 * blow)) {
            continue;
        }
        let h2 = rain_hash(key + vec2<f32>(8.1, 5.1));
        // (it gathers a while, runs, and its trail fills in with new rain)
        // (a drop grown heavy on upright glass runs off at some ten to twenty centimetres a
        // second, most of the way down the pane)
        let sit = 0.6 + 2.4 * h2.y;
        let v = (0.08 + 0.16 * h1.y * h1.y) * (1.0 + 0.8 * blow) * (1.0 + 0.4 * heavy);
        let run_len = 0.12 + 0.4 * h2.x;
        let run_t = run_len / v;
        let refill = mix(3.0, 1.5, wet);
        let period = sit + run_t + refill;
        let cycle = t / period + h2.x;
        let tau = fract(cycle) * period;
        let h3 = rain_hash(key + vec2<f32>(1.3, 29.7) + floor(cycle) * vec2<f32>(13.0, 41.0));
        // stick and slip: v s - v P / 2 pi sin(2 pi s / P) halts once a pulse P
        let pulse = 0.3 + 0.4 * h3.x;
        let st = clamp(tau - sit, 0.0, run_t * 1.05);
        let travel = min(v * st - v * pulse * 0.1591549 * sin(6.283185 * st / pulse), run_len);
        let vis = smoothstep(0.0, 0.6, tau) * (1.0 - smoothstep(period - 1.0, period, tau));
        let y0 = (seg + 0.02 + 0.32 * h3.x) * seg_h;
        let x0 = (lane + 0.3 + 0.4 * h3.y) * lane_w;
        let wseed = h3 * 57.0;
        let amp = 0.002 + 0.002 * h2.y;
        if (abs(qr.x - x0) > lane_w * 0.5 + amp * 1.5 || qr.y < y0 - 0.01 || qr.y > y0 + run_len + 0.01) {
            continue;
        }
        // (a downpour's runners are heavier; a drop gathers before it goes)
        let rh = size * (0.0012 + 0.0008 * h1.y) * (1.0 + 0.3 * heavy) * mix(0.6, 1.0, smoothstep(0.0, sit, tau));
        var head_d = mr * vec2<f32>(qr.x - (x0 + glass_meander(travel, wseed, amp)), qr.y - (y0 + travel));
        // (drawn out a little behind, full in front)
        head_d.y = head_d.y * select(0.8, 1.0, head_d.y > 0.0);
        let head = glass_dome(head_d, rh, px);
        // (once stopped it sits at the end of its run and is taken into the new rain)
        let head_vis = vis * (1.0 - smoothstep(sit + run_t, sit + run_t + refill * 0.8, tau));
        drop_take(a, vec4<f32>(head.xy, head.z * head_vis, head.w), rh, px);
        let along = qr.y - y0;
        if (along > 0.0 && along < travel - rh * 0.5) {
            let dxq = qr.x - (x0 + glass_meander(along, wseed, amp));
            let dx = (mr * vec2<f32>(dxq, 0.0)).x;
            // how long ago the head passed here
            let age = max(tau - sit - along / v, 0.0);
            let hw = rh * 0.9;
            let cleared = (1.0 - smoothstep(hw * 0.6, hw * 1.1, abs(dx))) * (1.0 - smoothstep(refill * 0.4, refill, age)) * vis;
            (*a).track = max((*a).track, cleared);
            // the thin wet trail it draws, draining within a second: a faint lens, no strip
            let tw = rh * 0.45 * exp(-age * 2.0);
            let trail = glass_dome(vec2<f32>(dx, 0.0), max(tw, 1e-5), px);
            let trail_cover = trail.z * 0.35 * (1.0 - smoothstep(0.2, 1.0, age)) * vis;
            drop_take(a, vec4<f32>(trail.xy * 0.4, trail_cover, trail.w), tw, px);
            // ... and a few beads, left behind where the trail broke
            let bc = floor(along / 0.009);
            let bh = rain_hash(vec2<f32>(lane * 3.1 + bc, seg * 1.9 + 41.0 + seed * 7.0));
            if (bh.x < 0.3) {
                let bcen = vec2<f32>((bh.y - 0.5) * hw * 0.8, ((bc + 0.5) * 0.009 - along) * (mr * vec2<f32>(0.0, 1.0)).y);
                let br = 0.0004 + 0.0005 * bh.y;
                let bead = glass_dome(vec2<f32>(dx, 0.0) - bcen, br, px);
                let bead_cover = bead.z * smoothstep(0.4, 1.0, age) * (1.0 - smoothstep(refill * 0.5, refill, age)) * vis;
                drop_take(a, vec4<f32>(bead.xy, bead_cover, bead.w), br, px);
            }
        }
    }
}

// The drops on one projection of the pane (`w` its share, see `glass_water`).
fn glass_drops(a: ptr<function, DropAcc>, b: PaneBase, p: PaneProj, w: f32, wet: f32, heavy: f32, bank: f32, size: f32, world: vec3<f32>, t: f32) {
    // --- the airstream: a drop's drag grows with the speed squared, and at some 40 km/h it
    // matches its weight. Along the glass the air drives the water back; where it meets the
    // glass head on (the windscreen) it parts, up and out. The runners go where weight and
    // air take them together; the drops that sit hold on.
    var flow = vec2<f32>(0.0, 1.0);
    var blow = 0.0;
    if (camera.wind.w > 0.5 && near_player_vehicle(world) > 0.5) {
        let air = -camera.wind.xyz;
        var aw = vec2<f32>(dot(air, p.side_w), dot(air, p.down_w));
        aw.y = aw.y - abs(dot(air, b.out)) * 0.6;
        let v2 = dot(aw, aw);
        blow = clamp(v2 / 130.0, 0.0, 3.0);
        flow = flow + aw / max(sqrt(v2), 1e-4) * blow;
    }
    if (p.steep) {
        // The runners' lanes are laid along the way the water runs, in sixteenths of a turn
        // (turned freely, a degree moved every lane a metre off by its width). Between two
        // such ways the two lattices cross-fade, so a change of speed fades the runners
        // over instead of jumping them. (`flow` is on the glass; the lanes in the
        // projection, whose down is the glass's down.)
        let fstep = 0.3926991;
        let fa = atan2(flow.x, flow.y) / fstep;
        let f0 = floor(fa);
        let fw = smoothstep(0.35, 0.65, fa - f0);
        for (var k = 0; k < 2; k = k + 1) {
            let wk = select(1.0 - fw, fw, k == 1) * w;
            if (wk < 0.01) {
                continue;
            }
            let ang = (f0 + f32(k)) * fstep;
            let cs = vec2<f32>(cos(ang), sin(ang));
            let mr = p.m * mat2x2<f32>(vec2<f32>(cs.x, -cs.y), vec2<f32>(cs.y, cs.x));
            glass_runners(a, rain_turn(p.q, ang), mr, f0 + f32(k), wet * wk, heavy, blow, size, t, b.px_area);
        }
    }
    // --- the drops that sit: four sizes, a great many small ones and fewer large ones (radius
    // 1 to 2, 1.6 to 3.2, 2.4 to 4.8 and 3.4 to 6.8 mm: smaller ones, true to a drizzle's,
    // were under a pixel from the driver's seat and drew the windscreen as an even grain),
    // all fewer in the runners' fresh
    // tracks and under the bank
    let room = (1.0 - (*a).track) * (1.0 - bank) * w;
    for (var layer = 0; layer < 4; layer = layer + 1) {
        let fl = f32(layer);
        var cell = 0.0068;
        var r0 = 0.001;
        var occ = 0.85 * smoothstep(0.02, 0.25, wet);
        if (layer == 1) {
            cell = 0.0108;
            r0 = 0.0016;
            occ = 0.75 * smoothstep(0.05, 0.4, wet);
        } else if (layer == 2) {
            cell = 0.016;
            r0 = 0.0024;
            occ = 0.6 * smoothstep(0.15, 0.6, wet);
        } else if (layer == 3) {
            cell = 0.023;
            r0 = 0.0034;
            occ = 0.4 * smoothstep(0.35, 0.9, wet);
        }
        glass_bead(a, p, fl, cell, occ * room, r0, size, t, b.px_area);
    }
}

// The rain on a pane at a pixel (see the head of this file). `water` is the pane's
// wetness (0..1; below 0 a blade has just wiped it, above 1 the bank a blade pushed),
// `t` the animation clock, `outside_in` true for the eye in the bus.
fn glass_water(world: vec3<f32>, local: vec3<f32>, n: vec3<f32>, water: f32, t: f32, outside_in: bool) -> GlassWater {
    let b = pane_base(world, local, n, outside_in);
    let wdx = dpdx(water);
    let wdy = dpdy(water);
    var g: GlassWater;
    g.n = b.out;
    g.out = b.out;
    g.cover = 0.0;
    g.mist = 0.0;
    g.veil = 0.0;
    g.rim = 0.0;
    g.size_px = 0.0;
    let wet = clamp(water, 0.0, 1.0);
    let wiped = clamp(-water * 25.0, 0.0, 1.0);
    let bank = clamp((water - 1.0) * 0.18, 0.0, 1.0);
    if (!b.ok || (wet <= 0.01 && wiped <= 0.01 && bank <= 0.0)) {
        return g;
    }
    // Which way the pane is laid flat: by the larger of its normal's parts across and along
    // the vehicle, the two cross-faded over a band round 45 degrees (a windscreen's curved
    // corner), each drop of either set coming or going whole; a pane facing up is a roof's.
    let an = abs(b.nl);
    let roof = an.z > max(an.x, an.y) * 2.0;
    let w_front = select(smoothstep(-0.12, 0.12, an.y - an.x), 1.0, roof);
    let main_axis = select(select(1, 0, w_front >= 0.5), 2, roof);
    let p = pane_proj(b, main_axis);
    // the wetness's gradient on the glass (from its screen derivatives)
    let jd = p.dqx.x * p.dqy.y - p.dqx.y * p.dqy.x;
    var grad = vec2<f32>(0.0);
    if (abs(jd) > 1e-14) {
        grad = vec2<f32>(wdx * p.dqy.y - wdy * p.dqx.y, wdy * p.dqx.x - wdx * p.dqy.x) / jd;
    }
    if (wiped > 0.01) {
        // Just wiped: fine streaks along the blade's sweep (the age gradient follows it)
        // bend the view for a moment, without bringing the drops back.
        var along = vec2<f32>(0.0, 1.0);
        if (length(grad) > 0.01) {
            along = normalize(grad);
        }
        let across = vec2<f32>(-along.y, along.x);
        let streak = rain_patches(vec2<f32>(dot(p.q, across) * 600.0, dot(p.q, along) * 4.0));
        g.cover = wiped * (0.012 + 0.018 * streak.x);
        g.veil = wiped * 0.06;
        g.size_px = 4.0;
        g.n = normalize(b.out + (p.side_w * across.x + p.down_w * across.y) * (streak.y - 0.5) * wiped * 0.003);
        return g;
    }
    let heavy = smoothstep(0.55, 1.0, wet);
    // (a drizzle leaves beads, a downpour full drops; on an upright side pane, which the rain
    // meets edge on and the airstream sweeps, they stay smaller than on a raked windscreen)
    let size = mix(0.85, 1.1, smoothstep(0.1, 0.8, wet)) * mix(0.8, 1.0, smoothstep(0.0, 0.4, abs(b.out.z)));
    var acc: DropAcc;
    acc.cover = 0.0;
    acc.slope = vec2<f32>(0.0);
    acc.rim = 0.0;
    acc.size_px = 0.0;
    acc.track = 0.0;
    acc.avg = 0.0;
    // the bank a blade pushed: its ridge bends the view
    if (bank > 0.0) {
        acc.cover = bank;
        acc.slope = clamp(-0.00005 * grad, vec2<f32>(-0.15), vec2<f32>(0.15)) * bank;
        acc.size_px = 8.0;
    }
    let w_main = select(select(1.0 - w_front, w_front, main_axis == 0), 1.0, roof);
    glass_drops(&acc, b, p, w_main, wet, heavy, bank, size, world, t);
    var side_w = p.side_w;
    var down_w = p.down_w;
    if (w_main < 0.99) {
        let p2 = pane_proj(b, 1 - main_axis);
        glass_drops(&acc, b, p2, 1.0 - w_main, wet, heavy, bank, size, world, t);
    }
    // --- the mist: droplets of a millimetre and less, too many to draw one by one far off
    var grain = 0.0;
    {
        let mg = rain_turn(p.q, 0.33) / 0.0024;
        let mc = floor(mg);
        let mh = rain_hash(mc + 71.0);
        let mr = 0.1 + 0.16 * mh.y;
        let at = mc + vec2<f32>(mr) + rain_hash(mc + 5.3) * (1.0 - 2.0 * mr);
        grain = step(mh.x, wet * 0.3) * (1.0 - smoothstep(mr * 0.6, mr, length(mg - at)));
    }
    // (seen from further than a millimetre a pixel: their average)
    let mist = mix(wet * 0.3 * 0.12, grain, smoothstep(0.0015, 0.0006, b.px));
    g.mist = clamp((mist + 0.6 * acc.avg) * (1.0 - acc.track), 0.0, 1.0);
    // The connected water between the drops scatters some of the view into a soft haze
    // (the veil, drawn from the picture's blurred levels), the same on every pane as wet.
    // A runner's fresh track is a clear sheet of water, sharper than the rest.
    let film = smoothstep(0.03, 0.4, wet) * (1.0 - 0.3 * acc.track);
    // (by the air's temperature, `camera.lod.w`: on cold glass the water spreads into a film
    // and a fine mist of droplets that scatter the view - near freezing the sleet's slush on
    // top - while in a warm summer rain it beads up and runs off, the glass between the drops
    // nearly clear)
    let cold = 1.0 - smoothstep(2.0, 24.0, camera.lod.w);
    // (light: a strong veil hid the drops behind a uniform fog)
    g.veil = film * mix(0.12, 0.25, heavy) * mix(0.7, 1.2, cold);
    g.cover = acc.cover;
    g.rim = acc.rim;
    g.size_px = acc.size_px;
    let slope = acc.slope;
    let nz = sqrt(max(1.0 - dot(slope, slope), 0.02));
    g.n = normalize(b.out * nz + side_w * slope.x + down_w * slope.y);
    return g;
}

// The picture behind the glass (`glass_mips.rs`, pre-exposed radiance) at `uv`, blurred to
// level `lod`: five reads round the point a texel of that level apart, so the small levels'
// texels do not show as blocks when they are spread over many pixels.
fn glass_blur(uv: vec2<f32>, lod: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(mx_tex2d[mx_mat.env]));
    let o = exp2(lod) * 0.75 / dims;
    let c = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));
    var s = textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], c, lod).rgb * 2.0;
    s = s + textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], clamp(c + vec2<f32>(o.x, o.y * 0.5), vec2<f32>(0.001), vec2<f32>(0.999)), lod).rgb;
    s = s + textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], clamp(c + vec2<f32>(-o.x * 0.5, o.y), vec2<f32>(0.001), vec2<f32>(0.999)), lod).rgb;
    s = s + textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], clamp(c + vec2<f32>(-o.x, -o.y * 0.5), vec2<f32>(0.001), vec2<f32>(0.999)), lod).rgb;
    s = s + textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], clamp(c + vec2<f32>(o.x * 0.5, -o.y), vec2<f32>(0.001), vec2<f32>(0.999)), lod).rgb;
    return s / 6.0;
}

// The snow a wiped pane holds while it snows (see enhanced/07_glass.wgsl `glass_snow`):
// `defrost` how far the windscreen's defroster has cleared it (0..0.85), `fz` the height on
// the glass in the bus's frame (-1 its bottom, 1 its top), for the drift along the bottom.
fn glass_snow_with(in: FsIn, cover: f32, defrost: f32, fz: f32) -> vec4<f32> {
    if (cover <= 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 0.0);
    }
    // (laid out on the glass as the drops are, `pane_proj`: the film's own coordinates
    // drew the crystals and the drifts out long on the windscreen)
    let b = pane_base(in.world, in.local, in.normal, false);
    let an = abs(b.nl);
    let axis = select(select(1, 0, an.y >= an.x), 2, an.z > max(an.x, an.y) * 2.0);
    let m = pane_proj(b, axis).q;
    let t = camera.post.y;
    let front = material.wipe_bounds.z > 0.0;
    let heat = select(0.15, 0.45 + 0.55 * clamp(defrost / 0.85, 0.0, 1.0), front);
    var flakes = 0.0;
    var melt_water = 0.0;
    for (var layer = 0; layer < 2; layer = layer + 1) {
        let fl = f32(layer);
        let cell = select(0.012, 0.022, layer == 1);
        let p = m / cell + fl * 37.0;
        let c = floor(p);
        let tm = rain_hash(c + vec2<f32>(5.0, 1.0 + fl));
        let life = mix(40.0, 7.0, heat) * (0.6 + 0.8 * tm.y);
        let cycle = t / life + tm.x;
        let ph = fract(cycle);
        let arrival = c + floor(cycle) * vec2<f32>(19.0, 47.0);
        let h = rain_hash(arrival);
        let h2 = rain_hash(arrival + vec2<f32>(91.0, 13.0));
        // (a crystal shows once the cover passes its own threshold: they come one by one)
        let shown = clamp((cover * select(1.15, 0.85, layer == 1) - h2.y) * 12.0, 0.0, 1.0) * smoothstep(0.0, 0.02, ph);
        // the second half of its while it melts: it rounds off, shrinks and goes clear
        let melt = smoothstep(0.45, 0.95, ph);
        let d = p - (c + 0.25 + 0.5 * h);
        let dl = length(d);
        let r = (0.16 + 0.2 * h2.x) * (1.0 - 0.5 * melt);
        // six arms, the rim reaching out along them
        let arm = pow(abs(cos(3.0 * atan2(d.y, d.x) + h.x * 6.283185)), 4.0);
        let rr = r * (0.55 + 0.45 * mix(arm, 1.0, melt));
        let edge = 1.0 - smoothstep(rr * 0.7, rr, dl);
        flakes = max(flakes, shown * edge * (0.6 + 0.4 * h.x) * (1.0 - melt));
        melt_water = max(melt_water, shown * edge * melt * (1.0 - smoothstep(0.92, 1.0, ph)));
    }
    // settled snow is never even: a smooth noise of a few centimetres and a finer one break
    // up where the layer closes and give the bank and the drifts their ragged edges
    let lumps = snow_noise(m / 0.045) * 0.65 + snow_noise(m / 0.012 + 7.0) * 0.35;
    var drift = 0.0;
    if (front) {
        // (by the height in the bus's own frame, not by the wiper mask's `wipe_uv`: a
        // windscreen's film is several slots - each wiper's field and the glass round them -
        // each with its own mask and its own 0 .. 1, and a bank laid along their edges drew
        // seams down the middle of the windscreen and round the wipers' arcs)
        if (camera.inside_c.w > 0.5) {
            let low = 1.0 - (fz * 0.5 + 0.5);
            drift = smoothstep(0.5, 0.75, low + 0.12 * (lumps - 0.5)) * smoothstep(0.1, 0.6, cover);
        }
    }
    let c = cover * (0.7 + 0.6 * lumps) * (1.0 - 0.3 * heat) + drift * 0.9;
    let sheet = smoothstep(0.5, 1.2, c) * 0.75 + smoothstep(1.0, 1.7, c) * 0.22;
    let white = mix(vec3<f32>(0.9, 0.92, 0.96), vec3<f32>(0.97, 0.98, 1.0), smoothstep(1.0, 1.8, c));
    let solid = max(flakes * 0.92, sheet);
    // (the melt water: a faint clear bead, the street a shade darker through it)
    let wet = melt_water * 0.16 * (1.0 - solid);
    let alpha = clamp(solid + wet, 0.0, 0.97);
    let col = (white * solid + vec3<f32>(0.5, 0.55, 0.6) * wet) / max(solid + wet, 1e-4);
    return vec4<f32>(col, alpha);
}

// The rain on a pane that has no film of its own to carry it (`glass_film`): the windows of
// the traffic's cars and buses, seen from the street. OMSI gives only some buses a rain film,
// and only the player's own wets it, so every other window stayed dry and clear in a downpour.
// The same drops and runners (`glass_water`), as wet as the roads (`camera.shadow.w`); none
// on the player's own bus, whose films carry its rain.
fn pane_rain(world: vec3<f32>, local: vec3<f32>, n: vec3<f32>) -> GlassWater {
    let wet = clamp(camera.shadow.w * 1.3, 0.0, 1.0);
    let own = near_player_vehicle(world) > 0.5;
    let far = distance(world, camera.cam_pos.xyz) > 80.0;
    let w = select(wet, 0.0, own || far);
    return glass_water(world, local, n, w, camera.post.y, false);
}

// The pane's colour `rgb` and cover `a` under that rain (`vn` towards the eye): the drops,
// small lenses darker than the glass round them with the sky mirrored on their domes
// (`refl`, what the dome mirrors), and between them the wet film, a little milky
// (`haze`, the daylight it scatters) and less see-through.
fn pane_rain_compose(g: GlassWater, rgb: vec3<f32>, a: f32, vn: vec3<f32>, refl: vec3<f32>, haze: vec3<f32>) -> vec4<f32> {
    let cosv = clamp(abs(dot(g.n, vn)), 0.0, 1.0);
    let f = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
    let drop = mix(rgb * 0.7, refl, clamp(f * 3.0, 0.15, 0.8));
    var c = mix(rgb, drop, g.cover * 0.85);
    let film = clamp(g.veil * 0.35 + g.mist * 0.2, 0.0, 0.5);
    c = mix(c, haze, film);
    let a2 = clamp(a + (1.0 - a) * (g.cover * 0.5 + film), a, 1.0);
    return vec4<f32>(c, a2);
}

@fragment
fn fs_main(in: FsIn) -> @location(0) vec4<f32> {
    var unused = 0.0;
    return shade_vanilla(in, &unused, camera.cam_pos.xyz);
}

// Shared with Enhanced and the wheel splash mask: pools spread as the road soaks.
fn road_puddle_coverage(world: vec3<f32>, normal: vec3<f32>, wet: f32) -> f32 {
    let xy = world_pattern_xy(world);
    let pn = vnoise_f(xy, 0.35, vec2<f32>(17.3, -9.1)) * 0.6
        + vnoise_f(xy, 1.6, vec2<f32>(-4.0, 8.0)) * 0.3
        + vnoise_f(xy, 5.0, vec2<f32>(2.7, 11.3)) * 0.1;
    // The pools spread from the lowest spots as the road soaks, but they stay pools: a road
    // wet through has standing water on about a fifth of it (PUDDLE_SPREAD in enhanced.wgsl,
    // the same in `omsi-app/src/puddles.rs`) and wet asphalt between.
    let threshold = 1.0 - wet * PUDDLE_SPREAD;
    return smoothstep(threshold - 0.02, threshold + 0.02, pn) * smoothstep(0.75, 0.95, normal.z);
}

@fragment
fn fs_vanilla_reflections(in: FsIn) -> EnhancedOut {
    var weight = 0.0;
    let c = shade_vanilla(in, &weight, camera.cam_pos.xyz);
    let coverage = select(c.a, 0.0, in.params2.w > 1.5);
    var out: EnhancedOut;
    out.color = c;
    out.mask = vec4<f32>(0.0, weight * 0.49, weight, coverage);
    return out;
}

fn shade_vanilla(in: FsIn, puddle_weight: ptr<function, f32>, eye: vec3<f32>) -> vec4<f32> {
    // (the pane's water, read once for the uses below)
    let film_water = window_wetness(in);
    if (material.emissive.w > 1.5) {
        // a pane's film of water: drops, not the sliding texture (`glass_film_classic`)
        return glass_film_classic(in, film_water);
    }
    var duv = tex_address(in.uv);
    if (material.extra.x > 0.5) {
        // terrain: uv is tile space; the ground texture repeats extra.z times per tile
        duv = in.uv * material.extra.z;
    }
    var tex = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, camera.lod.x), duv);
    if (is_snow_pane()) {
        // (as the enhanced picture lays it, `glass_snow_with`: no cabin air here, so no
        // defroster, and the height on the glass by the bus's level frame)
        let fz = clamp((in.world.z - camera.inside_a.z - camera.inside_c.z) / max(camera.inside_b.w, 0.1), -1.0, 1.0);
        tex = glass_snow_with(in, film_water, 0.0, fz);
    }
    // The texture coordinates without the [texcoordtransX/Y] offset: in Omsi.exe's
    // fixed-function pipeline the texture transform is the diffuse stage's alone, the
    // transmap, night map and light map stay in place under a scrolling roller blind.
    let buv = tex_address(in.uv - in.params.zw);
    if (material.extra.x > 0.5 && material.extra.y > 0.0) {
        // the ground texture's detail texture, repeated finer than the texture itself and
        // modulated over it as the original's terrain pass does. The stock detail maps are
        // bright (noise_low averages 242, gras_det 179), so they are meant to be multiplied
        // in plainly: doubling like a grey-centred D3D detail map blows the ground out.
        let det = textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], in.uv * material.extra.y, camera.lod.x);
        tex = vec4<f32>(clamp(tex.rgb * det.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), tex.a);
    }
    if (material.params.z > 0.5) {
        // [matl_transmap]: alpha comes from a separate map, its alpha channel (a map without one
        // is opaque, as D3D samples it: the WH UK AI cars' paint layer has a black 24-bit
        // `transmap_null.tga`, read as luminance the paint was invisible);
        // for terrain the map is the per-tile surface mask in tile space
        let tm = sample_transmap(buv);
        tex.a = select(1.0, tm.a, material.params.w > 0.5);
        if (material.extra.x > 0.5 && material.params.x > 1.5) {
            // A painted ground layer. The brush mask is coarse (0.6-3 m per texel) and
            // binary; the loader smooths it into a soft ramp around a smooth curve
            // (`smooth_paint_mask` in scene.rs). Sharpen only that coverage ramp:
            // diffuse/detail mip colours change with viewing angle, and including
            // their luminance made covered ground fade into lower layers at a slant.
            tex.a = smoothstep(0.32, 0.68, tex.a);
        }
    }
    // One night-map read supplies both coverage and lighting. Keep the tile map
    // in its north-up UVs; vehicle maps use the unscrolled material UVs.
    var night_sample = vec4<f32>(0.0);
    if (material.extra.w > 0.5) {
        let nuv = select(buv, vec2<f32>(in.uv.x, 1.0 - in.uv.y), material.extra.x > 0.5);
        night_sample = sample_nightmap(nuv);
    }
    tex.a = night_coverage(tex.a, night_sample.a) * tree_card_fade(in.world);
    let mode = material.params.x;
    // Mip-filtered alpha is the fraction of the pixel covered by leaves or fence wires.
    // Tighten the transition around the cutout edge before MSAA turns it into sample
    // coverage. The depth prepass leaves these draws out so its binary cutoff cannot hide
    // the scene behind samples that the colour pass leaves open.
    if (ALPHA_TEST && mode > 0.5 && mode < 1.5) {
        if (ALPHA_TO_COVERAGE) {
            let aa = max(fwidth(tex.a) * 0.5, 1.0 / 255.0);
            // (not `ALPHA_REF - aa`: a layer whose clear glass is alpha 128 kept a third of
            // its samples there, a moire of the layer's paint over every window)
            if (tex.a < ALPHA_REF) {
                discard;
            }
            tex.a = smoothstep(ALPHA_REF - aa, ALPHA_REF + aa, tex.a);
        } else if (tex.a < ALPHA_REF) {
            discard;
        }
    }
    // (a blended slot that keeps Omsi.exe's alpha test, `MaterialExtra::alpha_cut`: a
    // livery's clear glass is cut away, its tinted glass blended)
    tex.a = select(tex.a, 0.0, mode > 2.5 && tex.a < ALPHA_REF);
    let n = safe_normal(in.normal);
    let ndl = max(dot(n, camera.sun_dir.xyz), 0.0);
    // three lights like the original: direct sun (A), light from above (B), ambient (C)
    let from_above = 0.5 + 0.5 * n.z;
    let shadow = sun_shadow(in.world, n, mode > 0.5 && mode < 1.5);
    // screen-space ambient occlusion darkens the indirect light (sky and ambient) in
    // corners, under the bus, between the seats - not the sun, which the shadow map handles
    var ao = 1.0;
    // (not on a blended surface: the AO is the opaque depth's, and a translucent door
    // showed the shade of what stood behind it)
    if (camera.clouds.w > 0.5 && mode < 1.5) {
        ao = ao_at(in.clip.xy, in.world);
    }
    var diffuse = camera.sun_color.rgb * camera.sun_dir.w * ndl * shadow + (camera.sky_color.rgb * from_above * (0.6 + 0.4 * shadow) + camera.ambient.xyz) * ao;
    var albedo = tex.rgb;
    if (camera.flags.x > 0.5 && (material.extra.x > 0.5 || in.params2.w > 0.5)) {
        // detail texturing (setting): procedural grain on the ground and the roads, fading out with distance
        let dist0 = distance(in.world, camera.cam_pos.xyz);
        let k = clamp(1.0 - (dist0 - 25.0) / 120.0, 0.0, 1.0);
        let nz = detail_noise(world_pattern_xy(in.world), k);
        albedo = albedo * (1.0 + (nz - 0.5) * 0.42 * k);
    }
    // ([nomaplighting]: params.y 0.25 - not lit by the map's lamps)
    // (a light-mapped road or plate takes the map's lamps only in Vanilla+: as OMSI 2 shows
    // it, the vanilla picture lights it from the tile light map alone)
    let lm_only = light_map_mapped(material.params) && camera.sky_color.w > 0.5;
    // (and a [tree]'s leaf cards, params.y 0.15: OMSI 2 leaves a tree dark even right under
    // a street lamp, where the lamp's 40 m core lit the crown up yellow-green)
    let tree_unlamped = camera.sky_color.w > 0.5 && material.params.y > 0.1 && material.params.y < 0.2;
    let map_lamps = select(1.0, 0.0, (material.params.y > 0.2 && material.params.y < 0.3) || lm_only || tree_unlamped);
    let lamp_light = point_lights(in.world, n, map_lamps);
    // The headlight is lit at the vertices: see `classic_spot`.
    var light = diffuse + lamp_light + in.spot_diffuse + in.spot_ambient;
    // D3D lights the material's diffuse colour with the sun, the light from above and the
    // lamps, and its ambient colour with the ambient light (C) and the headlight's ambient
    // light. Omsi.exe makes every o3d slot's ambient white (0x7c62f8), so a white sign
    // texture on a green material shows white in the shade, not green.
    let ambient_light = camera.ambient.xyz * ao + in.spot_ambient;
    let mat_light = material.color.rgb * (light - ambient_light) + material.ambient.rgb * ambient_light;
    let light_mapped = material.params2.x > 0.5 && material.extra.x < 0.5;
    // The low-range Vanilla+ target has no exposure/tonemapping. Saturate the
    // diffuse light before modulating the texture, as the classic path does:
    // overlapping lamps must not clip a coloured texture into a white patch.
    // The light is OMSI's - the envir.cfg's sun, sky and ambient and its lamps - and those
    // numbers are what Direct3D multiplies into the gamma-encoded texture: it is laid on the
    // encoded texture here too, as in the classic picture below. Multiplied in linear light,
    // a light L showed as L^(1/2.2): the night's 0.06 as 0.28, every night, dusk and shade
    // in Vanilla+ several times as bright as in OMSI 2 (#300, #1612).
    var lit = srgb_decode(srgb_encode(albedo) * clamp(mat_light, vec3<f32>(0.0), vec3<f32>(1.0)));
    // The vanilla picture: Omsi.exe's texture stages multiply the gamma-encoded texture by
    // the vertex light (clamped at 1); here the texture is sampled linear and the target
    // encodes again, so multiplied here a light L showed as L^(1/2.2) - a night at 0.06
    // looked like 0.28, a late dusk (#300). So the product is made on the encoded texture
    // and decoded again, with the light map laid on encoded as well. (Through the sRGB curve
    // itself: taken as a power of 2.2, t v^2.2, the curve's linear foot below 0.0031 put a
    // night wall at a quarter of OMSI 2's - a texture of 0.66 under a light of 0.05 came out
    // at 2 of 255 instead of 8.)
    let classic = camera.sky_color.w > 0.5;
    // (the terrain's night map is its tile light map: light, not a glow - see below)
    let terrain_map = material.params.y < 0.5
        && material.extra.x > 0.5
        && material.extra.w > 0.5
        && material.extra.w < 1.5;
    let terrain_night = classic && terrain_map;
    if (classic && material.params.y < 0.5) {
        var v = clamp(material.emissive.rgb + mat_light + material.color.rgb * interior_lamps(in.world, n, in.params2.z), vec3<f32>(0.0), vec3<f32>(1.0));
        if (light_mapped) {
            let lm = srgb_encode(textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], buv, camera.lod.x).rgb) * clamp(in.params2.x, 0.0, 1.0);
            v = v + lm * (vec3<f32>(1.0) - v);
        }
        if (terrain_night) {
            // the terrain's tile light map lights the ground as the lamps' own light: added
            // to the vertex light before the texture is multiplied in. Laid over the lit
            // ground instead, as a night map glows, its faint fringe (0.01, linear) put
            // one beige veil over cobbles and grass alike, a whole car park the colour of
            // sand where OMSI 2 shows it dark grey.
            let nm = srgb_encode(night_sample.rgb);
            v = min(v + nm * camera.sun_color.w * clamp(in.params2.y, 0.0, 1.0), vec3<f32>(1.0));
        }
        if (lm_only) {
            // a road, a plate or a [LightMapMapping] object: the same tile light map, the
            // same way as the ground beside it (Omsi.exe lights both from it alone - its
            // `[maplight]`s only bake the map, 0x7903e0). Added after the texture in linear
            // light instead, the pools that lay bright on the verges hardly showed on the
            // asphalt between them: the street lamps lit everything but the road (#847).
            let lm = srgb_encode(light_map_at(in.world));
            v = min(v + lm * camera.sun_color.w, vec3<f32>(1.0));
        }
        lit = srgb_decode(srgb_encode(albedo) * v);
    } else if (light_mapped) {
        // [matl_lightmap], as Omsi.exe's texture stages have it (0x7fe4d3..0x7fe604): the
        // light map is laid onto the vertex light with D3DTOP_ADDSMOOTH (light + map x (1 -
        // light)) before the texture is multiplied in - a lit saloon glows at night and
        // hardly shows in daylight. The vertex light is D3D's: the material's emissive and
        // its colour times every light - the saloon lamps (`[interiorlight]`, D3D lights
        // too, 0x5fa8f0) among them - clamped at 1. (Added once more after the map, the
        // saloon lamps lit a cabin twice over, flat white where the map was full; and the
        // material's colour took the map down with it.)
        let lm = srgb_encode(textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], buv, camera.lod.x).rgb) * clamp(in.params2.x, 0.0, 1.0);
        let v = clamp(material.emissive.rgb + mat_light + material.color.rgb * interior_lamps(in.world, n, in.params2.z), vec3<f32>(0.0), vec3<f32>(1.0));
        lit = srgb_decode(srgb_encode(albedo) * (v + lm * (vec3<f32>(1.0) - v)));
    }
    // the D3D material's own highlight (specular colour and power of the o3d file or a
    // [matl_allcolor]), lit at the vertices (see `vertex_specular`) and added after the
    // texture as D3D's specular is; the sun's not in its shadow
    let specular = in.spec_sun * shadow + in.spec_sky;
    if (classic) {
        // D3D adds its vertex specular after the encoded texture stages. Adding
        // it in linear light encoded it again at the target and washed out dusk.
        lit = srgb_decode(min(srgb_encode(lit) + specular, vec3<f32>(1.0)));
    } else {
        lit = lit + specular;
    }
    if (material.params.y > 0.5) {
        lit = tex.rgb * material.color.rgb;
    }
    if ((!light_mapped && !classic) || material.params.y > 0.5) {
        lit = lit + tex.rgb * material.emissive.rgb;
    }
    // [interiorlight]: the saloon lamps on the meshes and passengers they illuminate (in
    // a light-mapped material's vertex light already, above)
    if (!light_mapped && !classic) {
        lit = lit + tex.rgb * interior_lamps(in.world, n, in.params2.z);
    }
    // (Vanilla+: the map's lamps light the ground, as they light the roads)
    if (material.extra.w > 0.5 && !terrain_map) {
        // [matl_nightmap]: self-illumination that fades in with the night
        // terrain: the tile light map in tile space (north at the top row)
        // a [matl_item] night map is switched by its variable (warning lamps, displays):
        // it glows whenever that is on, by day as well; the others fade in with the night
        let night = select(rain_lit_windows(in.local), 1.0, material.extra.w > 1.5);
        lit = lit + night_sample.rgb * night * select(clamp(in.params2.y, 0.0, 1.0), 1.0, material.extra.w > 1.5);
    }
    if (material.params2.y > 0.0) {
        // [matl_envmap]: sphere map reflection, masked by the diffuse alpha like the original
        let vdir = normalize(in.world - eye);
        var env_uv = in.env_uv;
        if (material.bump.y > 0.5) {
            env_uv = env_uv + bump_offset(duv);
        }
        let env = textureSample(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], env_uv);
        let diffuse_a = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, camera.lod.x), duv).a;
        if (material.params.x < 1.5) {
            // rain: a painted body goes darker and glossier when it is wet, so the bus
            // stands out against a grey street instead of fading into it
            let env_night = select(1.0 - 0.85 * clamp(camera.sun_color.w, 0.0, 1.0), 1.0, camera.sky_color.w > 0.5);
            let env_light = clamp(camera.sun_color.r * camera.sun_dir.w + camera.sky_color.r + camera.ambient.r, 0.05, 1.0) * env_night;
            let wet = camera.shadow.w * weather_outside_n(in.world, n, false, in.params2.w);
            lit = lit * (1.0 - 0.30 * wet);
            // the water film mirrors the (overcast) sky a little, mostly at grazing
            // angles; kept faint - a strong rim read as a white outline round the bus
            let facing = clamp(abs(dot(vdir, n)), 0.0, 1.0);
            let sheen = wet * min(material.params2.y, 1.0) * (0.05 + 0.22 * pow(1.0 - facing, 5.0));
            lit = mix(lit, camera.sky_color.rgb * env_light, sheen);
        }
        // Omsi.exe's environment stage (0x7fd6c4 at 0x7ff626, fixed function), for paint
        // and glass alike: the sphere map's colour as it is, over what the stages before
        // made of the lit texture - D3DTOP_LERP by the texture factor when the material has
        // neither a [matl_transmap] nor a [matl_envmap_mask], D3DTOP_BLENDCURRENTALPHA when
        // it has one - and the alpha left as the stages before made it (ALPHAARG1 CURRENT,
        // SELECTARG1, 0x7ff98a): a clear pane shows its reflection as faintly as it shows
        // itself. (A glass path of our own - a quarter reflection at least, a Fresnel rim
        // and the pane made opaque where it reflected - mirrored the street in every
        // window far more than the original does.) The factor is [matl_envmap]'s times
        // the ambient light plus `g` (the light the vehicle stands in: the lamps of the
        // tile's light map and the sky), each at most 1, packed into a D3DCOLOR without
        // saturation (0x7ff8ed: a factor over 1 spills its bits into the next channel, as
        // there). The alpha it blends by is the mask's, or the diffuse texture's times `g`
        // (0x7feacc, 0x7ff092) - never the diffuse alpha of a material without one of
        // them: taken as a reflection mask, a body whose texture carries an alpha channel
        // for other purposes was mirrored (dark) where that alpha was high.
        // `g` (0x861ca0, set per vehicle at 0x7d8735): the mean of the light the tile's
        // light map throws on it plus the day's light A (weather +0xac = the mean of
        // lightcolor A, 0x75333f - by it the stars fade), at most 1
        let g = clamp(dot(lamp_light, vec3<f32>(1.0 / 3.0)) + dot(camera.sun_color.rgb, vec3<f32>(1.0 / 3.0)), 0.0, 1.0);
        var kk = vec3<f32>(0.0);
        if (has_transmap_declared() || has_env_mask()) {
            let a = select(diffuse_a, textureSampleBias(mx_tex2d[mx_mat.envmask], mx_samplers[mx_mat.sampler], duv, camera.lod.x).a, has_env_mask());
            kk = vec3<f32>(a * g);
        } else {
            kk = omsi_texture_factor(material.params2.y, camera.ambient.rgb + vec3<f32>(g));
        }
        // The lerp is made on the encoded values, as the stage makes it, like the texture x
        // light product above: made on linear ones it showed the reflection about twice as
        // bright over a dark surface - since the nights got dark (#300) the instrument glass
        // of the MAN NL/NG (Fenster.tga, factor 0.5, the sky at the sphere map's bottom) lay
        // milky white over the unlit gauges. (Vanilla+ as well: its linear lerp laid the
        // sphere map as a grey sheen over every dark window band and chassis, #780.)
        // The sphere map's colour as it is, as the stage takes it: the night already weakens
        // the reflection through the factor (`kk`, the ambient light and `g`). Darkened once
        // more by that light, as a photograph lit by the night, the reflection went twice
        // and the paint and the windows lost their shine at dusk and by night, which they
        // keep in OMSI (#2284).
        lit = srgb_decode(mix(srgb_encode(lit), srgb_encode(env.rgb), kk * camera.cam_pos.w));
    }
    // wet road: a surface whose texture carries [moisture] darkens under rain and starts
    // to mirror the sky, strongest where you look along it (the Fresnel sheen that makes a
    // wet street read as wet)
    let outside = weather_outside_n(in.world, n, material.extra.x > 0.5, in.params2.w);
    let wet = camera.shadow.w * material.params2.z * outside * (1.0 - clamp(camera.ambient.w, 0.0, 1.0));
    if (wet > 0.0) {
        let vdir = normalize(in.world - eye);
        let facing = clamp(-dot(vdir, n), 0.0, 1.0);
        let fresnel = pow(1.0 - facing, 4.0);
        let puddle = road_puddle_coverage(in.world, n, wet);
        let water = 0.02 + 0.98 * pow(1.0 - facing, 5.0);
        let weight = clamp(mix(fresnel * wet * 0.85, water * wet, puddle), 0.0, 0.9);
        let sheen = camera.sky_color.rgb * 0.5 + camera.sun_color.rgb * camera.sun_dir.w * 0.35;
        if (classic) {
            // Vanilla blends encoded weather colours, but asphalt darkening
            // must stay in linear light to avoid turning the road black.
            lit = srgb_decode(mix(srgb_encode(lit * mix(1.0, 0.75, wet)), sheen, weight));
        } else {
            lit = mix(lit * mix(1.0, 0.55, wet), sheen, weight);
        }
        *puddle_weight = weight * puddle;
    }
    // snow: the ground, the roads and every upward-facing surface whiten under it
    // (not on a shadow blob: whitened, it lit the snow under the bus instead of shading it)
    // (not in vanilla: OMSI 2 shows snow only through the season's WinterSnow textures)
    // (nor on a texture that is the season's snow picture: the map's own WinterSnow
    // textures show the snow as OMSI 2 does, and whitened over, the snowy grass and the
    // grey road went one flat white, the lane markings left standing in it, #879)
    let snow_ok = outside * select(1.0, 0.0, in.params2.w > 1.5 || camera.sky_color.w > 0.5 || material.ambient.w > 0.5);
    // a carriageway (a [moisture] texture that is not the terrain) under snow that builds
    // up: its own cover, the ruts of the lanes and the tyres' tracks (road_snow.wgsl)
    let carriageway = material.extra.x < 0.5 && material.params2.z > 0.0 && road_snow_dynamic();
    var rs = vec3<f32>(0.0);
    if (carriageway && snow_ok > 0.0) {
        rs = road_snow(in.world);
    }
    // a vehicle's part: the snow its own roof has gathered, where that is kept (another
    // player's bus keeps the weather's)
    let vehicle = vehicle_snow_kept(in.params2.w);
    var on_vehicle = 0.0;
    if (vehicle && snow_ok > 0.0) {
        on_vehicle = vehicle_snow(in.params2.w, in.local);
    }
    let snow = select(select(camera.ambient.w, rs.x, carriageway), on_vehicle, vehicle) * snow_ok;
    if (snow > 0.0 || rs.y > 0.0) {
        let up = clamp(n.z, 0.0, 1.0);
        let ground = select(0.0, 1.0, material.extra.x > 0.5 || material.params2.z > 0.0);
        // (a road kept clear - "snow on road" off - stays asphalt, #1362; built-up snow
        // brings its own cover)
        let cleared = select(1.0, 0.0, !carriageway && camera.post.z > 0.5 && material.extra.x < 0.5 && material.params2.z > 0.0);
        // only surfaces that really face up get a cover; a soft threshold keeps the snow
        // off the sides and off the grazing rims that showed as a white outline
        let cover = cleared * snow * clamp(max(ground, smoothstep(0.78, 0.95, up) * 0.8), 0.0, 1.0);
        let light = camera.sun_color.rgb * camera.sun_dir.w * ndl * shadow * 0.6 + camera.sky_color.rgb * 0.7 + camera.ambient.xyz;
        let white = vec3<f32>(0.92, 0.94, 0.98) * light * ao;
        // (a vehicle's gathered snow lies thick enough to hide the paint: the ground's
        // dusting lets the texture show through)
        lit = mix(lit, white, min(cover * select(0.55 + 0.35 * tex.a, 1.2, vehicle), 1.0));
        // the slush in the ruts and the tracks: wet, grey, darker than the snow beside it
        lit = mix(lit, lit * 0.6 + white * 0.12, rs.y * snow_ok);
    }
    let dist = distance(in.world, camera.cam_pos.xyz);
    let f = 1.0 - exp(-fog_distance(in.world) * camera.fog.w);
    *puddle_weight *= 1.0 - clamp(f, 0.0, 1.0);
    var rgb = mix(lit, camera.fog.xyz, clamp(f, 0.0, 1.0));
    if (classic) {
        // D3D's fixed-function fog blends the encoded vertex fog colour too.
        // Treating that colour as linear raised a mid-grey fog from 128 to 188.
        rgb = srgb_decode(mix(srgb_encode(lit), camera.fog.xyz, clamp(f, 0.0, 1.0)));
    }
    if (camera.flags.z > 0.0) {
        // Never taken: flags.z (the old enhanced look's aerial perspective) is always 0
        // now - the enhanced path has its own fragment shader (enhanced.wgsl). The branch
        // stays because Metal's fast-math contracts the fog mix above differently without
        // it, and the vanilla picture is to stay what it was to the last code value.
        let vdir = normalize(in.world - camera.cam_pos.xyz);
        let towards_sun = clamp(dot(vdir, camera.sun_dir.xyz), 0.0, 1.0);
        let k = (1.0 - exp(-dist * 0.00045)) * camera.flags.z * (0.55 + 0.45 * clamp(camera.sun_dir.w, 0.0, 1.0));
        let scatter = mix(camera.sky_color.rgb * 1.4 + camera.ambient.rgb * 0.5, camera.sun_color.rgb * camera.sun_dir.w + camera.sky_color.rgb, towards_sun * towards_sun);
        rgb = mix(rgb, scatter, clamp(k * 0.6, 0.0, 0.5));
    }
    // (a NaN anywhere above - a zero-length vector normalised, 0 x infinity - was a black
    // patch on Windows, e.g. an EN92's headlights while off, and invisible on the Mac)
    rgb = finite_or(rgb, finite_or(albedo * material.color.rgb * diffuse, albedo));
    var a = tex.a * material.color.a;
    if (mode < 0.5) {
        a = 1.0;
    }
    if (!is_snow_pane()) {
        a = a * clamp(film_water, 0.0, 1.0);
    }
    if (mode > 1.5 && min(material.emissive.r, min(material.emissive.g, material.emissive.b)) > 0.99 && material.emissive.w < 1.5) {
        a = encoded_blend_alpha(rgb, a, light);
    }
    if (material.emissive.w > 0.5 && material.emissive.w < 1.5) {
        // a pane in the rain without a film of its own (`pane_rain`): the traffic's windows
        let g = pane_rain(in.world, in.local, in.normal);
        if (g.cover > 0.001 || g.veil > 0.001 || g.mist > 0.001) {
            let vn = normalize(camera.cam_pos.xyz - in.world);
            let o = pane_rain_compose(g, rgb, a, vn, rain_env_vanilla(reflect(-vn, g.n)), rain_env_vanilla(vec3<f32>(0.0, 0.0, 0.5)) * 0.9);
            rgb = o.rgb;
            a = o.a;
        }
    }
    return vec4<f32>(rgb, a);
}

// What lies under a self-lit layer, for `encoded_blend_alpha`: a surface of this encoded
// brightness lit by the light at the layer (the road's asphalt under the KVB Citaro's
// light cone is 0.44 - 0.49 encoded).
const BLEND_GROUND_ENCODED: f32 = 0.45;

// A self-lit blended layer (emissive white: a light cone laid on the road, a glow) as
// Omsi.exe blends it: Direct3D 9 mixes the gamma-encoded colours, the sRGB target here mixes
// linear ones, and a faint layer came out far brighter - the KVB Citaro's `lichtkegel.o3d`
// (alpha 0.14 on average) a flat beige sheet over the road where OMSI 2 shows a soft pool,
// 120 of 255 against its 85. The target cannot be read, so what is under the layer is taken
// to be ground lit by the light here (`light`, the vertex light the layer would have):
// the alpha is the one that gives Omsi.exe's encoded mix over it, never more than the
// layer's own.
fn encoded_blend_alpha(rgb: vec3<f32>, a: f32, light: vec3<f32>) -> f32 {
    let w = vec3<f32>(0.2126, 0.7152, 0.0722);
    let s = max(rgb, vec3<f32>(0.0));
    if (a <= 0.0) {
        return a;
    }
    let d_enc = clamp(light, vec3<f32>(0.0), vec3<f32>(1.0)) * BLEND_GROUND_ENCODED;
    let d = dot(srgb_decode(d_enc), w);
    let over = dot(srgb_decode(srgb_encode(s) * a + d_enc * (1.0 - a)), w);
    let span = dot(s, w) - d;
    if (span < 1e-4) {
        return a;
    }
    return clamp((over - d) / span, 0.0, a);
}

// The rain film on a pane in the classic pictures: the drops, the runners and the blur of the
// wet glass as the enhanced picture draws them (`glass_water`, enhanced/07_glass.wgsl
// `glass_film`), lit by the classic picture's own sky (`rain_env_vanilla`). Without the
// cabin's air there is no condensation behind them.
fn glass_film_classic(in: FsIn, water: f32) -> vec4<f32> {
    let v = camera.cam_pos.xyz - in.world;
    let dist = length(v);
    let vn = v / max(dist, 1e-4);
    let in_cab = inside_vehicle(camera.cam_pos.xyz) * near_player_vehicle(in.world) > 0.5;
    let g = glass_water(in.world, in.local, in.normal, water, camera.post.y, in_cab);
    if (g.cover <= 0.001 && g.mist <= 0.001 && g.veil <= 0.001) {
        return vec4<f32>(0.0);
    }
    let has_picture = camera.flags.z < -0.5;
    let dims = vec2<f32>(textureDimensions(mx_tex2d[mx_mat.env]));
    let uv0 = in.clip.xy / dims;
    let mist_col = rain_env_vanilla(vec3<f32>(0.0, 0.0, 0.5)) * 0.9;
    // the view through the wet glass between the drops, softened by the film of water
    let veil_lod = log2(max(dims.y * mix(0.002, 0.006, clamp(g.veil / 0.3, 0.0, 1.0)), 1.0));
    var behind = mist_col;
    if (has_picture) {
        behind = finite_or(glass_blur(uv0, veil_lod), mist_col);
    }
    // a drop: a small lens, the street small and upside down in it, darker at its rim
    var drop_col = behind;
    var a_drop = 0.0;
    if (g.cover > 0.001) {
        // (bent by the dome, the way out of the drop is looked up where it meets the street a
        // few metres on - not a fixed share of the screen aside, which from the driver's seat
        // landed on the dark dashboard and the pillars and drew every drop as a black speck:
        // the classic pictures keep no depth to tell those from the street)
        let tilt = g.n - g.out * dot(g.n, g.out);
        let thr = normalize(-vn - tilt * 0.9);
        var seen = behind;
        let cp = camera.view_proj * vec4<f32>(in.world + thr * 6.0, 1.0);
        if (has_picture && cp.w > 0.05) {
            let ndc = cp.xy / cp.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            let inside = smoothstep(0.0, 0.06, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
            let far = finite_or(glass_blur(clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.8 + 1.5 * g.mist), behind);
            seen = mix(behind, far, inside);
        }
        let rim = smoothstep(0.55, 1.0, g.rim);
        let facing = dot(g.out, vn) > 0.0;
        let cosv = clamp(abs(dot(g.n, vn)), 0.0, 1.0);
        let f = select(0.02, 0.02 + 0.98 * pow(1.0 - cosv, 5.0), facing);
        var c = seen * (1.0 - 0.45 * rim) * (1.0 - f) + rain_env_vanilla(reflect(-vn, g.n)) * f;
        if (facing) {
            let r = reflect(-vn, g.n);
            c = c + camera.sun_color.rgb * camera.sun_dir.w * pow(max(dot(r, camera.sun_dir.xyz), 0.0), 900.0) * f * 60.0;
        }
        let detail = smoothstep(0.5, 1.6, g.size_px);
        drop_col = mix(behind * 0.93, c, detail);
        a_drop = g.cover * mix(select(0.15, 0.5, has_picture), 0.95, detail) * (1.0 - smoothstep(5.0, 15.0, dist));
    }
    let a_mist = g.mist * 0.5;
    var a_veil = 0.0;
    if (g.veil > 0.001 && has_picture) {
        a_veil = g.veil * (1.0 - smoothstep(25.0, 60.0, dist));
    }
    let a_under = a_veil + a_mist * (1.0 - a_veil);
    let under = (behind * a_veil + mix(mist_col, behind, 0.9) * a_mist * (1.0 - a_veil)) / max(a_under, 1e-4);
    let a = a_drop + a_under * (1.0 - a_drop);
    if (a <= 1e-4) {
        return vec4<f32>(0.0);
    }
    let col = (drop_col * a_drop + under * a_under * (1.0 - a_drop)) / a;
    return vec4<f32>(min(col, vec3<f32>(1.5)), a);
}

// Snow on the carriageways: how far it has built up, the ruts the traffic keeps in it and
// the fresh tracks the tyres press into it (see `Renderer::set_snow_track_rect`).
//
// The field is a square of `place.z` metres that wraps around the world (a torus): texel
// (i, j) holds the world's point (x, y) where floor(x / texel) and floor(y / texel) are i and
// j modulo its size, so the app only rewrites the tiles the camera moves into. It is kept
// tile after tile (`TRACK_TILE` texels a side), each row after row, a texel an RGBA8 word:
//   r: the ruts of the lanes' wheel tracks (0..1)
//   g: a tyre has pressed the snow here (how much of the texel it covered)
//   b, a: how much snow had fallen when a tyre last ran here (16 bits, see `state.y`; for
//         a rut no tyre has run in yet, a moment as long ago as its lane's traffic makes
//         likely): the track and the rut fill as more falls

struct SnowTrack {
    // xy: the render origin modulo the field's side (m), z: the side (m), w: 1 while the
    // field holds the area around the camera
    place: vec4<f32>,
    // x: how far the snow covers the roads (0..1; below 0 the weather's "snow on road"
    // decides, as before), y: the snow fallen so far (thousandths of a full cover, modulo
    // 65536), z: how deep the lanes' ruts show (0..1), w: how far it thaws to slush (0..1;
    // Enhanced alone, 0 on the other paths)
    state: vec4<f32>,
};

@group(0) @binding(20) var<storage, read> snow_track_px: array<u32>;
@group(0) @binding(21) var<uniform> snow_track: SnowTrack;

const TRACK_TEXELS: i32 = 2048;
const TRACK_TILE: i32 = 256;

// One texel's rut and fresh track, each filled by the snow fallen since a tyre last ran
// over it.
fn snow_track_texel(p: vec2<i32>) -> vec2<f32> {
    let q = ((p % TRACK_TEXELS) + TRACK_TEXELS) % TRACK_TEXELS;
    let tile = (q.y / TRACK_TILE) * (TRACK_TEXELS / TRACK_TILE) + q.x / TRACK_TILE;
    let i = (tile * TRACK_TILE + q.y % TRACK_TILE) * TRACK_TILE + q.x % TRACK_TILE;
    let t = unpack4x8unorm(snow_track_px[i]);
    let then = round(t.b * 255.0) * 256.0 + round(t.a * 255.0);
    let since = (snow_track.state.y - then + 65536.0) % 65536.0;
    // (a track begins to blur after 8 % of a cover has fallen on it and is gone at 28 %; a
    // rut, worn deeper by the traffic, begins to fill at 15 % and is gone at 52 % - in a
    // heavy snowfall (Starker Schneefall) one or two minutes, and four or seven, without a
    // car: a driver who waits that long at a stop sees it happen)
    let fresh = t.g * (1.0 - smoothstep(80.0, 280.0, since));
    let rut = t.r * (1.0 - smoothstep(150.0, 520.0, since));
    return vec2<f32>(rut, fresh);
}

// The rut (x) and the fresh track (y) at a point of the render frame, blended between the
// four texels around it (each decoded first: the fill time does not blend).
//
// Faded out from 75 m off and not read beyond 90: the tiles the field takes as the camera
// moves lie 96 m off and more, and hold what they held before until their ruts are drawn
// a few frames later (see `SnowTracks::update`) - and a rut that far is a pixel or two.
fn snow_track_at(world: vec2<f32>) -> vec2<f32> {
    let fade = 1.0 - smoothstep(75.0, 90.0, distance(world, camera.cam_pos.xy));
    if (snow_track.place.w < 0.5 || fade <= 0.0) {
        return vec2<f32>(0.0);
    }
    let q = (world + snow_track.place.xy) * (f32(TRACK_TEXELS) / snow_track.place.z) - 0.5;
    let b = floor(q);
    let f = q - b;
    let i = vec2<i32>(b);
    let a00 = snow_track_texel(i);
    let a10 = snow_track_texel(i + vec2<i32>(1, 0));
    let a01 = snow_track_texel(i + vec2<i32>(0, 1));
    let a11 = snow_track_texel(i + vec2<i32>(1, 1));
    return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y) * fade;
}

// Whether a part's surface code carries its roof's own snow (`vehicle_snow`) rather than
// leaving it to the weather's (a roof the roads' snow does not keep: another player's bus).
fn vehicle_snow_kept(code: f32) -> bool {
    return code < -10000.0 && road_snow_dynamic();
}

// The snow on a vehicle's part (`code`: its instance's surface code, whose ten thousands
// carry how much snow its roof has gathered, 1..41 for 0..1, see upload.rs `roof_code`),
// at `local`, the point in the part's own mesh: patches at first, a closed layer as it
// goes on, laid out on the vehicle so that they ride with it.
fn vehicle_snow(code: f32, local: vec3<f32>) -> f32 {
    let amount = (floor(-code / 10000.0) - 1.0) / 40.0;
    let level = amount * 1.3 - 0.12;
    // (bare, or closed over whatever the patches: no noise to sample)
    if (amount <= 0.0) {
        return 0.0;
    }
    if (level >= 1.08) {
        return 1.0;
    }
    let p = local.xy;
    let p2 = vec2<f32>(p.x * 0.8 - p.y * 0.6, p.x * 0.6 + p.y * 0.8);
    let patches = vnoise_f(p, 0.9, vec2<f32>(4.3, 1.9)) * 0.65 + vnoise_f(p2, 3.1, vec2<f32>(8.1, 6.7)) * 0.35;
    return smoothstep(patches - 0.3, patches + 0.08, level);
}

// Whether the roads' snow is the built-up kind (`state.x` at or above 0) rather than the
// weather's on/off "snow on road".
fn road_snow_dynamic() -> bool {
    return snow_track.state.x >= 0.0;
}

// The snow on a carriageway at `world` (render frame): x the cover (0..1), y the slush in
// the ruts (0..1: the wheels keep it wet and grey), z a fresh tyre track (0..1).
fn road_snow(world: vec3<f32>) -> vec3<f32> {
    let base = clamp(snow_track.state.x, 0.0, 1.0);
    if (base <= 0.0) {
        return vec3<f32>(0.0);
    }
    // the map's own coordinates (modulo the pattern's period): the patches stay put
    let m = world.xy + camera.world_origin.xy;
    // It does not settle evenly: first a dusting in patches a stride or two across and in
    // the grain of the asphalt, which close up as it goes on falling.
    // (three octaves, the finer ones turned against the coarse: a single lattice's cells
    // showed as square blots where the cover's edge ran through them)
    // (once the cover has closed - its level over the patches' highest - every pixel of the
    // road paid for three octaves of noise that changed nothing)
    let m2 = vec2<f32>(m.x * 0.8 - m.y * 0.6, m.x * 0.6 + m.y * 0.8);
    let level = base * 1.35 - 0.15;
    var cover = 1.0;
    if (level < 1.1) {
        let m3 = vec2<f32>(m.x * 0.28 + m.y * 0.96, m.y * 0.28 - m.x * 0.96);
        let patches = vnoise_f(m, 0.45, vec2<f32>(3.1, 7.7)) * 0.55 + vnoise_f(m2, 1.3, vec2<f32>(11.3, 2.9)) * 0.3 + vnoise_f(m3, 3.7, vec2<f32>(5.9, 13.1)) * 0.15;
        cover = smoothstep(patches - 0.35, patches + 0.1, level);
    }
    let tracks = snow_track_at(world.xy);
    // The ruts: the wheels of the traffic keep two strips of each lane down to slush, the
    // more of it the more snow there is to churn; with a mere dusting they show the asphalt.
    let rut = tracks.x * snow_track.state.z;
    let slush = rut * smoothstep(0.15, 0.7, base);
    cover = cover * (1.0 - rut * 0.85);
    // A fresh track: the tread pressed the snow flat and grey. Its edge sharpened from the
    // texels' blend (a soft ramp over a whole texel read as a blur), and the pressed snow
    // grainy - the tread's blocks, as fine as the eye can still tell apart.
    let edge = smoothstep(0.2, 0.75, tracks.y);
    var track = 0.0;
    if (edge > 0.0) {
        track = edge * (0.75 + 0.25 * vnoise_f(m2, 14.0, vec2<f32>(1.7, 9.3)));
    }
    cover = cover * (1.0 - track * 0.6);
    var wet = max(slush, track * 0.5 * base);
    // Thawing (rain on it, or a wet road, while no fresh snow falls): what is left goes grey
    // and soaked, and it breaks up - the thin patches first, in the grain of a smooth noise
    // a few strides across - into slush over the wet asphalt.
    let thaw = clamp(snow_track.state.w, 0.0, 1.0);
    if (thaw > 0.0 && cover > 0.0) {
        let n = vnoise_f(m, 1.1, vec2<f32>(6.1, 2.3)) * 0.6 + vnoise_f(m2, 4.3, vec2<f32>(9.7, 4.1)) * 0.4;
        let bare = smoothstep(n - 0.06, n + 0.06, thaw * 0.6 - 0.08 + 0.3 * (1.0 - base));
        let left = cover * (1.0 - bare);
        wet = max(wet, (cover - left) * 0.7 + left * thaw * 0.8);
        cover = left * (1.0 - 0.55 * thaw);
    }
    return vec3<f32>(cover, wet, track);
}

// Enhanced graphics: what the enhanced main pass, the sky and the coronas share - the
// light of the physically based sky (see atmosphere.rs), in its units (1 = 10 000 lux),
// and the air between the camera and what it sees. The scene is drawn pre-exposed:
// radiance times `enh.exposure.x`, so that the half-float target keeps its precision by
// night as well as by day.
struct Enhanced {
    // x pre-exposure, y the level unlit things are drawn at, z a lit window's radiance
    // (pre-exposed), w how bright a light's corona is drawn
    exposure: vec4<f32>,
    // rgb irradiance of the sun after the clouds, w its angular radius (rad)
    sun: vec4<f32>,
    // irradiance from the sky and the ground: order-2 spherical harmonics, rgb
    sh: array<vec4<f32>, 9>,
    // rgb radiance of the ground, w the sky table's scale
    ground: vec4<f32>,
    // x weather fog extinction (1/m) at its base, y its height falloff (1/m), z the base
    // height (render-origin relative), w clear-air extinction (1/m) at the ground
    fog: vec4<f32>,
    // rgb fog radiance from the light around it (the sun's part is added by angle),
    // w the table scale the reflection probe was drawn with
    fog_color: vec4<f32>,
    // x wet roads, y snow cover, z rain, w cloud cover in front of the sun
    weather: vec4<f32>,
    // x mip levels of the probe, y illuminance of a light's core (maplight colour 1),
    // z cabin light illuminance, w how much of the sun the clouds let through
    lights: vec4<f32>,
    // rgb the sun disc's irradiance before the clouds, w frame time (s)
    sun_disc: vec4<f32>,
    // x which term the main pass shows alone (OMSI_DEBUG_ENHANCED, 0 = the picture),
    // y the puddles' F0, z the envmap photo's strength, w the tone curve's contrast
    debug: vec4<f32>,
    // xyz where the sky cube is drawn from, relative to the camera
    eye: vec4<f32>,
    // x how bright an LED panel's dots burn (0 = off), y whether the LED panels' `\S:n`
    // masks keep their mip chain (0: at full resolution, the dots stay visible when small), z 1
    // when lights glare and glow in the night air (the settings' `light_glow`)
    led: vec4<f32>,
    // xyz towards the moon, w its angular radius (rad)
    moon: vec4<f32>,
    // rgb the moon disc's irradiance before the clouds, w how much of the starry sky shows
    moon_disc: vec4<f32>,
    // rgb the sun's irradiance at 1400, 2100 and 2800 m (the cumulus layer) and 7000 m
    // (the high thin layer), through the atmosphere from up there
    cloud_sun: array<vec4<f32>, 4>,
    // rgb the moonlight on a surface facing the moon (after the clouds), w 1 while the
    // shadow maps are the moon's
    moon_light: vec4<f32>,
    // the condensation on the player's bus's panes (optical depths): windscreen, sides,
    // rear, and how far the defroster has cleared the windscreen (0..1; omsi-app
    // condensation.rs)
    condensation: vec4<f32>,
    // x the cumulus layer's base, y its top (m over the ground), z the altocumulus cover,
    // w the storm's activity
    layers: vec4<f32>,
    // the lightning of this frame: x the light it throws on a surface facing it (1 = 10 000
    // lux), yz towards it (horizontal unit vector), w its elevation (rad)
    flash: vec4<f32>,
    // xyz towards the flash (unit vector), w how bright its channel is
    flash2: vec4<f32>,
    // x a number the channel's shape is drawn from
    flash3: vec4<f32>,
    // the clouds' shadow field (cloud_field.rs): xy its middle in cloud space, z one over its
    // width, w 1 when there is one
    cloud_f1: vec4<f32>,
    // xy what takes a world point to cloud space, z the height of the plane it is taken on,
    // w the share of the sun the main pass has taken off already for the clouds over the camera
    cloud_f2: vec4<f32>,
    // the player's bus's pitch and roll: sin, cos of each (its panes' condensation sits on
    // the glass on a slope, `pane_local`)
    pane_tilt: vec4<f32>,
    // Slots of the weather overhaul, one owner each (zero until it fills them):
    // the volumetric clouds, the volumetric fog, the precipitation, the bus's panes, the
    // surfaces' PBR and the ray tracing, and two spare.
    clouds_ext: vec4<f32>,
    clouds_ext2: vec4<f32>,
    vol_fog: vec4<f32>,
    vol_fog2: vec4<f32>,
    precip_ext: vec4<f32>,
    glass_ext: vec4<f32>,
    // (pbr_ext: x the share of the sun's light that still comes as a beam - the rest the
    // clouds scatter, `sun_scattered` - y 1 when x is set)
    pbr_ext: vec4<f32>,
    rt_ext: vec4<f32>,
};
@group(0) @binding(11) var<uniform> enh: Enhanced;
@group(0) @binding(13) var s_lin: sampler;
@group(0) @binding(14) var t_sky_lut: texture_2d<f32>;

const PI: f32 = 3.14159265;

// Irradiance from the sky and the ground on a surface facing n.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let c = enh.sh;
    let e = c[0].rgb * 0.282095
        + c[1].rgb * (0.488603 * n.y) + c[2].rgb * (0.488603 * n.z) + c[3].rgb * (0.488603 * n.x)
        + c[4].rgb * (1.092548 * n.x * n.y) + c[5].rgb * (1.092548 * n.y * n.z)
        + c[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0)) + c[7].rgb * (1.092548 * n.x * n.z)
        + c[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// The lightning's light on a surface facing n: half of it from the cloud overhead and
// round it (a hemisphere), half along the way to the flash itself.
fn flash_irradiance(n: vec3<f32>) -> vec3<f32> {
    if (enh.flash.x <= 0.0) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(0.80, 0.90, 1.0) * enh.flash.x * (0.5 * (0.5 + 0.5 * n.z) + 0.5 * max(dot(n, enh.flash2.xyz), 0.0));
}

// Sky radiance towards d (unscaled: times enh.ground.w is the radiance). The table runs
// over the azimuth from the sun (0..π) and the elevation (denser at the horizon).
// The sky table's texture: the table (32 x 48) in its corner and the clouds' shadow field
// (192 x 192) beside it, one texel clear (cloud_field.rs `SKY_TEX_W`, `FIELD_X0`).
const SKY_TEX: vec2<f32> = vec2<f32>(225.0, 192.0);
const SKY_TABLE: vec2<f32> = vec2<f32>(32.0, 48.0);
const FIELD_ORIGIN: f32 = 33.0;
const FIELD_N: f32 = 192.0;

fn sky_table_raw(d: vec3<f32>) -> vec3<f32> {
    let s = camera.sun_dir.xyz;
    let ls = length(s.xy);
    let ld = length(d.xy);
    var az = 0.0;
    if (ls > 1e-4 && ld > 1e-4) {
        az = acos(clamp(dot(s.xy / ls, d.xy / ld), -1.0, 1.0));
    }
    let el = asin(clamp(d.z, -1.0, 1.0));
    let v = 0.5 + 0.5 * sign(el) * sqrt(min(abs(el) / 1.5707963, 1.0));
    // (kept a half texel inside the table: its neighbours in the texture are not the sky)
    let tx = clamp(vec2<f32>(az / PI, v) * SKY_TABLE, vec2<f32>(0.5), SKY_TABLE - vec2<f32>(0.5));
    return textureSampleLevel(t_sky_lut, s_lin, tx / SKY_TEX, 0.0).rgb;
}

// The share of the sun that reaches the point p through the clouds, as against what the main
// pass's sun (`enh.sun`) already has for the camera's own place: the cumulus layer's shadow
// field read where the sun's way through p meets the layer's middle. 1 where there is no
// field, beyond its edge and where the sun is low (the camera's own factor stands there).
fn cloud_shadow_ratio(p: vec3<f32>) -> f32 {
    let s = camera.sun_dir.xyz;
    if (enh.cloud_f1.w < 0.5 || s.z < 0.06) {
        return 1.0;
    }
    let cs = p.xy + s.xy / s.z * (enh.cloud_f2.z - p.z) + enh.cloud_f2.xy;
    let uv = (cs - enh.cloud_f1.xy) * enh.cloud_f1.z + vec2<f32>(0.5);
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
    let w = 1.0 - smoothstep(0.36, 0.48, edge);
    if (w <= 0.0) {
        return 1.0;
    }
    let tx = vec2<f32>(FIELD_ORIGIN, 0.0) + clamp(uv * FIELD_N, vec2<f32>(0.5), vec2<f32>(FIELD_N - 0.5));
    let f = textureSampleLevel(t_sky_lut, s_lin, tx / SKY_TEX, 0.0).r;
    let ratio = min(f / max(enh.cloud_f2.w, 0.02), 6.0);
    return mix(1.0, ratio, w);
}

fn sky_table(d: vec3<f32>) -> vec3<f32> {
    return sky_table_raw(d) * enh.ground.w;
}

// Henyey-Greenstein phase function.
fn hg_phase(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// Optical depth of an exponentially thinning layer (extinction sigma at height 0, falling
// off with k per metre) along a straight path of length dist from height h0 to h1.
fn layer_depth(sigma: f32, k: f32, h0: f32, h1: f32, dist: f32) -> f32 {
    if (sigma <= 0.0) {
        return 0.0;
    }
    let a = max(h0, 0.0);
    let b = max(h1, 0.0);
    let dh = b - a;
    if (abs(k * dh) < 1e-3) {
        return sigma * dist * exp(-k * 0.5 * (a + b));
    }
    return sigma * dist * (exp(-k * a) - exp(-k * b)) / (k * dh);
}

// The air between the camera and a point dist metres away in direction dir (from the
// camera), at heights h0 (camera) and h1 (point) above the fog's base: rgb the light it
// scatters towards the camera (radiance), a what it lets through of the point.
fn air(dir: vec3<f32>, dist: f32, h0: f32, h1: f32) -> vec4<f32> {
    return air_of(dir, dist, h0, h1, 1.0);
}

// The same with the clear air's share scaled (0 for the sky, whose table holds it already).
fn air_of(dir: vec3<f32>, dist: f32, h0: f32, h1: f32, clear: f32) -> vec4<f32> {
    let tau_fog = layer_depth(enh.fog.x, enh.fog.y, h0, h1, dist);
    let tau_air = layer_depth(enh.fog.w * clear, 1.0 / 1500.0, h0, h1, dist);
    let tau = tau_fog + tau_air;
    let t = exp(-tau);
    if (tau < 1e-5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // clear air takes on the sky's own colour at the horizon in this direction (which
    // already holds the glow round the sun); a weather fog glows with the light around it
    // and brighter towards the sun
    let horizon = sky_table(normalize(vec3<f32>(dir.x, dir.y, max(dir.z, 0.0) * 0.3 + 0.03)));
    let fog_in = enh.fog_color.rgb + enh.sun.rgb * hg_phase(dot(dir, camera.sun_dir.xyz), 0.55) * 0.9;
    let inscatter = mix(horizon, fog_in, tau_fog / tau);
    return vec4<f32>(inscatter * (1.0 - t), t);
}

// The water normal shared by shading and scene reflections. The integer hash and the
// 1000 m period match shader.wgsl, keeping the rings fixed through floating-origin moves.
fn puddle_hash(c: vec2<f32>) -> f32 {
    let cells = 8000.0;
    let w = c - cells * floor(c / cells);
    var v = vec2<u32>(w) * 1664525u + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return f32(v.x >> 8u) / 16777215.0;
}

// The rings the rain makes on standing water (Enhanced): xy the water's slope (a horizontal
// offset to its normal), z how much it is stirred (0..1, for its roughness as well).
//
// A falling drop sends out a short train of capillary waves: two or three rings close
// together, running out at a few decimetres a second and dying away within a hand's
// breadth. Three lattices of cells of 18 to 27 cm, turned against each other so that no grid
// shows, each take a drop now and then - the harder it rains the more often, and in more
// of them - every drop at a new place in its cell. A ring is a wave packet on its circle
// (a cosine under a Gaussian), its slope worked out exactly, and it stays inside its cell
// (no ring is cut off at a cell's edge).
fn puddle_ripple(pattern_xy: vec2<f32>, time: f32, rain: f32, coverage: f32) -> vec3<f32> {
    if (rain <= 0.001 || coverage <= 0.0) {
        return vec3<f32>(0.0);
    }
    var slope = vec2<f32>(0.0);
    var stir = 0.0;
    let rate = 0.7 + 1.6 * rain;
    let amp = min(1.0, 0.45 + rain) * coverage;
    for (var i = 0; i < 3; i = i + 1) {
        let fi = f32(i);
        let size = 0.18 + 0.045 * fi;
        let cs = vec2<f32>(cos(fi * 1.93), sin(fi * 1.93));
        let turned = vec2<f32>(pattern_xy.x * cs.x - pattern_xy.y * cs.y, pattern_xy.x * cs.y + pattern_xy.y * cs.x);
        let q = turned / size + vec2<f32>(fi * 17.0, fi * 5.0);
        let cell = floor(q);
        let seed = puddle_hash(cell + vec2<f32>(fi * 311.0, 0.0));
        let tt = time * rate * (0.8 + 0.4 * seed) + seed * 7.0;
        let n = floor(tt);
        let age = tt - n;
        let h = puddle_hash(cell + vec2<f32>(n - 97.0 * floor(n / 97.0) + 401.0, fi * 13.0 + 7.0));
        let hit = step(fract(h * 31.7), 0.3 + 0.65 * rain);
        let centre = vec2<f32>(0.3) + 0.4 * vec2<f32>(h, fract(h * 13.7));
        let local = fract(q) - centre;
        let r = length(local);
        let radius = 0.05 + 0.24 * sqrt(age);
        let d = r - radius;
        let w = 0.045;
        let k = 6.2831853 / 0.07;
        let env = exp(-(d * d) / (w * w));
        let a = (1.0 - age) * (1.0 - age) * hit;
        // the height is a * env * cos(k d): its slope along the radius
        let dh = a * env * (-2.0 * d / (w * w) * cos(k * d) - k * sin(k * d));
        slope = slope - local / max(r, 1e-4) * dh / k;
        stir = stir + a * env;
    }
    return vec3<f32>(slope * 0.03 * amp, clamp(stir * amp, 0.0, 1.0));
}

// The rings as a picture whose pixel covers `fp` metres of the water sees them: their
// slopes are a few centimetres long, and a pixel (or a puddle ray standing for several)
// wider than half of that lands on a ring's crest one frame and its trough the next - the
// water's picture trembled and broke into blocks. Faded out there, the rings leave their
// stir (the roughness) and the reflection keeps still.
fn puddle_ripple_seen(pattern_xy: vec2<f32>, time: f32, rain: f32, coverage: f32, fp: f32) -> vec3<f32> {
    let fade = 1.0 - smoothstep(0.012, 0.035, fp);
    let r = puddle_ripple(pattern_xy, time, rain, coverage);
    return vec3<f32>(r.xy * fade, r.z);
}

// The classic rings (Vanilla+'s puddle reflections): one lattice of 12.5 cm cells.
fn puddle_ripple_classic(pattern_xy: vec2<f32>, time: f32, rain: f32, coverage: f32) -> vec3<f32> {
    let cell = floor(pattern_xy * 8.0);
    let seed = puddle_hash(cell);
    let phase = fract(time * (0.8 + seed * 0.9) + seed * 13.0);
    let local = fract(pattern_xy * 8.0) - vec2<f32>(0.5);
    let ring = abs(length(local) - phase * 0.4);
    let hit = step(1.0 - clamp(0.25 + 0.6 * rain, 0.0, 0.9), fract(seed * 31.7));
    let strength = (1.0 - smoothstep(0.0, 0.06, ring)) * (1.0 - phase) * rain * coverage * hit;
    // Shallow rain rings disturb the image gently; steep normals break neighbouring
    // reflection rays apart and make otherwise still puddles look like rough waves.
    let bump = normalize(local + vec2<f32>(1e-5, 0.0)) * strength * 0.018;
    return vec3<f32>(bump, strength);
}

// `beam_shadow.wgsl` for the shaders without the lamp shadow tiles: nothing stands in a beam.
fn beam_tile(li: u32, l: PointLight) -> u32 {
    return 0xffffffffu;
}

fn beam_reaches(k: u32, x: vec3<f32>) -> f32 {
    return 1.0;
}

// Enhanced graphics: the lamps as the scene and the sky share them - a headlamp's beam
// and the light the weather's fog scatters from the lamps towards the camera (needs the
// point lights, their grid and `camera.light_grid`).

// A headlamp's intensity towards `t` (from the lamp) by the angles of a road lamp, not
// around its axis: wide across, brightest just under the lamp's horizon where it reaches
// far down the road, weak straight down, and with a low beam a sharp cut-off above it.
// `dir` is the lamp's axis in its vehicle's floor: its rise tilts the horizon with the
// vehicle, up a ramp and down (#1897).
fn headlamp(t: vec3<f32>, dir: vec3<f32>, low: bool) -> f32 {
    let fwd = normalize(dir.xy + vec2<f32>(1e-6, 0.0));
    let ahead = dot(t.xy, fwd);
    if (ahead <= 0.0) {
        return 0.0;
    }
    let across = abs(t.x * fwd.y - t.y * fwd.x) / ahead;
    let wide = 0.12 * smoothstep(1.0, 0.45, across) + 0.88 * exp(-across * across / 0.06);
    let rise = t.z / max(length(t.xy), 1e-3);
    let tilt = dir.z / max(length(dir.xy), 1e-3);
    // (the tangent of the angle under that horizon)
    let drop = (tilt - rise) / max(1.0 + rise * tilt, 1e-3);
    // full out to where the road is 0.06 under the lamp's horizon, then less as the cube of
    // the drop and a little more: the road is lit evenly from the bumper on, a little
    // brighter as far as the beam reaches, and not as one hot pool where its axis lands
    var up = min(1.0, pow(0.06 / max(abs(drop), 1e-4), 3.4));
    if (low) {
        up = up * smoothstep(-0.012, 0.025, drop);
        return wide * up;
    }
    // a full beam reaches far: a narrow, bright core along the lamp's horizon
    let hot = exp(-across * across / 0.012 - drop * drop / 0.0004);
    return wide * up + 6.0 * hot;
}

// The lamps' light the weather's fog scatters towards the camera (radiance, not
// pre-exposed): single scattering along the view ray from `start` to `len` metres in
// direction `d` - the glow round a street lamp in the mist, its cone under it, a bus's
// beams reaching ahead into the fog or the falling snow. The droplets scatter strongly
// forward (fog and cloud droplets of some ten micrometres: g about 0.8), so the glow is
// brightest looking towards a lamp and a cone shows faintly from the side.
//
// Each lamp is taken once, over the whole stretch of the ray within its reach, in the grid
// cell the ray's point nearest to it lies in (the ray's cells are walked through; every
// cell lists the lamps reaching into it). Along a straight ray past a point source, with
// t - t0 = h tan(phi) (h the ray's distance from the lamp, t0 where it passes nearest),
// the inverse square cancels against the path element: dt / r^2 = dphi / h, and what is
// left is the phase function over the angle, cos(theta) = -sin(phi) - a smooth integral
// taken with five Gauss-Legendre nodes, at each of which a spot's or a headlamp's beam is
// asked how much it sends that way. The fog's extinction dims the light on its way from
// the lamp and to the camera (taken at the nearest point). A beam's light, which reaches
// only part of the way, is summed along the ray instead, its steps staggered by `jitter`.
// The droplets' phase function: a strong forward peak and the back-scattered light a
// fog's lidar ratio of some 18 sr asks for (0.055 per steradian straight back) - a single
// Henyey-Greenstein lobe of g 0.8 gave a tenth of it, and a bus's own headlights lit no
// wall of fog in front of the driver. Two lobes, 0.9 of g 0.85 and 0.1 of g -0.5.
fn fog_phase(c: f32) -> f32 {
    return 0.9 * hg_phase(c, 0.85) + 0.1 * hg_phase(c, -0.5);
}
const AIRLIGHT_CELLS: u32 = 16u;

// The clear night air's own haze, which lights a lamp's halo and a headlamp's beam as the fog's
// droplets do, only fainter: the lowest hundred metres of a city's air hold its dust, exhaust
// and humidity at several times the column's average aerosol extinction (`enh.fog.w`), 1-3e-4
// per metre (a visibility of 15-40 km). A lamp then has a soft glow round it and a low beam
// a faint cone, as in any night photograph, instead of a lit disc on a black ground.
const GROUND_HAZE_GAIN: f32 = 8.0;

// `droplets`: how much of the weather's extinction scatters like fog (see `FogLampUniform`).
fn lamp_airlight(c: vec3<f32>, d: vec3<f32>, start: f32, len: f32, jitter: f32, droplets: f32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let sigma0 = enh.fog.x * droplets + enh.fog.w * GROUND_HAZE_GAIN * enh.led.z;
    let cell = camera.light_grid.z;
    let side = u32(camera.light_grid.w);
    if (sigma0 < 4e-5 || cell <= 0.0 || side == 0u || len <= start) {
        return sum;
    }
    // (past a few optical depths nothing comes back)
    let end = min(len, start + 4.0 / sigma0);
    let gx = array<f32, 5>(-0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798);
    let gw = array<f32, 5>(0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269);
    // the walk through the grid's cells along the ray's plan
    let o = (c.xy - camera.light_grid.xy) / cell;
    let dd = d.xy / cell;
    var ix = i32(floor(o.x + dd.x * start));
    var iy = i32(floor(o.y + dd.y * start));
    let sx = select(-1, 1, dd.x >= 0.0);
    let sy = select(-1, 1, dd.y >= 0.0);
    let inv_x = select(1e9, 1.0 / abs(dd.x), abs(dd.x) > 1e-7);
    let inv_y = select(1e9, 1.0 / abs(dd.y), abs(dd.y) > 1e-7);
    var tx = select((f32(ix) - o.x) / dd.x, (f32(ix + 1) - o.x) / dd.x, dd.x >= 0.0);
    var ty = select((f32(iy) - o.y) / dd.y, (f32(iy + 1) - o.y) / dd.y, dd.y >= 0.0);
    if (abs(dd.x) <= 1e-7) { tx = 1e9; }
    if (abs(dd.y) <= 1e-7) { ty = 1e9; }
    var t = start;
    for (var k = 0u; k < AIRLIGHT_CELLS; k = k + 1u) {
        if (t >= end) {
            break;
        }
        let t_out = min(min(tx, ty), end);
        if (ix >= 0 && iy >= 0 && ix < i32(side) && iy < i32(side)) {
            let base = (u32(iy) * side + u32(ix)) * CELL_CAP;
            for (var j = 0u; j < CELL_CAP; j = j + 1u) {
                let li = grid[base + j];
                if (li == 0xffffffffu) {
                    break;
                }
                let l = lights[li];
                let to_l = l.pos.xyz - c;
                let t0 = dot(to_l, d);
                // the ray's point nearest to the lamp: taken in this cell only
                let tc = clamp(t0, start, end);
                if (tc < t || tc > t_out) {
                    continue;
                }
                let range = l.extra.w;
                let h = length(to_l - d * t0);
                if (h >= range) {
                    continue;
                }
                // A beam (a headlamp, a spot) is lit in the weather's fog only: its cone is a sliver of
                // sixteen jittered steps at half size, which the fog's density hides and the clear
                // air's haze shows as shimmer and soap. The lamps' halo alone has the ground haze.
                let beam_lamp = l.extra.z != 0.0 || l.dir.w > -1.5;
                let fog_sigma = enh.fog.x * droplets;
                if (beam_lamp && enh.fog.x < 2e-4) {
                    continue;
                }
                let s0 = select(sigma0, fog_sigma, beam_lamp);
                var core = l.extra.y;
                if (core <= 0.0) {
                    core = range * 0.125;
                }
                // (inside its core a lamp is as bright as at the core's edge)
                let hh = max(h, select(core * 0.7, 0.4, l.extra.z != 0.0));
                let w = sqrt(range * range - h * h);
                let ta = max(start, t0 - w);
                let tb = min(end, t0 + w);
                if (tb <= ta) {
                    continue;
                }
                var acc = 0.0;
                var half = 0.0;
                if (l.extra.z != 0.0 || l.dir.w > -1.5) {
                    // a beam (a headlamp, a spot): lit only where its cone or cut-off lets
                    // it - a sliver of the angles the lamp is seen under, which the five
                    // nodes missed. Sixteen even steps over that angle instead (dphi =
                    // h dt / r^2: the inverse square is in the step). (Sixteen even steps
                    // along the way put the few next to the lamp, where nearly all of its
                    // light is, a metre or more apart: the pixels' staggered steps came
                    // out as a coarse checker in the fog round every headlamp.)
                    let pa = atan((ta - t0) / hh);
                    let dphi = (atan((tb - t0) / hh) - pa) / 16.0;
                    let tile_k = beam_tile(li, l);
                    for (var q = 0u; q < 16u; q = q + 1u) {
                        let tq = t0 + hh * tan(pa + (f32(q) + jitter) * dphi);
                        let to = l.pos.xyz - (c + d * tq);
                        let ld = to * inverseSqrt(max(dot(to, to), 1e-6));
                        var beam = 0.0;
                        if (l.extra.z != 0.0) {
                            beam = headlamp(-ld, l.dir.xyz, l.extra.z > 0.0);
                        } else {
                            beam = smoothstep(l.dir.w, l.extra.x, dot(-ld, l.dir.xyz));
                        }
                        // (the map is looked at only where the beam shines at all, and only for
                        // the few lamps that have a tile: no shadows, no tile, no tap)
                        var reach = 1.0;
                        if (beam > 0.002 && tile_k < 44u) {
                            reach = beam_reaches(tile_k, c + d * tq);
                        }
                        acc = acc + beam * reach * fog_phase(dot(ld, d)) * dphi;
                    }
                    half = 1.0;
                } else {
                    let pa = atan((ta - t0) / hh);
                    let pb = atan((tb - t0) / hh);
                    half = 0.5 * (pb - pa);
                    let mid = 0.5 * (pa + pb);
                    let tile_p = beam_tile(li, l);
                    for (var q = 0u; q < 5u; q = q + 1u) {
                        let phi = mid + half * gx[q];
                        let tq = t0 + hh * tan(phi);
                        let ld = normalize(l.pos.xyz - (c + d * tq));
                        var beam = 1.0;
                        if (l.dir.z < -0.5) {
                            // (a lamp in a housing: see `lamp_light`)
                            beam = 0.05 + 0.95 * smoothstep(-0.1, 0.3, ld.z);
                        }
                        // (a lamp with a shadow map: what stands between it and the air
                        // keeps its light out of that air, as the headlamps' beams)
                        var reach = 1.0;
                        if (beam > 0.002 && tile_p < 44u) {
                            reach = beam_reaches(tile_p, c + d * tq);
                        }
                        acc = acc + gw[q] * beam * reach * fog_phase(-sin(phi));
                    }
                }
                // a lamp's irradiance is core^2 / r^2 of its colour's (a headlamp's 1 / r^2)
                let strength = select(core * core, 1.0, l.extra.z != 0.0);
                let xc = c + d * tc;
                let sigma = s0 * exp(-enh.fog.y * max(xc.z - enh.fog.z, 0.0));
                let dl = distance(xc, l.pos.xyz);
                let atten = exp(-layer_depth(s0, enh.fog.y, c.z - enh.fog.z, xc.z - enh.fog.z, tc - start) - sigma * dl);
                sum = sum + l.color.rgb * (l.color.w * strength * acc * half / hh * sigma * atten);
            }
        }
        t = t_out;
        if (tx < ty) {
            tx = tx + inv_x;
            ix = ix + sx;
        } else {
            ty = ty + inv_y;
            iy = iy + sy;
        }
    }
    return sum * enh.lights.y;
}

// Enhanced: the light a raindrop or a snowflake sends towards the eye - what it scatters of
// the sky's light, the sun's and the lamps' round it, the drop's strongly forward (a
// streak flashes against a lamp seen through the rain, and is next to invisible in the
// dark between the lamps), the flake's nearly every way. (Drawn at the display's level
// whatever the light, the flakes of a snowfall at night shone white in the dark, far from
// any lamp.)
fn precip_light(x: vec3<f32>, to_eye: vec3<f32>, snow: bool) -> vec3<f32> {
    // the sky's and the ground's light round it, and the sun's on a small sphere
    var e = enh.fog_color.rgb * (PI / 0.9) + enh.sun.rgb * 0.25;
    let cell = camera.light_grid.z;
    let side = u32(camera.light_grid.w);
    if (cell > 0.0 && side > 0u) {
        let f = (x.xy - camera.light_grid.xy) / cell;
        let gx = i32(floor(f.x));
        let gy = i32(floor(f.y));
        if (gx >= 0 && gy >= 0 && gx < i32(side) && gy < i32(side)) {
            let base = (u32(gy) * side + u32(gx)) * CELL_CAP;
            for (var j = 0u; j < CELL_CAP; j = j + 1u) {
                let li = grid[base + j];
                if (li == 0xffffffffu) {
                    break;
                }
                let l = lights[li];
                let dl = l.pos.xyz - x;
                let dist2 = dot(dl, dl);
                let range = l.extra.w;
                if (dist2 >= range * range) {
                    continue;
                }
                let ld = dl * inverseSqrt(max(dist2, 1e-6));
                var core = l.extra.y;
                if (core <= 0.0) {
                    core = range * 0.125;
                }
                let q = dist2 / (range * range);
                let window = (1.0 - q * q) * (1.0 - q * q);
                var k = core * core / sqrt(dist2 * dist2 + core * core * core * core) * window;
                if (l.extra.z != 0.0) {
                    k = headlamp(-ld, l.dir.xyz, l.extra.z > 0.0) / max(dist2, 0.3) * window;
                } else if (l.dir.w > -1.5) {
                    k = k * smoothstep(l.dir.w, l.extra.x, dot(-ld, l.dir.xyz));
                } else if (l.dir.z < -0.5) {
                    k = k * (0.05 + 0.95 * smoothstep(-0.1, 0.3, ld.z));
                }
                // (the scattering towards the eye relative to an even one: a drop's light
                // goes on forwards, a flake's every way. The angle is the one between the
                // light's way from the lamp, -ld, and on to the eye: taken against -to_eye
                // it put the forward peak behind the drop, and the drops in a bus's own
                // beams, seen from behind the lamps, shone a hundred times too bright as
                // white streaks down the road ahead)
                let c = dot(-ld, to_eye);
                let lobe = select(fog_phase(c), 0.7 * hg_phase(c, 0.3) + 0.3 / (4.0 * PI), snow) * 4.0 * PI;
                e = e + l.color.rgb * l.color.w * enh.lights.y * k * lobe;
            }
        }
    }
    return e / PI;
}


// Enhanced graphics: the main pass's own lighting model, the usual metal/roughness one:
// GGX (Trowbridge-Reitz) microfacets with height-correlated Smith visibility and Schlick's
// Fresnel (F0 4 % for a dielectric, its colour for a metal), their multiple scattering given
// back (Fdez-Aguera), and Lambert diffuse with the share the specular part leaves it, so that
// nothing sends out more light than falls on it. Lit by the sun (soft shadows whose penumbra
// grows with the distance to the blocker; what the clouds scatter of it as a shadowless light
// of the sky's kind), the sky and the ground (spherical harmonics, occluded by the SSAO or
// the traced occlusion), a reflection probe of the sky for the reflections (the split sum
// with Karis' analytic DFG, occluded by the AO as Lagarde has it and by the horizon of a
// normal map), point and spot lights falling off with the square of the distance, and the
// air in between. Roughness is widened by the normal's spread across the pixel (specular
// antialiasing) and held off zero for the sun's disc. OMSI's material data
// decides the surface: [matl_envmap] factor x mask is how much it reflects and how smooth
// it is, the o3d specular power how rough a surface without one is, [matl_bumpmap] bends
// the normal. Everything is drawn pre-exposed into the high-range target.

// Only painted terrain may skip empty brush-mask pixels.
override TERRAIN_PAINT: bool = false;

@group(0) @binding(12) var t_probe: texture_cube<f32>;

fn d_ggx(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// The microfacets of a film of water over a road: the slopes of a water surface are
// Gaussian (Cox and Munk 1954), the Beckmann distribution. Its tail falls off as
// exp(-tan^2 / a^2), where GGX's falls as a^2 / tan^4: under GGX a headlamp - a hundred
// thousand candela towards the road - kept a highlight brighter than the lit road some ten
// degrees off its mirror direction, and its reflection ran as one white line from the lamp
// down to the camera.
fn d_beckmann(nh: f32, a: f32) -> f32 {
    let c2 = max(nh * nh, 1e-4);
    let a2 = a * a;
    return exp(-(1.0 - c2) / (c2 * a2)) / (PI * a2 * c2 * c2);
}

// Height-correlated Smith visibility (G / (4 n.l n.v)).
fn v_smith(nv: f32, nl: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

// A diffuse surface's albedo under a film of water (n = 1.33; Lekner and Dorf 1988): the
// film lets in 1 - 0.066 of diffuse light, and of what the surface scatters back up only
// 1 - 0.472 leaves through the film's top, the rest returns to the surface.
fn wet_albedo(a: vec3<f32>) -> vec3<f32> {
    return 0.934 * a * 0.528 / (vec3<f32>(1.0) - 0.472 * a);
}

fn f_schlick(f0: vec3<f32>, c: f32) -> vec3<f32> {
    let f = pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * f;
}

// The split-sum's integral of the GGX lobe over the hemisphere for a white and a black F0
// (Karis' analytic fit of the DFG table): the reflectance is f0 * x + y.
fn dfg_ab(rough: f32, nv: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = c0 * rough + c1;
    let a004 = min(r.x * r.x, pow(2.0, -9.28 * nv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// The split-sum environment term, single scattering only.
fn env_brdf(f0: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let ab = dfg_ab(rough, nv);
    return f0 * ab.x + vec3<f32>(ab.y);
}

// What a surface reflects specularly towards the eye of light from every direction, with the
// light its microfacets pass on to each other before it leaves (Fdez-Aguera 2019, after
// Kulla and Conty): rgb that reflectance (the probe's weight; one minus it is what is left
// for the diffuse part, so the two never add up to more than came in), w the factor by which
// the single-scattering lobe of a direct light falls short of it. Without the multiple
// scattering a rough metal lost up to half its light and went dark grey.
fn spec_ms(f0: vec3<f32>, rough: f32, nv: f32) -> vec4<f32> {
    let ab = dfg_ab(rough, nv);
    let ess = clamp(ab.x + ab.y, 0.05, 1.0);
    let fss = f0 * ab.x + vec3<f32>(ab.y);
    let ems = 1.0 - ess;
    let favg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let fms = fss * favg / (vec3<f32>(1.0) - ems * favg);
    let comp = 1.0 + dot(f0, vec3<f32>(0.2126, 0.7152, 0.0722)) * (1.0 / ess - 1.0);
    return vec4<f32>(min(fss + fms * ems, vec3<f32>(1.0)), comp);
}

// Specular antialiasing (Tokuyoshi and Kaplanyan 2019): the spread of the normal across the
// pixel widens the lobe, so that a mirror-smooth finish on a curved, faceted or normal-mapped
// surface does not sparkle and crawl where it is smaller than a pixel. Returns the
// perceptual roughness.
fn rough_aa(rough: f32, n: vec3<f32>) -> f32 {
    let dx = dpdx(n);
    let dy = dpdy(n);
    let variance = 0.25 * (dot(dx, dx) + dot(dy, dy));
    let kernel = min(2.0 * variance, 0.18);
    let a = rough * rough;
    return sqrt(sqrt(clamp(a * a + kernel, 0.0, 1.0)));
}

// Horizon occlusion (Jimenez et al. 2016): a normal map can turn a reflection below the
// surface the geometry really has; that part of the lobe sees the surface itself, not the sky.
fn horizon_occlusion(r: vec3<f32>, geo_n: vec3<f32>) -> f32 {
    let h = clamp(1.0 + 1.3 * dot(r, geo_n), 0.0, 1.0);
    return h * h;
}

// The sun's light through the clouds that is no longer a beam (`enh.pbr_ext.x` the beam's
// share, lighting/enhanced.rs `sun_directness`): what a veil or a thin deck scatters round the
// sun lies on the street as a bright dome overhead with a glow towards the sun, casting no
// shadow. It is the flux the beam would have brought to the ground (its share on a horizontal
// surface), spread so: a cloudy day's light is flat, not a weaker sun with sharp shadows.
fn sun_scattered(n: vec3<f32>, e: vec3<f32>) -> vec3<f32> {
    let s = camera.sun_dir.xyz;
    let dome = 0.5 + 0.5 * n.z;
    let lobe = (1.0 + dot(n, s)) / (1.0 + max(s.z, 0.0));
    return e * max(s.z, 0.0) * mix(dome, lobe, 0.4);
}

// The beam's share of the sun's light (1 under a clear sky; 0 while the pass has not set it).
fn sun_beam_share() -> f32 {
    return select(1.0, clamp(enh.pbr_ext.x, 0.0, 1.0), enh.pbr_ext.y > 0.5);
}

// Ambient occlusion that lets bright surfaces bounce some light back into their own
// corners (Jimenez' multi-bounce fit), so white walls do not get grey creases.
fn ao_bounce(ao: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - vec3<f32>(0.3324);
    let b = -4.7951 * albedo + vec3<f32>(0.6417);
    let c = 2.7552 * albedo + vec3<f32>(0.6903);
    return max(vec3<f32>(ao), ((a * ao + b) * ao + c) * ao);
}

// World direction to the probe's cube coordinates (and back: the swap is its own inverse).
fn to_cube(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(d.x, d.z, d.y);
}

// What a raindrop on a pane shows (`glass_film`): the sky probe that way, blurred a little
// by the drop's small lens, and low down the street's light in place of the probe's
// horizon (see the reflections in `shade_enhanced`).
fn rain_env_enhanced(d: vec3<f32>, lod: f32) -> vec3<f32> {
    let e = textureSampleLevel(t_probe, s_lin, to_cube(d), lod).rgb * enh.fog_color.w;
    let surround = enh.fog_color.rgb * (0.25 / 0.9);
    return mix(surround, e, smoothstep(-0.05, 0.35, d.z));
}

// A fixed, deterministic PCF kernel (shader.wgsl's `shadow_tent`). The old PCSS blocker
// search was unstable for alpha-tested foliage: a few leaves entering or leaving its
// 12-sample search changed the penumbra radius, producing checkerboard patches and
// camera-driven strips. The filter is fixed in shadow-map space, so the only thing that
// can change is the actual caster. `thin`: foliage, whose normals OMSI points up for even
// lighting - no receiver plane can be taken from them, a leaf compares at its own depth.
fn sun_shadow_soft(world_in: vec3<f32>, n: vec3<f32>, thin: bool) -> f32 {
    if (camera.shadow.x < 0.5) {
        return 1.0;
    }
    // (half a metre towards the sun for a leaf, as in `sun_shadow`)
    let world = world_in + select(vec3<f32>(0.0), camera.sun_dir.xyz * 0.5, thin);
    let ndl = clamp(dot(n, camera.sun_dir.xyz), 0.0, 1.0);
    // (a veil of high cloud spreads the sun into an aureole a few degrees wide: the light
    // comes from a larger source, and the shadow's edge widens with it)
    // (the tent's four gathers step apart for it: see `shader.wgsl`'s `shadow_tent_near`)
    let widen = 1.0 + 3.0 * enh.cloud_sun[1].w;
    let close = shadow_close(world, n, ndl, thin);
    if (close.y >= 0.999) {
        return close.x;
    }
    let near_lp = camera.light_view_proj * vec4<f32>(world + shadow_push_near(n, ndl), 1.0);
    let near_uv = vec2<f32>(near_lp.x * 0.5 + 0.5, 0.5 - near_lp.y * 0.5);
    let near_edge = max(abs(near_uv.x - 0.5), abs(near_uv.y - 0.5));
    let near_valid = near_edge < 0.48 && near_lp.z >= 0.0 && near_lp.z <= 1.0;
    var near_slope = shadow_receiver_slope(camera.light_view_proj, n);
    var far_slope = shadow_receiver_slope(camera.light_view_proj_far, n);
    if (thin) {
        near_slope = vec2<f32>(0.0);
        far_slope = vec2<f32>(0.0);
    }
    // Blend cascades by radial distance, not by the near map's projected square.  The
    // projected-edge test made the blend factor change when the camera moved sideways or
    // turned, even for a stationary receiver; that reads as a full shadow shimmer at the
    // cascade boundary.  The light boxes still validate coverage above, while this gives
    // the overlap a camera-centred, continuous transition.
    let radial = distance(world, camera.cam_pos.xyz);
    let near_weight = select(
        0.0,
        1.0 - smoothstep(camera.shadow.z * 0.62, camera.shadow.z * 0.92, radial),
        near_valid,
    );
    // Each cascade is filtered only where it counts: sampling both everywhere (32 compares
    // a pixel) halved the enhanced picture's frame rate against shadows off.
    var near_value = 1.0;
    if (near_weight > 0.001) {
        near_value = shadow_pcf_near(near_uv, near_lp.z, near_slope, widen);
    }
    near_value = mix(near_value, close.x, close.y);
    if (near_weight >= 0.999) {
        return near_value;
    }
    let flp = camera.light_view_proj_far * vec4<f32>(world + shadow_push_far(n, ndl), 1.0);
    let fuv = vec2<f32>(flp.x * 0.5 + 0.5, 0.5 - flp.y * 0.5);
    let far_valid = fuv.x >= 0.0 && fuv.x <= 1.0 && fuv.y >= 0.0 && fuv.y <= 1.0 && flp.z >= 0.0 && flp.z <= 1.0;
    var far_safe = 1.0;
    if (far_valid) {
        let far_value = shadow_pcf_far(fuv, flp.z, far_slope, widen);
        let far_edge = max(abs(fuv.x - 0.5), abs(fuv.y - 0.5));
        far_safe = mix(1.0, far_value, clamp((0.5 - far_edge) * 12.0, 0.0, 1.0));
    }
    return mix(far_safe, near_value, near_weight);
}

// Enhanced+: the solid casters' shadow where no traced one is there for a surface. The close
// and near maps hold only the cut-out casters then (`shadow_plan`), and such a surface - a
// pane, the edge of a step or a handrail thinner than a pixel of the depth prepass - lay in
// the sun under a roof: bright specks and stair-stepped lines along the edges, flickering
// as the camera moved and the traced texel was found or not. The far map keeps them all.
fn sun_shadow_far(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (camera.shadow.x < 0.5) {
        return 1.0;
    }
    let ndl = clamp(dot(n, camera.sun_dir.xyz), 0.0, 1.0);
    let flp = camera.light_view_proj_far * vec4<f32>(world + shadow_push_far(n, ndl), 1.0);
    let fuv = vec2<f32>(flp.x * 0.5 + 0.5, 0.5 - flp.y * 0.5);
    if (fuv.x < 0.0 || fuv.x > 1.0 || fuv.y < 0.0 || fuv.y > 1.0 || flp.z < 0.0 || flp.z > 1.0) {
        return 1.0;
    }
    return shadow_pcf_far(fuv, flp.z, shadow_receiver_slope(camera.light_view_proj_far, n), 1.0);
}

// The sun's shadow at a point by one compare in the finest cascade that holds it (the close
// one, else the near one; lit beyond): for glass (see fs_enhanced).
fn sun_shadow_hard(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (camera.shadow.x < 0.5) {
        return 1.0;
    }
    let ndl = clamp(dot(n, camera.sun_dir.xyz), 0.0, 1.0);
    let range = camera.flags.w;
    if (range > 0.0 && distance(world, camera.cam_pos.xyz) < range * 0.75) {
        let lp = camera.light_view_proj_close * vec4<f32>(world + shadow_push_close(n, ndl), 1.0);
        let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
        if (max(abs(uv.x - 0.5), abs(uv.y - 0.5)) < 0.48 && lp.z >= 0.0 && lp.z <= 1.0) {
            let scale = select(1.0, camera.post.w, camera.post.w > 0.0);
            let a = vec2<f32>(uv.x * 0.5 * scale + 0.5, uv.y * scale);
            return textureSampleCompareLevel(t_shadow, s_shadow, a, lp.z - SHADOW_BIAS_CLOSE / SHADOW_DEPTH_RANGE);
        }
    }
    let lp = camera.light_view_proj * vec4<f32>(world + shadow_push_near(n, ndl), 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (max(abs(uv.x - 0.5), abs(uv.y - 0.5)) < 0.48 && lp.z >= 0.0 && lp.z <= 1.0) {
        return textureSampleCompareLevel(t_shadow, s_shadow, vec2<f32>(uv.x * 0.5, uv.y), lp.z - SHADOW_BIAS_NEAR / SHADOW_DEPTH_RANGE);
    }
    return 1.0;
}

// What a surface reflects and scatters: diffuse albedo, specular colour at normal
// incidence, perceptual roughness.
struct Surface {
    albedo: vec3<f32>,
    f0: vec3<f32>,
    rough: f32,
    // what the specular reflection leaves the diffuse part (`spec_ms`), and the factor
    // that gives the specular lobe back the light of its multiple scattering
    kd: vec3<f32>,
    comp: f32,
};

// How much of lamp `li` reaches the point (1 for a lamp without a shadow map): its map is
// one of the tiles under the far shadow map (lib.rs `LampShadow`), which the light's record
// names (`light_marks`): looking down from a street lamp's head, or along the road from a
// vehicle's headlamps; a few taps of it soften the edge, as the lamp's glowing bowl does.
// (the player's headlamps - marked so - share one map from between them along the road,
// `passes::headlamp_shadow`; it is read once for the pair, `own_shadow`)
var<private> own_shadow: f32;
fn lamp_shadow_at(li: u32, p: vec3<f32>, n: vec3<f32>, thin: bool) -> f32 {
    let marks = light_marks(lights[li]);
    let own = marks.x > 0.5;
    if (marks.z < 0.5) {
        return 1.0;
    }
    var k = u32(marks.z - 0.5);
    if (own && own_shadow >= 0.0) {
        return own_shadow;
    }
    let to_lamp = p - lights[li].pos.xyz;
    // a lamp that shines every way has a block of six tiles, one a direction (+x -x +y -y +z -z,
    // lib.rs `CUBE_SLOTS`): the one the lit point lies in
    if (k >= 44u) {
        let a = abs(to_lamp);
        var face = select(5u, 4u, to_lamp.z > 0.0);
        if (a.x >= a.y && a.x >= a.z) {
            face = select(1u, 0u, to_lamp.x > 0.0);
        } else if (a.y >= a.z) {
            face = select(3u, 2u, to_lamp.y > 0.0);
        }
        k = k + face;
    }
    let d = length(to_lamp);
    // (off the surface by a texel or so of the map there; a leaf card towards the lamp)
    let off = select(n * (0.03 + 0.012 * d), normalize(lights[li].pos.xyz - p) * 0.3, thin);
    let q = camera.lamp_view_proj[k] * vec4<f32>(p + off, 1.0);
    if (q.w <= 0.0) {
        return 1.0;
    }
    let ndc = q.xyz / q.w;
    if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0 || ndc.z >= 1.0 || ndc.z <= 0.0) {
        return 1.0;
    }

    let tuv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // (the far map's texture: its width, its height with the tiles under it; the tile's
    // texels in the atlas's terms, kept a texel off its edges)
    let dims = vec2<f32>(1.0 / camera.shadow.y, camera.lamp_atlas.z);
    let tile = camera.lamp_atlas.x;
    let cols = camera.lamp_atlas.y;
    let tx = clamp(tuv * tile, vec2<f32>(1.5), vec2<f32>(tile - 1.5));
    let base = vec2<f32>(f32(k % u32(cols)) * tile, dims.x + f32(k / u32(cols)) * tile) + tx;
    // (nine taps; a saloon's lamp - tiles 28 on - four: its tile's texels are three centimetres
    // and every pixel of a saloon takes a handful of lamps)
    // (two loops of fixed bounds: one bounded by the tile's kind was not unrolled)
    let saloon = k >= 28u && k < 44u;
    let zr = ndc.z - 0.00002;
    var sum = 0.0;
    var taps = 9.0;
    if (saloon) {
        for (var y = 0; y <= 1; y = y + 1) {
            for (var x = 0; x <= 1; x = x + 1) {
                let a = (base + (vec2<f32>(f32(x), f32(y)) - vec2<f32>(0.5)) * 1.25) / dims;
                sum = sum + textureSampleCompareLevel(t_shadow_far, s_shadow, a, zr);
            }
        }
        taps = 4.0;
    } else {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let a = (base + vec2<f32>(f32(x), f32(y)) * 1.25) / dims;
                sum = sum + textureSampleCompareLevel(t_shadow_far, s_shadow, a, zr);
            }
        }
    }
    let lit = sum / taps;
    if (own) {
        own_shadow = lit;
    }
    return lit;
}

const CABIN_FILL: f32 = 0.2;

// How much of the player's saloon lamps' light reaches a point past what stands in it, 1 where
// nothing does: each lamp that has a shadow map (`camera.cabin_lamps`: the first sixteen, by
// their records in the light buffer, -1 for none) weighs by what it would give the point
// unshadowed. The saloon's meshes are lit by the sets of lamps their models name and by light
// maps made for the lamps; this is what lets a handrail or a seat shade what it stands over
// in both.
fn cabin_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var open_light = 0.0;
    var seen_light = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let at = camera.cabin_lamps[i / 4u][i % 4u];
        if (at < 0.0) {
            break;
        }
        let li = u32(at);
        let l = lights[li];
        let d = l.pos.xyz - p;
        let dist2 = max(dot(d, d), 1e-4);
        let ndl = dot(n, d * inverseSqrt(dist2));
        if (ndl <= 0.0 || l.color.w <= 0.0) {
            continue;
        }
        let core = max(l.extra.y, 0.01);
        let share = (l.color.r + l.color.g + l.color.b) * l.color.w * core * core / dist2 * ndl;
        open_light = open_light + share;
        seen_light = seen_light + share * lamp_shadow_at(li, p, n, false);
    }
    // (the light a closed room's lining sends back where the lamps do not reach: a saloon's
    // walls, ceiling and seats, a third reflecting, lit by the lamps again and again - about
    // a fifth of what the lit parts have)
    let direct = clamp(seen_light / max(open_light, 1e-6), 0.0, 1.0);
    return select(1.0, mix(CABIN_FILL, 1.0, direct), open_light > 1e-4);
}

// A light cookie layer's texels a side (lib.rs `cookies::COOKIE_SIZE`).
const COOKIE_SIZE: u32 = 32u;

// The enhanced path's marks on a light (lib.rs `prepare_lights`, carried in `pos.w` below
// zero, in steps of 32 for its tile and the low bits below): x 1 for one of the player's
// headlamps, y 1 + the layer of its beam's picture (0: none), z 1 + its shadow map's tile
// (0: none, see `LAMP_MARK`).
fn light_marks(l: PointLight) -> vec3<f32> {
    let m = max(-l.pos.w, 0.0);
    let tile = floor(m / 32.0);
    let rest = m - tile * 32.0;
    let own = select(0.0, 1.0, rest > 15.5);
    return vec3<f32>(own, rest - own * 16.0, tile);
}

// A light cookie's texel (x, y) of the layer whose records start at `base`.
fn cookie_texel(base: u32, x: u32, y: u32) -> vec3<f32> {
    let i = min(y, COOKIE_SIZE - 1u) * COOKIE_SIZE + min(x, COOKIE_SIZE - 1u);
    let rec = lights[base + i / 4u];
    let k = i % 4u;
    return select(select(rec.pos.xyz, rec.color.xyz, k == 1u), select(rec.dir.xyz, rec.extra.xyz, k == 3u), k >= 2u);
}

// What a lamp with a picture of its beam (`[spotlight_cookie]`) sends towards `t` (unit,
// from the lamp): the picture's light there, as the beam would show on a wall square to
// the lamp - the picture's right towards the lamp's right as seen from behind it, its top
// up - and nothing outside the field it covers (`camera.cookies`: x, y its half angles'
// tangents, z where its layer starts in the light buffer, four linear texels a record),
// filtered between its four nearest texels.
fn cookie_light(t: vec3<f32>, l: PointLight) -> vec3<f32> {
    let f = l.dir.xyz;
    let ahead = dot(t, f);
    let rc = cross(f, vec3<f32>(0.0, 0.0, 1.0));
    let r = select(vec3<f32>(1.0, 0.0, 0.0), normalize(rc), dot(rc, rc) > 1e-8);
    let layer = camera.cookies[min(u32(light_marks(l).y - 0.5), 7u)];
    let q = vec2<f32>(dot(t, r), dot(t, cross(r, f))) / (max(ahead, 1e-3) * max(layer.xy, vec2<f32>(1e-3)));
    let st = clamp(vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5) * f32(COOKIE_SIZE) - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(f32(COOKIE_SIZE) - 1.0));
    let c0 = vec2<u32>(st);
    let w = st - floor(st);
    let base = u32(layer.z);
    let top = mix(cookie_texel(base, c0.x, c0.y), cookie_texel(base, c0.x + 1u, c0.y), w.x);
    let bottom = mix(cookie_texel(base, c0.x, c0.y + 1u), cookie_texel(base, c0.x + 1u, c0.y + 1u), w.x);
    return select(vec3<f32>(0.0), mix(top, bottom, w.y), ahead > 1e-3 && all(abs(q) < vec2<f32>(1.0)));
}

// The radiance of the street round the point as the lamps light it (`lamp_light` leaves it
// here): what a reflection pointing at the street and the walls beside it sees at night.
var<private> lamp_ground: vec3<f32>;

// The point and spot lights of the pixel's grid cell: diffuse and specular.
// `thin`: foliage, lit from whichever side the lamp is on (see the sun below).
// `film`: a wet road's film of water - x how much of the surface it is (0: none), y how
// rough it is, z the share of it that mirrors - with which the lamps are reflected as the
// sky is (`refl_rough`, `wet_share` in fs_enhanced), not by the dry surface's own lobe.
fn lamp_light(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, sf: Surface, thin: bool, shadows: bool, film: vec3<f32>, back: bool) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    lamp_ground = vec3<f32>(0.0);
    own_shadow = -1.0;
    // the lamps' light on the ground round the point (a horizontal surface's, unshadowed:
    // the ground the point looks down at is wider than its own shadow)
    var ground_e = vec3<f32>(0.0);

    let cell = camera.light_grid.z;
    let side = u32(camera.light_grid.w);
    if (cell <= 0.0 || side == 0u) {
        return sum;
    }
    let f = (p.xy - camera.light_grid.xy) / cell;
    let x = i32(floor(f.x));
    let y = i32(floor(f.y));
    if (x < 0 || y < 0 || x >= i32(side) || y >= i32(side)) {
        return sum;
    }
    let a = max(sf.rough * sf.rough, 0.02);
    let a_film = max(film.y * film.y, 0.02);
    let nv = max(dot(n, v), 1e-4);
    let facing = select(1.0, 0.0, back);
    let base = (u32(y) * side + u32(x)) * CELL_CAP;
    for (var j = 0u; j < CELL_CAP; j = j + 1u) {
        let li = grid[base + j];
        if (li == 0xffffffffu) {
            break;
        }
        let l = lights[li];
        let d = l.pos.xyz - p;
        let dist2 = dot(d, d);
        // (pos.w is 0 on a spot light, which the vanilla shader is to pass by)
        let range = l.extra.w;
        if (dist2 >= range * range) {
            continue;
        }
        let dist = sqrt(dist2);
        let ld = d / max(dist, 1e-3);
        // inverse-square beyond the core with a soft knee at it, not flat within it; windowed to zero at
        // the range so the grid cut-off does not show
        var core = l.extra.y;
        if (core <= 0.0) {
            core = range * 0.125;
        }
        let q = dist2 / (range * range);
        let window = (1.0 - q * q) * (1.0 - q * q);
        var e = core * core / sqrt(dist2 * dist2 + core * core * core * core) * window;
        // (a lamp's picture of its beam: its colour that way, see `cookie_light`)
        var tint = vec3<f32>(1.0);
        if (l.extra.z != 0.0) {
            e = headlamp(-ld, l.dir.xyz, l.extra.z > 0.0) / max(dist2, 0.3) * window;
        } else if (l.dir.w > -1.5) {
            let cd = dot(-ld, l.dir.xyz);
            if (light_marks(l).y > 0.5) {
                tint = cookie_light(-ld, l);
                e = e * step(1e-4, max(tint.r, max(tint.g, tint.b)));
            } else {
                e = e * smoothstep(l.dir.w, l.extra.x, cd);
            }
        } else if (l.dir.z < -0.5) {
            // a lamp in a housing (a street lamp's head, a platform's light): its reflector
            // sends the light down and out, its housing keeps it from the sky - a few per
            // cent above its horizon, what the bowl and the pole scatter (the upward light
            // ratio of road lighting, EN 13201 / CIE 115). Shining evenly every way, every
            // lamp lit the crowns of the trees and the upper floors over it as brightly
            // as the street.
            e = e * (0.05 + 0.95 * smoothstep(-0.1, 0.3, ld.z));
        }
        if (e <= 0.0 || (!(l.extra.z == 0.0 && l.dir.w < -1.5 && l.dir.z < -0.5) && own_vehicle_blocks(p, l.pos.xyz))) {
            continue;
        }
        var irr = l.color.rgb * tint * l.color.w * enh.lights.y * e;
        let nl = dot(n, ld);
        ground_e = ground_e + irr * max(ld.z, 0.0);
        // Ground bounce is unshadowed. A lamp behind an ordinary surface contributes
        // no direct light, so it must not sample its shadow map before being rejected.
        if (!thin && nl <= 0.0) {
            continue;
        }
        // the lamps' own shadow maps (the few lighting the view most, see `lamp_shadow_at`)
        if (shadows) {
            irr = irr * lamp_shadow_at(li, p, n, thin);
        }
        if (thin) {
            // a headlamp skims the grass: it lights the tips, not a crown's every side
            let wrap = select(0.45 + 0.25 * nl, 0.15 + 0.6 * max(nl, 0.0), l.extra.z != 0.0);
            sum = sum + irr * wrap * sf.albedo / PI;
            continue;
        }
        let h = normalize(ld + v);
        let nh = max(dot(n, h), 0.0);
        let fr = f_schlick(sf.f0, dot(v, h));
        var spec = d_ggx(nh, a) * v_smith(nv, nl, a) * fr;
        if (film.x > 0.0) {
            spec = mix(spec, d_beckmann(nh, a_film) * v_smith(nv, nl, a_film) * fr * film.z, film.x);
        }
        spec = spec * facing * sf.comp;
        sum = sum + irr * nl * (sf.albedo / PI * sf.kd + spec);
    }
    // What the lit ground throws back up: a diffuse reflector of the street's albedo (asphalt
    // and paving about 0.18, under water darker, fresh snow 0.75), seen by a surface over the
    // lower half of its view, (1 - n.z) / 2 of it - a wall half, the underside of a shelter's
    // roof all, the ground itself nothing. Without it the lamps lit only what faces them, and
    // a facade or a bus's flank beside a lit street stood black.
    let snow = clamp(enh.weather.y, 0.0, 1.0);
    let rho = mix(0.18, 0.75, snow) * mix(1.0, 0.6, clamp(enh.weather.x, 0.0, 1.0) * (1.0 - snow));
    let seen = select((1.0 - clamp(n.z, -1.0, 1.0)) * 0.5, 0.5, thin);
    sum = sum + ground_e * rho * seen * sf.albedo / PI;
    lamp_ground = ground_e * rho / PI;
    return sum;
}

// The relief a rough surface has under its microfacets - asphalt's grit and chippings, a
// paving stone's dressing, a cobble's dome (slopes of a quarter and more, at millimetres to
// centimetres) - shadows and masks itself (Smith's function for that slope spread,
// height-correlated) towards a light and an eye low over it. GGX's own roughness is the
// grain under that; on its own it let a low sun's grazing glint off every rough road and
// wall shine white towards the camera, where a real road keeps it dull. The spread grows
// with the rougher surface, and with the patch the pixel covers: farther off, the road's
// waves and ruts are under the pixel too. A smooth surface is left as it is.
fn relief_masking(nl: f32, nv: f32, rough: f32, footprint: f32) -> f32 {
    let s2 = 0.0625 * smoothstep(0.45, 0.85, rough) + 0.02 * smoothstep(0.01, 0.12, footprint) * smoothstep(0.45, 0.7, rough);
    if (s2 <= 0.0) {
        return 1.0;
    }
    // (Smith's Lambda for a slope variance s2: (sqrt(1 + s2 tan^2) - 1) / 2)
    let ml = clamp(nl, 1e-3, 1.0);
    let mv = clamp(nv, 1e-3, 1.0);
    let ll = 0.5 * (sqrt(1.0 + s2 * (1.0 - ml * ml) / (ml * ml)) - 1.0);
    let lv = 0.5 * (sqrt(1.0 + s2 * (1.0 - mv * mv) / (mv * mv)) - 1.0);
    return 1.0 / (1.0 + ll + lv);
}

// The light inside a cab relative to the average light outside: a bus's big windows let in
// most of the sky, and the dashboard lies right under the windscreen.
const CAB_AMBIENT: f32 = 1.15;

// How far the puddle threshold drops with the wetness: see the puddle mask in `shade_enhanced`.
const PUDDLE_SPREAD: f32 = 0.37;

/// The mip level a pixel's footprint asks for, in levels of the texture whose size is
/// `texels` (the usual `log2` of the larger derivative, held at 0 and up). An LED panel is
/// sampled with this, held at `enh.led.y` (`Lighting::led_mips`): 0 point-samples it, which
/// is the sharpest and shimmers worst - a regular dot grid is the worst case for a point
/// sample - and every level the chain is allowed to take is a 2x2 average that a point
/// sample of the level below does not have. (The derivatives have to be taken in uniform
/// control flow: the panels are per draw, so a call inside the material's branch is not.)
fn led_lod(uv: vec2<f32>, texels: vec2<f32>) -> f32 {
    let dx = dpdx(uv) * texels;
    let dy = dpdy(uv) * texels;
    return max(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0);
}

// A tangent-space normal `tn` (a PBR normal map's) turned into the world about the
// surface normal `n`, with the tangent frame taken from how the position and the uv change
// across the pixel (no tangents in OMSI's meshes). `tn.y` down the texture, as Direct3D's
// normal maps have it.
fn perturb_normal(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, tn: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let m = max(dot(t, t), dot(b, b));
    if (m < 1e-20) {
        return n;
    }
    let k = inverseSqrt(m);
    return safe_normal(t * k * tn.x + b * k * tn.y + n * max(tn.z, 0.05));
}

// The enhanced pass's two targets: the picture, and the screen mask (r: 1 on the bus's own
// screens, carried by the coverage of what is drawn over them; g: 1 on an LED panel's own
// dots, 0.25 on other screens, see MASK_FORMAT; b is wet puddles' reflected-light weight).
struct EnhancedOut {
    @location(0) color: vec4<f32>,
    @location(1) mask: vec4<f32>,
    //RT @location(2) gbuf: vec4<f32>,
    //RT @location(3) aux: vec4<f32>,
};

// Enhanced+ (the window's picture, camera.clouds.w 2): the surface the traced reflections
// start from - its normal and how much of the reflection lands on the screen (w), its
// distance from the eye and its roughness, all times that weight (see GBUF_FORMAT). The
// probe's reflection is left out of the colour there: the traced one is added in its place.
var<private> rt_gbuf: vec4<f32>;
var<private> rt_aux: vec4<f32>;

fn enhanced_surface(in: FsIn) -> EnhancedOut {
    rt_gbuf = vec4<f32>(0.0);
    rt_aux = vec4<f32>(0.0);
    var puddle_weight = vec2<f32>(0.0);
    let c = shade_enhanced(in, &puddle_weight, false, camera.cam_pos.xyz);
    let screen = material.flags.x > 0.5;
    // an LED panel's dots stay in the glow's source (`post.wgsl`), the other screens'
    // letters stay out of it
    let led = select(0.0, 1.0, material.emissive.w < -1.5);
    var out: EnhancedOut;
    out.color = c;
    // Non-LED screens carry g = r / 4. The ratio survives a clear pane's coverage
    // blending, so a faded screen marker cannot be mistaken for condensation.
    // Water keeps its occluded sky weight in g and reflection weight in b.
    // A vehicle's shadow is light blocked from the road, not a new dry surface. Its
    // colour still blends normally, but it must preserve the road's reflection mask.
    // A transparent screen layer must leave the masks underneath intact. Otherwise
    // unused LED colour layers make the whole panel's background bloom as lit dots.
    let coverage = select(c.a, 0.0, in.params2.w > 1.5);
    // (r under 0.5: how much light the player's bus's misted glass scatters there, for the
    // tone mapping's blur - post.wgsl `misted`. The mask is blended by the coverage, so the
    // share is divided by it to land as MIST_MASK x the share over what lies behind;
    // a misted pane's coverage is at least half its share, see `shade_enhanced`)
    let mist = select(0.0, glass_fog * MIST_MASK / max(coverage, 1e-3), glass_fog > 0.001);
    let screen_code = select(0.25, 1.0, led > 0.5);
    out.mask = vec4<f32>(select(min(mist, 0.49), 1.0, screen), select(max(led, puddle_weight.y * 0.49), screen_code, screen), puddle_weight.x, coverage);
    //RT out.gbuf = rt_gbuf;
    //RT out.aux = rt_aux;
    return out;
}

@fragment
fn fs_enhanced(in: FsIn) -> EnhancedOut {
    return enhanced_surface(in);
}

// A headset uses mapped shadows, not screen-space ray reflections. Its colour
// pass needs only colour and the screen mask, even on an Enhanced+ renderer.
struct EnhancedColorOut {
    @location(0) color: vec4<f32>,
    @location(1) mask: vec4<f32>,
};

@fragment
fn fs_enhanced_color(in: FsIn) -> EnhancedColorOut {
    let surface = enhanced_surface(in);
    return EnhancedColorOut(surface.color, surface.mask);
}

// A self-lit picture (a display, a light-mapped surface lit fully) at its own colour after
// the tone curve (post.wgsl `natural_tone`): it bleaches what is brighter than its knee -
// a display's yellow-green text came out olive - and gives the middle tones the contrast
// of `enh.debug.w` about mid grey. So the colour is kept under the knee and the contrast
// undone in advance (with the small black offset the text has always been drawn with).
// A display's own brightness for the light round it: as a bus's screens dim at night, to
// 15 % of the day's, and never lifted over their own colour by the dark-adapted metering.
fn display_dim(lift: f32) -> f32 {
    let night = clamp(camera.sun_color.w, 0.0, 1.0);
    return mix(lift, min(lift, 1.0) * 0.15, night);
}

fn display_level(t: vec3<f32>) -> vec3<f32> {
    let peak = max(t.r, max(t.g, t.b));
    let tk = t * min(1.0, 0.64 / max(peak, 1e-3));
    let c = max(enh.debug.w, 1.0);
    let x = 0.18 * pow(tk / 0.18 + vec3<f32>(1e-7), vec3<f32>(1.0 / c));
    return x + 0.04 * smoothstep(vec3<f32>(0.0), vec3<f32>(0.08), x);
}

// [matl_glow] (see MaterialExtra::glow): the material is its own light. `buv` is where its
// mask is read (the light map's own coordinate, the slot it rides in) and `col` the colour
// the light is drawn in - the material's own, so a destination panel keeps the colour its
// display draws. The mask is a greyscale picture of how much shines where (white full, black
// none). The strength is the mod's own (the .cfg value x0.25, on the scale of the LED
// panels' `Led glow`) and is held against the metering as an LED panel's dots are; the
// glare round it comes from how bright it is, as every light's does (post.wgsl).
fn matl_glow_light(buv: vec2<f32>, col: vec3<f32>) -> vec3<f32> {
    let gm = textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], buv, camera.lod.x);
    return col * dot(gm.rgb, vec3<f32>(0.299, 0.587, 0.114)) * material.glow.x * max(enh.exposure.z * 2.0, 0.8);
}

// How much of the light crossing the player's bus's pane its condensation scatters (0..1,
// 07_glass.wgsl `glass_condensation`), for the screen mask.
var<private> glass_fog: f32;
// The screen mask's r for a pane scattering all the light (post.wgsl `MIST_MASK`).
const MIST_MASK: f32 = 0.24;
// Heading, pitch and roll: with the heading alone the film slid along the glass on a hill (#3958).
fn pane_local(world: vec3<f32>) -> vec3<f32> {
    let d = world - camera.inside_a.xyz;
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let v = vec3<f32>(d.x * ch - d.y * sh, d.x * sh + d.y * ch, d.z);
    let t = enh.pane_tilt;
    let p = vec3<f32>(v.x, v.y * t.y + v.z * t.x, v.z * t.y - v.y * t.x);
    let b = vec3<f32>(p.x * t.w - p.z * t.z, p.y, p.z * t.w + p.x * t.z);
    return b - camera.inside_c.xyz;
}

fn shade_enhanced(in: FsIn, puddle_weight: ptr<function, vec2<f32>>, capture: bool, eye: vec3<f32>) -> vec4<f32> {
    // (the pane's water, read once for the uses below)
    let film_water = window_wetness(in);
    if (material.emissive.w > 1.5) {
        // a pane's film of water: drops laid out on the glass, each a lens that mirrors the
        // sky probe and shows the street upside down through itself, and the haze of the wet
        // glass between them (07_glass.wgsl `glass_film`)
        return glass_film(in, film_water);
    }
    // --- the surface's texture and alpha, exactly as the vanilla pass reads them
    let terrain = material.extra.x > 0.5;
    var duv = tex_address(in.uv);
    if (terrain) {
        duv = in.uv * material.extra.z;
    }
    // (without the [texcoordtransX/Y] offset: the transmap, night map and light map stay
    // in place, see fs_main)
    let buv = tex_address(in.uv - in.params.zw);
    let msk_lod = led_lod(buv, vec2<f32>(textureDimensions(mx_tex2d[mx_mat.trans])));
    // Empty paint also needs no tiled diffuse or detail samples. Its brush mask
    // replaces diffuse alpha, so coverage can be checked before those reads.
    if (TERRAIN_PAINT && material.params.x > 1.5 && material.params.z > 0.5
        && material.params.w > 0.5 && enh.debug.x <= 0.5 && material.ambient.w <= 1.5) {
        var coverage = sample_transmap(buv).a;
        if (material.emissive.w < -1.5 && enh.led.y < msk_lod) {
            coverage = textureSampleLevel(mx_tex2d[mx_mat.trans], mx_samplers[mx_mat.sampler], buv, enh.led.y).a;
        }
        coverage = smoothstep(0.32, 0.68, coverage);
        if (coverage * material.color.a * in.params.x == 0.0) {
            discard;
        }
    }
    // An LED panel is sampled at the level its screen footprint asks for, held at
    // `enh.led.y` (`Led mip strength`): its dots keep their gaps much further out than the
    // full chain allows, and the shimmer is a fraction of a full-resolution sample's. The
    // levels are worked out here and not inside the branch: derivatives are undefined in
    // non-uniform control flow, and which of the two samples runs is per draw. When the
    // setting does not bite, the plain (anisotropic) sample of the hardware is the better
    // one and stays.
    let pic_lod = led_lod(duv, vec2<f32>(textureDimensions(mx_tex2d[mx_mat.diffuse])));
    let led_pic = material.emissive.w < -1.5 && enh.led.y < pic_lod;
    // The painted ground's albedo comes from the picture `fs_ground` laid (the base's and its
    // layers', the frame's `camera.lod.y`): the terrain's base is lit once for all of them.
    // (A condition of the draw alone, so the texture reads below stay in uniform control flow;
    // `Renderer::plan_ground` has seen to it that every draw this holds for is in that picture.)
    let from_ground = camera.lod.y > 0.5 && terrain && !TERRAIN_PAINT && !capture && material.params.x < 0.5 && material.params.z < 0.5;
    var tex = vec4<f32>(1.0);
    if (from_ground) {
        tex = vec4<f32>(textureLoad(t_ao, vec2<i32>(in.clip.xy), 0).rgb, 1.0);
    } else {
        tex = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, camera.lod.x), duv);
        if (is_snow_pane()) {
            tex = glass_snow(in, film_water);
        }
        if (led_pic) {
            tex = diffuse_border(textureSampleLevel(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, enh.led.y), duv);
        }
    }
    let diffuse_a = tex.a;
    if (terrain && !from_ground && material.extra.y > 0.0) {
        let det = textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], in.uv * material.extra.y, camera.lod.x);
        tex = vec4<f32>(clamp(tex.rgb * det.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), tex.a);
    }
    if (material.params.z > 0.5) {
        // (an LED panel's `\S:n` mask is taken the same way: the dots stay dots when the
        // panel is small, without the full-resolution shimmer)
        var tm = sample_transmap(buv);
        if (material.emissive.w < -1.5 && enh.led.y < msk_lod) {
            tm = textureSampleLevel(mx_tex2d[mx_mat.trans], mx_samplers[mx_mat.sampler], buv, enh.led.y);
        }
        tex.a = select(1.0, tm.a, material.params.w > 0.5);
        if (terrain && material.params.x > 1.5) {
            // Coverage belongs to the brush mask, not the angle-dependent diffuse mip.
            tex.a = smoothstep(0.32, 0.68, tex.a);
        }
    }
    // One night-map read supplies both coverage and lighting. Keep the tile map
    // in its north-up UVs; vehicle maps use the unscrolled material UVs.
    var night_sample = vec4<f32>(0.0);
    if (material.extra.w > 0.5) {
        let nuv = select(buv, vec2<f32>(in.uv.x, 1.0 - in.uv.y), terrain);
        night_sample = sample_nightmap(nuv);
    }
    tex.a = night_coverage(tex.a, night_sample.a) * tree_card_fade(in.world);
    let mode = material.params.x;
    // Keep the filtered fractional coverage, but tighten its transition around the cutout
    // edge before MSAA turns it into sample coverage. The MSAA depth prepass skips these
    // draws so uncovered samples keep the depth and colour of the scene behind them.
    if ((ALPHA_TEST || capture) && mode > 0.5 && mode < 1.5) {
        if (ALPHA_TO_COVERAGE) {
            let aa = max(fwidth(tex.a) * 0.5, 1.0 / 255.0);
            // (not `ALPHA_REF - aa`: a layer whose clear glass is alpha 128 kept a third of
            // its samples there, a moire of the layer's paint over every window)
            if (tex.a < ALPHA_REF) {
                discard;
            }
            tex.a = smoothstep(ALPHA_REF - aa, ALPHA_REF + aa, tex.a);
        } else if (tex.a < ALPHA_REF) {
            discard;
        }
    }
    // (a blended slot that keeps Omsi.exe's alpha test, `MaterialExtra::alpha_cut`: a
    // livery's clear glass is cut away, its tinted glass blended)
    tex.a = select(tex.a, 0.0, mode > 2.5 && tex.a < ALPHA_REF);
    var alpha = tex.a * material.color.a;
    if (mode < 0.5) {
        alpha = 1.0;
    }
    if (!is_snow_pane()) {
        alpha = alpha * clamp(film_water, 0.0, 1.0);
    }
    // Sparse brush masks still cover the whole tile mesh. Empty pixels contribute
    // neither colour nor reflection coverage, so avoid lighting them. Keep fractional
    // edges, debug views, and water (whose Fresnel can raise zero alpha) unchanged.
    if (TERRAIN_PAINT && alpha == 0.0 && enh.debug.x <= 0.5 && material.ambient.w <= 1.5) {
        discard;
    }
    let pre = enh.exposure.x;
    let to_cam = eye - in.world;
    let dist = length(to_cam);
    let v = to_cam / max(dist, 1e-4);
    let h_cam = camera.cam_pos.z - enh.fog.z;
    let h_pt = in.world.z - enh.fog.z;
    // (no fog inside the cab: only the part of the way outside the bus is misty)
    // (the lamps' light in the fog is added over the whole picture afterwards: fog_lamps.wgsl)
    let aer = air(-v, fog_distance(in.world), h_cam, h_pt);
    if (in.params2.w > 1.5) {
        // A vehicle's shadow blob is no surface: it is the sky light the body keeps off the
        // road, a layer that only darkens what lies under it (as OMSI blends it). Shaded as
        // a black surface it took the sky, the wet road's sheen and the lamps on top and
        // hardly darkened anything. The air in front of it stays (the road's own fog, under
        // the blob, is darkened with it as in the original).
        return vec4<f32>(aer.rgb * pre, alpha * aer.a);
    }
    if (material.params.y > 0.5) {
        // unlit (mirror glass, script and text textures): shown at their own brightness,
        // under the tone curve's knee (`display_level`), and `exposure.y` undoes the
        // metering (see ExposureLog in lib.rs)
        let t = tex.rgb * material.color.rgb;
        // A mirror (params.y 0.9) is no display: its picture is the street drawn a moment
        // ago, dark at night. Brightened like a display by the metering (up to 1.6 in the
        // dark) it showed a street far brighter than the one through the windscreen.
        let lift = select(display_dim(enh.exposure.y), min(enh.exposure.y, 1.0), material.params.y < 0.95);
        // (no [matl_glow] here: an unlit slot is drawn at its own brightness already, and
        // reading the light map's slot in this branch failed Metal's shader compiler)
        let c = display_level(t) * lift;
        return vec4<f32>(c * aer.a + aer.rgb * pre, alpha);
    }
    let outside = weather_outside_n(in.world, safe_normal(in.normal), terrain, in.params2.w);
    // the normal as the content gives it: never turned towards the viewer (Direct3D does
    // not either, and OMSI's foliage points every leaf's normal up so the whole crown is lit
    // evenly - turned round, the crown went dark above the horizon line)
    var n = safe_normal(in.normal);
    // the map's water (`MaterialExtra::water`): small waves running over it ripple its
    // normal - fading with the distance, where they would only flicker - so that what it
    // mirrors breaks up as on a lake instead of standing in it as in a pane of glass (#841)
    let is_water = material.ambient.w > 1.5;
    if (is_water) {
        n = normalize(n + vec3<f32>(water_ripple(world_pattern_xy(in.world), camera.post.y) * clamp(1.0 - dist / 250.0, 0.2, 1.0), 0.0));
    }
    // Leaves and fences transmit light. A terrain road-cut mask only removes ground:
    // its remaining pixels must shade like the uncut ground on terrain-mapped splines.
    // (people's cut-out hair and clothes are bodies: material flag 8)
    let thin = !terrain && mode > 0.5 && mode < 1.5 && (u32(material.params2.w + 0.5) & 8u) == 0u;
    let has_env = material.params2.y > 0.0;
    // A blended transmap body is a masked paint surface, not glass. Traffic cars often
    // use this material layout for their body; depth-disabled blends remain glass.
    // Window assets are not consistent about carrying an envmap.  The reliable signal
    // is an explicitly named window, or a depth-disabled blended layer with env/transmap
    // data that identifies a pane rather than a dirt/text overlay. This keeps bus
    // panes on the reflection/transmission path after the material-depth repair.
    // Painted terrain also blends a transmap without writing depth. It is never glass:
    // Fresnel opacity on its empty mask pixels darkens every lower layer at grazing angles.
    // A script's screen is never a pane, though it is laid out as one - blended, not writing
    // depth, its letters a `\S:n` transparency map: the EMT Citaro's Krueger matrix
    // (Mallorca) was taken for glass, which gives no light of its own, and its letters stayed
    // grey at night while its light map lit them in Vanilla+ (#2927).
    let glass = !terrain && mode > 1.5 && (material.bump.z > 0.5 || material.emissive.w > 0.5) &&
        (has_env || material.params.z > 0.5 || material.emissive.w > 0.5) && material.flags.x < 0.5;
    // An envmap on opaque vehicle paint is legacy material data, not a request to make
    // the whole body behave like glass. Keep the diffuse/sun response, but use the
    // ordinary rough dielectric path for the body.
    // Foliage and other alpha-tested assets are diffuse silhouettes, not polished surfaces.
    // Letting the generic probe term reflect them creates white sparkles at grazing angles;
    // the photographed envmap is reserved for materials that explicitly request it.
    // (opaque materials reflect as well - chrome handrails, bumpers and wheel trims are
    // opaque with a sphere map, and left out they showed no reflection at all in the
    // enhanced picture, #266 - but through the clear-coat path below: metal only where a
    // mask of its own says so)
    // (not the water: its sphere map is OMSI's stand-in for the sky it mirrors, which the
    // probe holds itself - drawn as paint's photo, it lay on the water as green specks)
    // (a body painted through a [matl_transmap] too - most mod buses' and cars' paint,
    // the W123's: its alpha is the transmap's, the envmap's mask the texture's own, as in
    // the vanilla picture; left out, their paint showed no reflection at all while their
    // windows did)
    let reflective_env = has_env && !thin && !is_water;
    // see-through glass is seen from either side: from inside the bus its normal points
    // away (the vanilla pass takes the angle either way round too); taken as it is, the
    // grazing Fresnel turned the whole windscreen into a milky mirror
    var geo_n = n;
    if (glass && dot(n, v) < 0.0) {
        n = -n;
        geo_n = n;
    }
    var face_n = n;
    var face_v = 0.0;
    var face_w = 1.0;
    if (reflective_env && !glass && !terrain) {
        let c = cross(dpdx(in.world), dpdy(in.world));
        if (dot(c, c) > 1e-14) {
            let face = normalize(c) * select(-1.0, 1.0, dot(c, v) >= 0.0);
            face_n = face;
            face_v = dot(face, v);
            face_w = smoothstep(0.5, 0.9, dot(n, face));
        }
    }
    if (reflective_env && material.bump.y > 0.5) {
        // [matl_bumpmap]: the height map's slope bends the normal (tangent frame from the
        // screen-space derivatives of position and texture coordinate)
        let slope = bump_offset(duv);
        let dp1 = dpdx(in.world);
        let dp2 = dpdy(in.world);
        let du1 = dpdx(duv);
        let du2 = dpdy(duv);
        let dp2p = cross(dp2, n);
        let dp1p = cross(n, dp1);
        let t = dp2p * du1.x + dp1p * du2.x;
        let b = dp2p * du1.y + dp1p * du2.y;
        let inv = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-12));
        n = normalize(n - (t * inv * slope.x + b * inv * slope.y) * 0.6);
    }
    var albedo = select(tex.rgb * material.color.rgb, tex.rgb, from_ground);
    // OMSI materials have separate diffuse and ambient colours. Some interiors have
    // black diffuse but white ambient: using diffuse for both made them pitch black.
    // Keep the direct response, and weather both colours with the same surface effects.
    // A slot without a texture (ambient.w -1) takes its diffuse colour for both: Omsi.exe's
    // white ambient of every o3d slot under a strong sky turned a coloured model white (#1737).
    var ambient_albedo = select(tex.rgb * material.ambient.rgb, albedo, material.ambient.w < -0.5 || from_ground);
    var detail_factor = 1.0;
    if (camera.flags.x > 0.5 && (terrain || in.params2.w > 0.5)) {
        let k = clamp(1.0 - (dist - 25.0) / 120.0, 0.0, 1.0);
        let pattern_xy = world_pattern_xy(in.world);
        detail_factor = 1.0 + (detail_noise(pattern_xy, k) - 0.5) * 0.42 * k;
        albedo = albedo * detail_factor;
        ambient_albedo = ambient_albedo * detail_factor;
    }
    // --- the material in physical terms
    var refl = 0.0;
    if (reflective_env) {
        // (with a mask of its own or a [matl_transmap] the share is the mask's alone, as
        // Omsi.exe's stage takes it - the factor counts only where there is none)
        let by_mask = material.params.z > 0.5 || (u32(material.params2.w + 0.5) & 1u) != 0u;
        refl = clamp(select(min(material.params2.y, 1.0), 1.0, by_mask) * reflection_mask(duv, diffuse_a), 0.0, 1.0);
    }
    var metal = 0.0;
    var f0 = vec3<f32>(0.04);
    var rough = 0.8;
    let probe_only = camera.clouds.w < 1.5;
    if (terrain) {
        rough = 0.92;
        f0 = vec3<f32>(0.03);
    } else if (glass) {
        // OMSI's window materials are deliberately very transparent. Raise the base
        // dielectric reflection slightly so the outside scene remains readable through
        // the pane instead of leaving a flat pale surface at normal incidence.
        rough = 0.04;
        // the pane's [matl_envmap] factor says how much it mirrors (its alpha is its
        // transparency, never a mask): from glass's own 4 % up to 12 % for a factor of 1
        // (0.4 on the Scania's panes). Up to 26 % as before, every window of every bus
        // was a mirror - far more than the original's panes reflect (issue #176).
        // ... measured against the vanilla picture: Omsi.exe lays the sphere map over the
        // pane by its factor and the pane shows that as faintly as it shows itself - the
        // share it mirrors is the pane's alpha times the factor (0.25 .. 0.5 on the stock
        // windows), at every angle, with no Fresnel term. Taken as the reflectance
        // averaged over the angles (Schlick's mean, f0 + (1 - f0) / 21, as for the paint),
        // the pane mirrors as much as there and rises towards grazing as glass does; 4 to
        // 12 % at the normal, and three quarters of that, left the windows showing the
        // saloon where the vanilla picture shows the street.
        let share = clamp(alpha * min(material.params2.y, 1.0), 0.0, 1.0) * camera.cam_pos.w;
        f0 = vec3<f32>(select(clamp((share - 1.0 / 21.0) * 21.0 / 20.0, 0.04, 0.9), 0.04, probe_only));
    } else if (reflective_env) {
        // Paint reflects its few per cent through a smooth clear coat; much more than a few
        // per cent is polished metal - but only where the model says so with a mask of its
        // own ([matl_envmap_mask]). Nearly every car body carries `[matl_envmap] â€¦ 1` with
        // no mask, which OMSI takes as "reflect the sphere map fully" and not as chrome:
        // read as metalness it made a Golf's bonnet a mirror, in which the envmap photo's
        // trees stood as contour lines across the paint at close range.
        //
        // The paint itself is a clear coat: a lacquer of refractive index 1.5 (4 % at normal
        // incidence, Fresnel's 100 % at grazing) polished smooth, which mirrors the street
        // as sharply as the windows do. How much of the sphere map OMSI lays over a
        // material says how glossy it is: the stock bodies carry 1 or 10, the cabin's
        // panels, frames and plastic trim 0.05 .. 0.2 - satin, a blurred sheen; the mask
        // (the texture's alpha) says where. (Roughness
        // 0.22 for every body, against the photo's contour lines on a car's bonnet, left
        // the paint matte beside its sharp windows.)
        let masked = (u32(material.params2.w + 0.5) & 1u) != 0u;
        let metal_ok = (u32(material.params2.w + 0.5) & 4u) != 0u;
        metal = select(0.0, smoothstep(0.3, 0.85, refl), masked || metal_ok);
        // The coat is as smooth as lets it mirror, averaged over the angles, the share
        // Omsi.exe lays over it: a polished 4 % coat mirrors 8.7 % on average (Fresnel's
        // rise at grazing), a rough one less (2.5 % at 0.8, its grazing light scattered and
        // shadowed by its own roughness). So a share of 8.7 % and more is smooth lacquer, the
        // stock bodies' few per cent - the MAN NL's 15 of 255 - a satin one (taken as rough
        // up to a share of a half they came out matte, #2067; smooth from 4 %, a body meant
        // to mirror barely shone like glass against the light, #2474), and where the mask
        // says nothing reflects - a tyre or a seal - the rough rubber or plastic it is.
        // Between those and chrome, the share says how much the paint mirrors, not how
        // sharply: Omsi.exe lays a soft sphere map over it, a blurred sheen of the sky. Taken
        // as polished glass from 8.7 % on (0.2.42), the many mod bodies whose texture alpha
        // holds a tenth to a half mirrored the grey sky like chrome - silver buses and AI
        // cars that glowed (#2563, #2301). Only a share near the whole is a sharp mirror.
        let coat_rough = max(clamp(0.06 + (0.087 - refl) * 11.9, 0.06, 0.8), mix(0.3, 0.06, smoothstep(0.5, 0.95, refl)));
        // Where the model says how much it mirrors - a [matl_envmap_mask], or the texture's
        // alpha under a [matl_transmap] (most mod bodies) - Omsi.exe lays the sphere map
        // over the paint by that share at every angle (shader.wgsl's environment stage):
        // OMSI has no Fresnel term, so the share is the reflectance averaged over the angles
        // the surface is seen at. A Fresnel reflectance with that average (Schlick's, whose
        // hemispherical mean is f0 + (1 - f0) / 21) mirrors as much as the vanilla picture
        // and rises towards grazing as a real coat does. (A plain 4 % coat left those
        // bodies far fainter than in the vanilla picture.) A body with neither keeps the
        // lacquer's 4 %: there Omsi.exe's lerp by the factor alone made every Golf a mirror.
        let authored = masked || material.params.z > 0.5;
        let coat = select(0.04, clamp((refl - 1.0 / 21.0) * 21.0 / 20.0, 0.04, 0.95), authored);
        f0 = mix(vec3<f32>(coat), mix(albedo, vec3<f32>(1.0), 0.4) * refl, metal);
        rough = mix(coat_rough, 0.14, metal);
    } else if (!thin && material.specular.w > 0.0 && dot(material.specular.rgb, vec3<f32>(1.0)) > 0.05) {
        // the o3d material's Blinn-Phong power as GGX roughness
        rough = clamp(sqrt(sqrt(2.0 / (material.specular.w + 2.0))), 0.4, 0.9);
    } else if (thin) {
        rough = 0.7;
    }
    if (is_water) {
        // water: a dielectric of 2 % at normal incidence, nearly a mirror where it is seen
        // flat, smooth but for its waves
        f0 = vec3<f32>(0.02);
        rough = 0.06;
        metal = 0.0;
    }
    // --- a PBR set beside the diffuse texture (`foo_n.png`, `foo_r` / `_m` / `_ao` or
    // `foo_orm`: see omsi_texture::pbr): the normal map bends the normal, the packed map
    // gives the occlusion, roughness and metalness in place of the guesses above
    var pbr_ao = 1.0;
    if (material.pbr.x > 0.5 && !terrain) {
        var tn = textureSampleBias(mx_tex2d[mx_mat.pbr_normal], mx_samplers[mx_mat.sampler], duv, camera.lod.x).xyz * 2.0 - vec3<f32>(1.0);
        // (an OpenGL-style map, green up: `_gl` in its name)
        if (material.pbr.x > 1.5) {
            tn.y = -tn.y;
        }
        n = perturb_normal(n, in.world, duv, tn);
    }
    if (material.pbr.y + material.pbr.z + material.pbr.w > 0.5 && !terrain) {
        let orm = textureSampleBias(mx_tex2d[mx_mat.pbr_orm], mx_samplers[mx_mat.sampler], duv, camera.lod.x).rgb;
        if (material.pbr.y > 0.5) {
            pbr_ao = orm.r;
        }
        if (material.pbr.z > 0.5) {
            rough = clamp(orm.g, 0.03, 1.0);
        }
        if (material.pbr.w > 0.5) {
            metal = orm.b;
            f0 = mix(vec3<f32>(0.04), albedo, metal);
        }
    }
    // rain: roads with [moisture] turn dark and smooth, everything outside a little glossier
    // (not under snow: a snowy road took the rain's gloss and the snow on it shone like
    // plastic in every headlight)
    let dry_snow = 1.0 - clamp(enh.weather.y, 0.0, 1.0);
    let wet_road = camera.shadow.w * material.params2.z * outside * dry_snow;
    let wet_any = camera.shadow.w * outside * select(0.35, 0.0, glass) * dry_snow;
    var puddle = 0.0;
    if (wet_road > 0.0) {
        albedo = albedo * mix(1.0, 0.5, wet_road);
        ambient_albedo = ambient_albedo * mix(1.0, 0.5, wet_road);
        rough = mix(rough, 0.12, wet_road);
        f0 = mix(f0, vec3<f32>(0.02), wet_road);
        // Puddles: standing water pools in low, flat spots rather than spreading evenly
        // over the whole carriageway the way the sheen above does. A world-space mask at a
        // frequency of its own (unrelated to the road's detail noise, so the two do not
        // look tied together) picks out patches that grow with the wetness and vanish again
        // as the road dries; a puddle goes glass-flat and mirrors what is above it. While it
        // is actually raining rather than merely wet, and never while it is snow settling
        // instead (`enh.weather.z` is fed by any precipitation, `.y` singles snow back out),
        // concentric rings cross a puddle where drops land, fading as they widen - the other
        // half of the wheel splashes in `puddles.rs`.
        let pattern_xy = world_pattern_xy(in.world);
        // (the pools: see `road_puddle_coverage`, shared with the classic picture)
        puddle = road_puddle_coverage(in.world, n, wet_road);
        if (puddle > 0.001) {
            // A drop is a few millimetres across and its ring dies away within a hand's
            // breadth, so the shared ripple grid is 12.5 cm wide and a ring grows
            // to RIPPLE_MAX of a cell, about 5 cm: at 2 m cells with rings a metre and a half
            // across, every drop looked like a puddle of its own. Cells run at their own pace
            // and only some of them carry a drop at a time, so the surface reads as many small
            // impacts rather than one pulsing pattern.
            let raining = enh.weather.z * (1.0 - enh.weather.y);
            // (faded out where the pixel is wider than half a ring's wave: past a few
            // metres they only made the water shimmer)
            let ripple_fp = max(length(dpdx(in.world)), length(dpdy(in.world)));
            let ripple = puddle_ripple_seen(pattern_xy, camera.post.y, raining, puddle, ripple_fp);
            // Standing water hides most of the fine asphalt grain. Keep dry and damp
            // asphalt's detail, but let the reflected image read across a filled pool.
            albedo = albedo * (1.0 - 0.68 * puddle) / mix(1.0, detail_factor, puddle * 0.8);
            ambient_albedo = ambient_albedo * (1.0 - 0.68 * puddle) / mix(1.0, detail_factor, puddle * 0.8);
            rough = clamp(mix(rough, 0.03, puddle) - ripple.z * 0.12, 0.02, 1.0);
            f0 = mix(f0, vec3<f32>(enh.debug.y), puddle);
            n = normalize(mix(n, geo_n, puddle) + vec3<f32>(ripple.xy, 0.0));
        }
    } else if (wet_any > 0.0) {
        // (a body's paint wears a film of water as smooth as glass: the street and the sky
        // mirror in a wet bus as sharply as in its windows)
        // (Lagarde's wet surfaces: the water stands on what faces up and runs off a wall,
        // so a wall only darkens where a ledge or a step goes glossy)
        let up = smoothstep(0.1, 0.9, geo_n.z);
        // (the rougher, the more porous - soil, plaster, unglazed stone - and the more water
        // it takes in; a smooth, sealed surface keeps its colour under the film)
        let porous = smoothstep(0.25, 0.7, rough);
        if (!terrain) {
            rough = mix(rough, select(mix(rough * 0.75, min(rough, 0.2), up), min(rough, 0.04), reflective_env), select(wet_any, min(wet_any * 2.6, 1.0), reflective_env));
        }
        // Soaked by the rain, a porous surface - soil, a paving stone, plaster, bark -
        // turns darker and deeper in colour: the light that enters the water film and is
        // scattered back by the surface under it is partly reflected down again at the
        // film's top (total internal reflection), and meets the surface once more, which
        // absorbs a share of it each time (Lekner and Dorf, "Why some things are darker
        // when wet", Applied Optics 1988). Painted metal, glass and leaves shed the water.
        let soak = camera.shadow.w * outside * dry_snow * select(0.75, 1.0, terrain) * mix(0.3, 1.0, porous)
            * mix(0.45, 1.0, clamp(n.z, 0.0, 1.0)) * select(1.0, 0.0, reflective_env || glass || is_water || thin || metal > 0.5);
        albedo = mix(albedo, wet_albedo(albedo), soak);
        ambient_albedo = mix(ambient_albedo, wet_albedo(ambient_albedo), soak);
    }
    // snow lies on what faces up
    // (not on a shadow blob: whitened, it lit the snow under the bus instead of shading it)
    // (nor on a texture that is the season's snow picture, which shows the map's own snow
    // as OMSI 2 does: see the vanilla shader, #879)
    let snow_ok = outside * select(1.0, 0.0, in.params2.w > 1.5 || material.ambient.w > 0.5);
    // a carriageway under snow that builds up: its own cover, ruts and tracks (road_snow.wgsl)
    let carriageway = !terrain && material.params2.z > 0.0 && road_snow_dynamic();
    var rs = vec3<f32>(0.0);
    if (carriageway && snow_ok > 0.0) {
        rs = road_snow(in.world);
    }
    // a vehicle's part: the snow its own roof has gathered, where that is kept (another
    // player's bus keeps the weather's)
    let vehicle = vehicle_snow_kept(in.params2.w);
    var on_vehicle = 0.0;
    if (vehicle && snow_ok > 0.0) {
        on_vehicle = vehicle_snow(in.params2.w, in.local);
    }
    let snow = select(select(enh.weather.y, rs.x, carriageway), on_vehicle, vehicle) * snow_ok;
    if (rs.y > 0.0) {
        // the slush in the ruts and the tracks: wet and grey, darker than the snow, glossy
        let slush = rs.y * snow_ok;
        albedo = mix(albedo, albedo * 0.55 + vec3<f32>(0.08), slush);
        ambient_albedo = mix(ambient_albedo, ambient_albedo * 0.55 + vec3<f32>(0.08), slush);
        rough = mix(rough, 0.35, slush);
    }
    if (snow > 0.0) {
        let up = clamp(n.z, 0.0, 1.0);
        let ground = select(0.0, 1.0, terrain || material.params2.z > 0.0);
        // (a road kept clear - "snow on road" off - stays asphalt: whitened, it turned the
        // roads white exactly when the weather says they are cleared, #1362; built-up snow
        // brings its own cover)
        let cleared = select(1.0, 0.0, !carriageway && camera.post.z > 0.5 && !terrain && material.params2.z > 0.0);
        // (a vehicle's gathered snow lies thick enough to hide the paint)
        let cover = min(cleared * snow * clamp(max(ground, smoothstep(0.78, 0.95, up) * 0.8), 0.0, 1.0) * select(0.55 + 0.35 * tex.a, 1.2, vehicle), 1.0);
        albedo = mix(albedo, vec3<f32>(0.82, 0.84, 0.88), cover);
        ambient_albedo = mix(ambient_albedo, vec3<f32>(0.82, 0.84, 0.88), cover);
        // fresh snow is all but matte: it scatters the light and shows no highlight
        rough = mix(rough, 0.95, cover);
        f0 = mix(f0, vec3<f32>(0.02), cover);
        metal = metal * (1.0 - cover);
    }
    // specular antialiasing: where the normal turns quickly across a pixel (a low-poly
    // bonnet close up, a curved body far away) the surface cannot reflect sharper than that
    // spread, or the interpolated normals show as bands in a mirror-like finish
    if (reflective_env && !glass) {
        let dndx = dpdx(n);
        let dndy = dpdy(n);
        let spread = min((dot(dndx, dndx) + dot(dndy, dndy)) * 2.0, 0.4);
        rough = sqrt(min(rough * rough + spread, 1.0));
    } else if (!thin && !glass) {
        rough = rough_aa(rough, n);
    }
    let env_share = select(1.0, camera.cam_pos.w, reflective_env && !glass);
    var sf: Surface;
    sf.albedo = albedo * (1.0 - metal);
    ambient_albedo = ambient_albedo * (1.0 - metal);
    sf.f0 = f0;
    sf.rough = rough;
    let nv = clamp(mix(face_v, dot(n, v), face_w), 1e-4, 1.0);
    // the specular reflectance over the hemisphere, its multiple scattering included: one
    // minus it is the share the diffuse part gets, so a surface never sends out more than falls
    // on it (and a rough metal is not darker than its colour)
    let spec_e = spec_ms(f0, rough, nv);
    sf.kd = vec3<f32>(1.0) - spec_e.rgb;
    sf.comp = spec_e.w;
    // (how wide a patch of the surface the pixel covers, m: see `relief_masking`)
    let relief_footprint = length(dpdx(in.world)) + length(dpdy(in.world));
    // The same depth-aware traced sample feeds the sun, AO and the debug view. Repeating
    // rt_at here and inside ao_gi_at overflowed the optimizer's stack in the DX12 FXC
    // fallback (used when dxcompiler.dll cannot be loaded); DXC compiles either form.
    // Read it once after the early material returns, with the same taps and their order,
    // so that the fallback can compile fs_enhanced too.
    let surface_rt = rt_at(in.clip.xy, in.world);
    // --- the sun
    let s = camera.sun_dir.xyz;
    let nl = dot(n, s);
    var direct = vec3<f32>(0.0);
    if (enh.lights.w > 0.0 && max(enh.sun.r, enh.sun.g) > 1e-5) {
        var shadow = 0.0;
        // (Enhanced+ without a traced shadow for the surface: the far map's as well, below -
        // one call of it for both cases: two, each inlined, made the scene shader too big for
        // Apple's M4 compiler, whose service crashed on every scene pipeline)
        var far = false;
        if (glass && nl > 0.0) {
            // a pane lets nearly all of it through: the sun on it is its glint, for which one
            // compare says enough (the soft filter over a cab's windscreen, doors and side
            // windows was a millisecond of the frame)
            shadow = sun_shadow_hard(in.world, n);
            far = camera.clouds.w > 1.5;
        } else if (nl > 0.0 || thin) {
            // Enhanced+: the traced shadow where there is one for this surface
            // (whose cut-out leaves and fences the shadow map holds alone: see `render_inner`)
            let traced = surface_rt;
            shadow = sun_shadow_soft(in.world, n, thin);
            if (traced.w > 0.5 && traced.z >= 0.0) {
                shadow = shadow * traced.z;
            } else {
                far = camera.clouds.w > 1.5 && traced.w < 0.5;
            }
        }
        if (far) {
            shadow = shadow * sun_shadow_far(in.world, n);
        }
        // (only the beam casts the shadow: what the clouds scatter of it is the sky's, below)
        let e_sun = enh.sun.rgb * (shadow * cloud_shadow_ratio(in.world) * sun_beam_share());
        if (thin) {
            // foliage: a crown of leaves facing every way, whose normals OMSI points up
            // only to light it evenly - lit by the sun from any side (the shadow map
            // darkens its far side), a little through the leaves as well. A crown is
            // hundreds of leaves at every angle: on average about half of the sun falls on
            // them square on, whatever the sun's height (with the up-pointing normals alone a
            // low sun left every tree dull and dark while the walls beside it glowed). Seen
            // against the sun a leaf glows with what comes through it - yellow-green, the
            // light having passed the leaf's colour twice.
            let through = pow(clamp(dot(-v, s), 0.0, 1.0), 3.0);
            let leaf = sf.albedo / PI;
            direct = e_sun * (leaf * (0.42 + 0.3 * max(nl, 0.0) + 0.08 * max(-nl, 0.0))
                + leaf * min(sf.albedo * 2.2, vec3<f32>(1.0)) * through * 0.45);
        } else if (nl > 0.0) {
            // the sun is a disc, not a point: no highlight sharper than it
            let a = max(rough * rough, 0.012);
            let h = normalize(s + v);
            var spec = d_ggx(max(dot(n, h), 0.0), a) * v_smith(nv, nl, a) * f_schlick(f0, dot(v, h));
            // (a rough, unpolished surface's own relief hides its glint from a low sun: see
            // `relief_masking`; glass, paint, chrome, water and a wet film keep theirs)
            if (!glass && !reflective_env && !is_water && metal < 0.5) {
                spec = spec * relief_masking(nl, nv, rough, relief_footprint);
            }
            // Optional road glare: only the specular sun highlight on terrain or roads.
            // Diffuse sunlight, wet reflections and vehicle paint remain unchanged.
            if ((terrain || material.params2.z > 0.0) && !is_water) {
                spec = spec * camera.lod.z;
            }
            spec = select(spec, vec3<f32>(0.0), dot(geo_n, v) < 0.0 && !is_water);
            direct = e_sun * nl * (sf.albedo / PI * sf.kd + spec * sf.comp);
        }
    }
    // --- the moon: by night a directional light as the sun is, with its shadow when the
    // maps are drawn along it
    let em = enh.moon_light.rgb;
    if (max(em.r, em.g) > 1e-9) {
        let m = enh.moon.xyz;
        let nlm = dot(n, m);
        if (nlm > 0.0 || thin) {
            var ms = 1.0;
            if (enh.moon_light.w > 0.5) {
                ms = sun_shadow_soft(in.world, n, thin);
            }
            let leaf = sf.albedo / PI;
            if (thin) {
                direct = direct + em * ms * leaf * (0.42 + 0.3 * max(nlm, 0.0));
            } else {
                direct = direct + em * ms * nlm * leaf;
            }
        }
    }
    // --- sky and ground
    var ao = 1.0;
    let screen_ao = camera.clouds.w > 0.5 && !capture;
    var bounce = vec3<f32>(0.0);
    if (screen_ao) {
//VK         ao = ao_traced_at(in.clip.xy, in.world, surface_rt);
        let ag = ao_gi_at(in.clip.xy, in.world, surface_rt);
        ao = ag.x;
        bounce = ag.yzw;
    }
    // AO is generated from the opaque depth buffer.  A transparent bus pane therefore
    // samples the seats/dashboard behind it and makes their dark outlines crawl across the
    // glass as the camera moves.  The pane has its own transmission/reflection path; it must
    // not inherit occlusion belonging to geometry on the far side of the window.
    // (nor any other blended surface: a van's translucent door showed the shade of what
    // stood behind it)
    if (glass || mode > 1.5) {
        ao = 1.0;
    }
    ao = ao * pbr_ao;
    let fr = f_schlick(f0, nv);
    // inside the player's vehicle the light comes in through the windows and off the
    // cabin's own walls: less of it, and neither as blue nor as directional as the sky's
    let in_cab = 1.0 - outside;
    // The cab's light is one even ambient already (cab_e below); the screen-space occlusion
    // on top of it went black in the hollow under the windscreen - the steering wheel and
    // the dashboard stood dark beside a brightly lit cash desk. Inside the cab it is taken
    // at a third of its strength.
    ao = mix(ao, 1.0 - (1.0 - ao) * 0.35, in_cab);
    let avg_e = enh.fog_color.rgb * (PI / 0.9);
    let cab_e = vec3<f32>(dot(avg_e, vec3<f32>(0.2126, 0.7152, 0.0722))) * vec3<f32>(1.0, 0.98, 0.95);
    // (the sun the clouds scatter: shadowless, a light of the sky's kind)
    var sky_e = sh_irradiance(n);
    let beam = sun_beam_share();
    if (beam < 0.999 && enh.lights.w > 0.0) {
        sky_e = sky_e + sun_scattered(n, enh.sun.rgb * cloud_shadow_ratio(in.world) * (1.0 - beam));
    }
    var e_amb = mix(sky_e, cab_e * CAB_AMBIENT, 0.6 * in_cab) * ao_bounce(ao, ambient_albedo)
        + flash_irradiance(n) * ao * (1.0 - 0.7 * in_cab);
//VK     if (screen_ao && in_cab < 1.0) {
//VK         e_amb = e_amb + bounce_at(in.clip.xy, in.world) * mix(1.0, ao, 0.5) * (1.0 - in_cab);
//VK     }
    e_amb = e_amb + bounce * mix(1.0, ao, 0.5) * (1.0 - in_cab);
    var ambient = e_amb * ambient_albedo / PI * (vec3<f32>(1.0) - spec_e.rgb * env_share);
    // --- reflections: the sky probe, as sharp as the surface is smooth
    var r = reflect(-v, normalize(mix(face_n, n, face_w)));
    let bend_n = normalize(mix(face_n, geo_n, face_w));
    let horizon = horizon_occlusion(r, bend_n);
    // a reflection pointing into the surface would show the probe's ground: bend it up
    let below = dot(r, bend_n);
    if (below < 0.0) {
        r = normalize(r - bend_n * below * 1.02);
    }
    // A wet road, a wet pavement is no mirror: a soft, dim sheen of what stands round it, a
    // puddle a blurred picture of it (panes, envmapped paint and still water keep their own
    // smoothness). Its reflection is taken this rough, at this share.
    let wet_only = !(reflective_env || glass || is_water || ((material.pbr.z > 0.5 || material.pbr.w > 0.5) && !terrain));
    let refl_rough = select(rough, max(rough, mix(0.2, 0.16, puddle)), wet_only);
    let wet_share = select(1.0, mix(0.6, 0.7, puddle), wet_only);
    let lod = refl_rough * (enh.lights.x - 1.0);
    var env = textureSampleLevel(t_probe, s_lin, to_cube(r), lod).rgb * enh.fog_color.w;
    // the probe holds only the sky: in a street the low sky is hidden behind houses and
    // trees, which reflect about as much light as a quarter-white wall (without this every
    // surface wore a milky sheen of the bright horizon, strongest against a low sun)
    let surround = enh.fog_color.rgb * (0.25 / 0.9);
    let open = smoothstep(-0.05, 0.35, r.z);
    env = mix(surround, env, mix(open, 1.0, select(0.25, 0.5, reflective_env && !(glass && probe_only))));
    // the cabin reflects the cabin, not the sky - and so does the bus's own glass seen
    // from the driver's seat
    let cab_view = max(in_cab, near_player_vehicle(in.world) * inside_vehicle(camera.cam_pos.xyz));
    // From inside the player bus the pane is near the cabin, but it must still show the
    // outside sky/street. Using the opaque-surface cabin weight here made every window a
    // flat pale patch and hid the useful reflection while driving.
    // A pane is the boundary between the cabin and the outside. Mixing the cabin
    // irradiance into its reflection made every windscreen look like a pale grey sheet
    // from the driver's seat; the interior is already behind the alpha-blended pane.
    let cabin_reflection = select(0.85, 0.0, glass);
    env = mix(env, cab_e * 0.35 / PI, cabin_reflection * cab_view);
    // ... but the inner face of the bus's own glass cannot mirror the sky, which lies
    // behind it: it mirrors the cab, far darker than the street seen through it. Taken as
    // the sky, the doors (seen at a grazing angle from the driver's seat) went a milky
    // grey sheet in fog and snow, when the probe is bright all round.
    let own_pane = select(0.0, near_player_vehicle(in.world) * inside_vehicle(camera.cam_pos.xyz), glass);
    if (reflective_env) {
        // OMSI's photographed environment gives the reflection its structure (trees,
        // houses, the street), the probe its light
        // A pane looks it up by the reflection's direction in the world, not in the
        // camera's frame as Direct3D's sphere map does: taken from the camera, what a
        // standing bus's windows mirrored changed whenever the view turned, trees sliding
        // in from nowhere over the street seen through them. Height gives the photo's rows
        // (sky, the horizon's trees and houses, the ground), the compass direction a
        // gentle drift across it that stays put in the world.
        let az = atan2(r.y, r.x);
        var env_uv = vec2<f32>(0.5 + 0.3 * sin(az), 0.5 + 0.45 * clamp(r.z, -1.0, 1.0));
        // The paint reads the photo as Omsi.exe's environment stage lays it on: Direct3D's
        // sphere map of the reflection in the camera's frame (`sphere_map_uv`, 0x7ff670 sets
        // D3DTSS_TCI_SPHEREMAP and no texture matrix), so the picture on a body turns with
        // the camera's yaw and pitch as it does in OMSI. (A pane keeps the world's: through
        // it the street itself is seen, and the photo turning over it read as a smear.)
        if (!glass) {
            env_uv = in.env_uv;
        }
        if (material.bump.y > 0.5) {
            env_uv = env_uv + bump_offset(duv);
        }
        // (with the mip level its footprint needs on top of the blur: a fixed level showed
        // the photo's trees as contour lines across a car's curved bonnet close up, and
        // paint takes it as blurred as the vanilla path does)
        let photo = textureSampleBias(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], env_uv, max(rough * 16.0, select(2.0, 0.0, glass))).rgb;
        let photo_avg = textureSampleLevel(mx_tex2d[mx_mat.env], mx_samplers[mx_mat.sampler], vec2<f32>(0.5, 0.5), 12.0).rgb;
        let lum_avg = max(dot(photo_avg, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.02);
        // (a tint, not a picture: an unbounded ratio drew the photo's trees as bands)
        // (the photo's own contrast, as the vanilla picture shows it, held only off black
        // and off a blinding white; clamped to 0.35 .. 2 and laid on at two thirds, the
        // windows mirrored a flat grey where the vanilla ones mirror trees and sky)
        let ratio = clamp(photo / lum_avg, vec3<f32>(0.12), vec3<f32>(select(3.5, 1.0, glass && probe_only)));
        let band = 1.0 - smoothstep(0.25, 0.7, abs(r.z));
        // Only a smooth surface mirrors the photo; a satin one a quarter of its contrast.
        // (A car's bonnet is a handful of triangles: the photo's mip level by its footprint
        // above keeps its trees from turning into contour lines across it close up.)
        let sharpness = 1.0 - smoothstep(0.05, 0.25, rough);
        // ... and only as far as the reflection is of the outside at all: the driver's own
        // windscreen reflects the cab, and the photo's trees drawn over that stood as
        // contour lines across everything seen through it (a car's bonnet close up)
        let outside_env = 1.0 - cab_view;
        // ... and not in a fog: there is nothing sharp out there to mirror. The photo's
        // trees stood as flat grey silhouettes on every pane in fog, in front of the real
        // (fogged) trees behind the glass - outlines of trees that are not there
        let clear_air = exp(-enh.fog.x * 150.0);
        env = env * mix(vec3<f32>(1.0), ratio, band * mix(0.25, 1.0, sharpness) * outside_env * clear_air * enh.debug.z);
    }
    // what the SSAO darkens, it also keeps reflections out of
    let spec_occ = clamp(pow(nv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    // (a PBR set's roughness or metalness map says how it reflects: the probe, as for an envmap)
    let pbr_reflects = (material.pbr.z > 0.5 || material.pbr.w > 0.5) && !terrain;
    // (a wet road mirrors the sky probe as well; and what reflects nothing keeps the light
    // the Fresnel term took off its ambient above - at a grazing angle that term is near 1,
    // and the far road and ground went dark with no reflection in its place, #374)
    // Every solid surface reflects its few per cent: a dielectric's 4 % at normal incidence
    // rising towards grazing, by its roughness and the light its microfacets pass on (the
    // diffuse part took that share off above). Leaves and fences are left out: alpha-tested
    // silhouettes, whose grazing sheen was white sparkle. A wet road's film reflects at its
    // own share.
    let reflects = !thin;
    var refl_f = spec_e.rgb * spec_occ * horizon * select(1.0, mix(1.0, wet_share, wet_road), wet_only);
    if (!reflects) {
        ambient = e_amb * ambient_albedo / PI;
    }
    if (glass) {
        // Transparent bus panes need a readable outside reflection from the driver's
        // viewpoint; opaque paint must never receive this boost.
        refl_f = refl_f * (1.0 - 0.85 * own_pane);
    }
    refl_f = refl_f * env_share;
    var reflection = select(vec3<f32>(0.0), env * refl_f, reflects);
    // Enhanced+: traced instead (rough surfaces keep the probe, which is as good as a
    // ray there): the windows, the chrome and the paint's clear coat mirror the real street
    // round them - the coat as sharply as the glass, where it is polished (`coat_rough`)
    // (a dry plain surface - plaster, plastic, a satin wall - is traced up to a lesser
    // roughness: its few per cent of a blurred street are as good from the probe, and the rays
    // are spent where the reflection is seen; between the two the traced one fades in)
    let trace_top = select(0.75, 0.5, wet_only && wet_road <= 0.0);
    let trace_k = 1.0 - smoothstep(trace_top - 0.15, trace_top, refl_rough);
    // (not the player's own panes seen from inside: their faint reflection of the cab is the
    // glass's own, and the street behind them - a wet road through the windscreen - keeps
    // its traced one, see below)
    let traced = camera.clouds.w > 1.5 && reflects && trace_k > 0.0 && !capture && !(glass && own_pane > 0.5);
    if (traced) {
        var w = dot(refl_f, vec3<f32>(0.2126, 0.7152, 0.0722)) * aer.a * trace_k;
        if (glass) {
            // A pane mirrors what the rays find as glass does: 4 % at the normal, Fresnel's
            // rise towards grazing (a tinted one a little more against what little comes
            // through it). The share above is the vanilla picture's, which lays a dim photo
            // over the pane; the traced street at that share - a fifth to a half - turned
            // every window into a mirror of the bright sky.
            let f0_glass = vec3<f32>(mix(0.04, 0.07, clamp(alpha * 1.5, 0.0, 1.0)));
            w = dot(spec_ms(f0_glass, rough, nv).rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) * spec_occ * horizon * env_share * aer.a * trace_k;
            w = w * smoothstep(0.0, 0.05, alpha);
        } else if (mode > 1.5) {
            // (a blended layer reflects only where it is there: a painted ground's brush mask)
            w = w * clamp(alpha, 0.0, 1.0);
        }
        let rough_t = refl_rough;
        if (w > 0.002) {
            reflection = reflection * (1.0 - trace_k);
            rt_gbuf = vec4<f32>(n * w, w);
            rt_aux = vec4<f32>(dist * w, rough_t * w, 0.0, w);
            if (glass) {
                // A pane takes the pixel's surface over (coverage 1, its weight in the
                // normal's length, rt_reflect.wgsl): blended over the seats behind it by its
                // own few per cent, the saloon's surfaces had weighed nine tenths - the rays
                // left from a point between the pane and the seats along a normal halfway
                // between theirs, and the window wore white blotches of the sky with the
                // seats' outlines.
                rt_gbuf = vec4<f32>(n * w, 1.0);
                rt_aux = vec4<f32>(dist, rough_t, 0.0, 1.0);
            }
        }
    }
    // --- the lamps, the cabin light and what glows by itself
    // ([nomaplighting] objects are not lit by the map's lamps; light-mapped roads are, with
    // the tile light map on top)
    // (no shadows inside the player's own vehicle, whose cab the depth hardly shows, nor in
    // the probe's capture)
    let film = select(vec3<f32>(0.0), vec3<f32>(wet_road, refl_rough, wet_share), wet_only && wet_road > 0.0);
    // A clear pane scatters nothing: what lights it shows only in what it mirrors (the
    // lamps' glints, the sky, the street) and in the little its tint takes from the daylight
    // through it. Lit as a grey Lambertian layer, every lamp and the cab's own lamps laid a
    // milky veil over the windows at night, over everything seen through them.
    var sf_lit = sf;
    if (glass) {
        sf_lit.albedo = vec3<f32>(0.0);
    }
    if ((terrain || material.params2.z > 0.0) && !is_water) {
        sf_lit.f0 = sf_lit.f0 * camera.lod.z;
    }
    let lamps = lamp_light(in.world, n, v, sf_lit, thin, !capture, film, dot(geo_n, v) < 0.0 && !is_water) * select(1.0, 0.0, material.params.y > 0.2 && material.params.y < 0.3);
    // [interiorlight]: OMSI adds its lamps' light to the lit meshes whatever the daylight,
    // so a switched-on saloon is brighter by day as well and only stands out more at night.
    // Taken as a lamp against the daylight exposure it vanished by day altogether.
    // (in the picture's own terms, as the vanilla pass adds it: scaled with the metering it
    // grew with the night and whitened the saloon)
    // (the lamps' light does not reach into the gaps under the seats and round the
    // handrails either: without the ambient occlusion on it they glowed through there)
    // A sphere-mapped pane or body mirrors the street round it: at night that is lit by the
    // lamps, not by the sky the probe holds (dark then). The street's radiance under them
    // (`lamp_ground`) is laid into the reflection where it points at the street and the
    // walls: a bus's windows and paint under a street lamp show the lit street, the ones on
    // an unlit road stay dark. (Not through the photo's structure: its bright rows are the
    // day's sky, which the lamps do not light.)
    if ((reflective_env || glass) && !traced) {
        let lit_round = lamp_ground * (1.0 - open);
        reflection = reflection + lit_round * refl_f * (1.0 - cab_view);
    }
    // (in the player's saloon: what its lamps' light meets on the way, see `cabin_visibility`)
    var cabin_vis = 1.0;
    if (in_cab > 0.5 && camera.lamp_atlas.w > 0.5) {
        cabin_vis = cabin_visibility(in.world, n);
    }
    let cabin_light = interior_lamps(in.world, n, in.params2.z) * cabin_vis;
    let cabin = sf_lit.albedo * cabin_light * mix(1.0, ao, 0.85);
    var rgb = (direct + ambient + lamps) * pre + cabin;
    // A material's own emissive colour ([matl_allcolor], an .x's emissive) is the texture at
    // full brightness in Omsi.exe: shown at the screen's white, not scaled with the eye's
    // night adaptation - a texture lit that way by [matl_allcolor] glared at several times
    // white at night (#1228, #1236). (The night maps of lit windows keep their light.)
    // (not a pane's: glass gives off no light of its own, see `sf_lit`)
    var emit = select(tex.rgb * material.emissive.rgb * clamp(enh.exposure.z * 2.0, 0.8, 1.0), vec3<f32>(0.0), glass);
    // (the tile light map on the splines and [LightMapMapping] objects is the vanilla
    // path's: here the map's lamps light them, tinted from that map, as they light every
    // other surface - added on top it lit the roads twice, with a hard edge where a road
    // met a square that is an object)
    if (material.extra.w > 0.5) {
        let switched = material.extra.w > 1.5;
        let night = select(rain_lit_windows(in.local), 1.0, switched);
        // (a switched one is the display's own state: not dimmed with the instance's night
        // lighting, which is 0 by day and left the Procity's pressure screen black)
        let nm = night_sample.rgb * night * select(clamp(in.params2.y, 0.0, 1.0), 1.0, switched);
        // (The ground gets no light from the tile's light map here: it is the lamps' pools as
        // OMSI bakes them, a `[maplight]`'s colour times (radius / distance)^2 out to sixteen
        // radii, taken through every wall and roof, with no shadow of anything and with a
        // tail past where the lamp itself reaches. The lamps are lights of their own in this
        // picture - the same inverse square, cut off where their reach ends, shadowed by the
        // maps of `LampShadow` - and what the map added three times over was a snow-covered
        // lawn in a courtyard lit orange by floodlights across a building, or the ground of a
        // whole street glowing with one lamp's tail.)
        if (!terrain) {
            // lit windows and signs; a switched lamp or display holds up against daylight
            // as the self-illuminated materials do
            // (a window lit from inside shows some 80 cd/mÂ² - a room's 300 lux off its
            // walls and through a curtain - an illuminated sign more: over the dark street
            // round it, a little over the screen's white, so that it keeps its colour)
            emit = emit + nm * select(min(enh.exposure.z * 3.6, 1.2), mix(max(enh.exposure.z * 2.0, 0.8), 0.8, near_player_vehicle(in.world) * inside_vehicle(camera.cam_pos.xyz) * (1.0 - outside)), switched);
        }
    }
    if (material.params2.x > 0.5 && !terrain && !glass) {
        // [matl_lightmap]: laid onto the light with ADDSMOOTH in Omsi.exe (see the vanilla
        // shader), so it adds what the surface's own light leaves: little by day, fully at
        // night, and less where the saloon lamps light the surface already. (As an emission
        // at 0.6 of the texture by day and on top of the lamps at night, a switched-on
        // cabin's light-mapped parts were flat white.)
        let lm = textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], buv, camera.lod.x).rgb * clamp(in.params2.x, 0.0, 1.0);
        if (material.emissive.w < -1.5) {
            let night = clamp(camera.sun_color.w, 0.0, 1.0);
            let left = (vec3<f32>(1.0) - clamp(cabin_light, vec3<f32>(0.0), vec3<f32>(1.0))) * (0.12 + 0.88 * night);
            emit = emit + tex.rgb * lm * left * max(enh.exposure.z * 2.0, 0.6);
        } else {
            // (By day as by night: ADDSMOOTH lifts the surface to its texture's colour by the
            // map whatever light it has, and what the light already gives - the day's, the
            // saloon lamps' - is taken off below. Lifted by an eighth of that by day, a
            // display in the cab's shade was far darker than in Vanilla+, #2051.)
            // ... and as there it lifts the surface to its texture's own colour at the
            // most, shown as a display is (see the unlit path): taken as a light of the
            // night's level on top, a white light map (`weiss.bmp`, laid on a display so
            // it shows at night) went past the tone curve's knee and bleached its colours -
            // the Procity's red and blue gauges pink and lavender (#827). (An LED panel's
            // light map stays as it was: its dots are meant to burn above their colour.)
            // The map's script variable controls its backlight. Dimming every light-mapped
            // surface as a cab screen hid instrument backlights even with the lights on.
            // (the lift is the saloon lamps' light: where their light does not reach for a
            // handrail or a seat standing in it, the map does not lift the surface either)
            emit = emit + max(display_level(tex.rgb) * enh.exposure.y - rgb, vec3<f32>(0.0)) * lm * cabin_vis;
        }
    }
    if (material.emissive.w < -1.5) {
        // an LED panel (see MaterialExtra::led): the lit dots - the alpha the `\S:n` script
        // texture carries, in the colour of the panel's own texture - are the panel's own
        // light, drawn as bright as the settings ask for (`Led glow`, 16 levels, 0 = off).
        // The glow takes them where it leaves every other screen out of its source
        // (`post.wgsl`) and blooms a halo around the panel. (Kept at their own brightness
        // however the metering treats the scene, as a display's text is.) Its light is its
        // white light map's, so it goes out with that map's variable (the busbar, the
        // lights) as the Omsi.exe stage does.
        let lm_gate = select(1.0, clamp(in.params2.x, 0.0, 1.0), material.params2.x > 0.5);
        // (in the cab - a dashboard's LCD lit by a white map reads as a panel - it dims at night;
        // not a page's destination sign (-3), which is read from the street through the windscreen)
        let cab_dim = select(1.0 - outside, 0.0, material.emissive.w < -2.5);
        // (`material.glow.y`: the share of it the modeller gives the panel, `[led_glow_effect]`)
        emit = emit + tex.rgb * enh.led.x * material.glow.y * alpha * lm_gate * max(enh.exposure.z * 2.0, 0.8) * mix(1.0, display_dim(1.0), cab_dim);
    } else if (material.emissive.w < -0.5) {
        // a display's text (see MaterialExtra::display)
        emit = emit + tex.rgb * 0.35 * max(enh.exposure.z * 2.0, 0.8) * display_dim(1.0);
    }
    if (material.glow.x > 0.0) {
        // [matl_glow] (see `matl_glow_light`): the material is its own light, drawn in HDR.
        // The classic picture does not know the keyword (`params2.x` is off, see
        // `add_material_extra`).
        emit = emit + matl_glow_light(buv, tex.rgb);
    }
    rgb = rgb + emit;
    if (enh.debug.x > 0.5) {
        // OMSI_DEBUG_ENHANCED: one term alone, as it lands on the screen
        let dm = i32(enh.debug.x);
        var dc = vec3<f32>(0.0);
        switch dm {
            case 1: {
                let tr = surface_rt;
                let sm = sun_shadow_soft(in.world, n, thin);
                dc = select(vec3<f32>(sm), vec3<f32>(tr.z * sm, tr.z * sm, tr.z * 0.6 + 0.4), tr.w > 0.5 && tr.z >= 0.0);
            }
            case 2: { dc = vec3<f32>(ao); }
            case 3: { dc = n * 0.5 + vec3<f32>(0.5); }
            case 4: { dc = vec3<f32>(aer.a); }
            case 5: { dc = e_amb * pre * 0.1; }
            case 6: { dc = reflection * pre; }
            case 7: { dc = sf.albedo; }
            case 8: { dc = direct * pre; }
            case 9: { dc = aer.rgb * pre; }
            case 11: { dc = vec3<f32>(fract(dist / 10.0), dist / 1000.0, in.clip.z * 100.0); }
            case 12: { dc = vec3<f32>(mode * 0.5, f32(terrain), in.params2.w); }
            case 13: { dc = vec3<f32>(in_cab, ao, spec_occ); }
            case 14: { dc = (ambient + direct) * pre; }
            case 15: { dc = lamps * pre; }
            case 16: { dc = emit; }
            case 17: { dc = vec3<f32>(rough, f0.g * 10.0, metal); }
            default: { dc = vec3<f32>(alpha, f32(glass), f32(has_env)); }
        }
        return vec4<f32>(dc, 1.0);
    }
    if (glass) {
        // see-through glass: the reflection is added on top of what shows through, so
        // the blend keeps it where the glass itself is faint - but not where the texture
        // is not there at all: Omsi.exe leaves the alpha as the texture has it, so a
        // see-through part of a blended layer (the clear ground of a sticker on the
        // cab's wall, #861) shows no reflection; made up to a quarter opaque by it, the
        // whole rectangle of the sticker mirrored the sky
        let cover = smoothstep(0.0, 0.05, alpha);
        // (or, from inside the player's bus, the lit cabin it mirrors: `glass_reflection`)
        let refl_rgb = glass_reflection(in.world, n, v, reflection * pre) * cover;
        let rl = dot(refl_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        // (a pane passes most light: a reflection making it up to 60 % opaque laid a grey
        // veil over the destination display behind the windscreen and the saloon)
        var a2 = clamp(alpha + (1.0 - alpha) * clamp(rl * 0.25 + fr.g, 0.0, 0.25) * (1.0 - 0.85 * own_pane) * cover, alpha, 1.0);
        var c = (rgb * alpha + refl_rgb) / max(a2, 1e-3);
        // the rain on a pane without a film of its own (`pane_rain`): the traffic's windows
        {
            let pr = pane_rain(in.world, in.local, in.normal);
            if ((pr.cover > 0.001 || pr.veil > 0.001 || pr.mist > 0.001) && cover > 0.0) {
                let o = pane_rain_compose(pr, c, a2, v, rain_env_enhanced(reflect(-v, pr.n), 1.0) * pre, sh_irradiance(pr.out) / PI * 0.9 * pre);
                c = o.rgb;
                a2 = o.a;
            }
        }
        // the condensation on the player's bus's panes (omsi-app condensation.rs): a film of
        // droplets, each many wavelengths wide, takes twice its shadow out of the light
        // (the extinction paradox): half of what it scatters is diffracted a degree or two
        // forward - what lies behind goes soft, in the tone mapping (post.wgsl `misted`) -
        // and half is refracted and reflected every way, a milky veil lit by the street
        // through the glass and, the part the droplets send back, by the cabin
        let fog = glass_condensation(in.world, n) * cover;
        glass_fog = fog;
        if (fog > 0.001) {
            // (of the widely scattered half, some 30 % goes back: a fully misted pane sends
            // some 15 % of the light back, Briscoe and Galvin 1991)
            let back = 0.3;
            let lit_street = avg_e / PI * pre;
            let lit_cabin = cab_e * CAB_AMBIENT / PI * pre + cabin_light;
            let milky = lit_street * (1.0 - back) + lit_cabin * back;
            let veil = 0.5 * fog;
            let a3 = a2 + (1.0 - a2) * veil;
            c = (c * a2 * (1.0 - veil) + milky * veil) / max(a3, 1e-3);
            a2 = a3;
        }
        return vec4<f32>(c * aer.a + aer.rgb * pre, a2);
    }
    rgb = rgb + reflection * pre;
    // The later screen-space pass replaces only this fraction of the sky reflection.
    // A miss adds zero, so the existing sky, Fresnel and fog stay the fallback. Colour
    // and coverage are resolved together, including terrain painting and glass over roads.
    if (!glass && !reflective_env) {
        // (a pool mirrors by its Fresnel term and its share of the road; so does the sheen of a wet
        // road, at the Fresnel term of water a little under full - the lamps' and the cars'
        // streaks down a wet street at night, which come from the light's own brightness and so
        // need no more than that share - and still water as a whole)
        let fresnel = 0.02 + 0.98 * pow(1.0 - clamp(nv, 0.0, 1.0), 5.0);
        let pool = puddle * env_brdf(f0, rough, nv).g * select(wet_road, 1.0, pbr_reflects) * wet_share * aer.a;
        let sheen = select(0.8 * smoothstep(0.55, 0.95, wet_road), 1.0, is_water) * fresnel * aer.a;
        let weight = clamp(max(pool, sheen), 0.0, 1.0);
        *puddle_weight = vec2<f32>(weight, weight * spec_occ);
    }
    // water hides what lies under it as far as it reflects (Fresnel): see-through from
    // above, a mirror of the sky towards the horizon
    let a_out = select(alpha, clamp(alpha + (1.0 - alpha) * fr.g, alpha, 1.0), is_water);
    return vec4<f32>(rgb * aer.a + aer.rgb * pre, a_out);
}

// Small waves on the water at map point `p` (m) and time `t` (s): the slope of four wave
// trains of 1.3 to 4.7 m running different ways.
fn water_ripple(p: vec2<f32>, t: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    let dirs = array<vec2<f32>, 4>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.45, 0.89), vec2<f32>(0.96, -0.28), vec2<f32>(-0.7, -0.71));
    let lens = array<f32, 4>(4.7, 2.9, 1.9, 1.3);
    for (var i = 0; i < 4; i = i + 1) {
        let k = 6.2831853 / lens[i];
        // deep-water waves: their speed goes with the square root of their length
        let w = sqrt(9.81 * k);
        g = g + dirs[i] * cos(dot(p, dirs[i]) * k - w * t) * 0.045;
    }
    return g;
}

// Shade the complete local vehicle from the reflected eye. Main-camera ambient
// occlusion is not applicable to this view beneath the road.
@fragment
fn fs_puddle_vehicle(input: FsIn) -> @location(0) vec4<f32> {
    // One reflected camera for the complete vehicle, including its transparent panes.
    let plane = vehicle_reflection.plane;
    let height = dot(plane.xyz, input.world) - plane.w;
    if (height < 0.0 || material.emissive.w > 1.5) { discard; }
    var unused = vec2<f32>(0.0);
    let eye = camera.cam_pos.xyz - 2.0 * plane.xyz * (dot(plane.xyz, camera.cam_pos.xyz) - plane.w);
    if (camera.post.x < 0.5) {
        var unused_vanilla = 0.0;
        return shade_vanilla(input, &unused_vanilla, eye);
    }
    return shade_enhanced(input, &unused, true, eye);
}

@fragment
fn fs_puddle_chassis() -> @location(0) vec4<f32> {
    if (camera.post.x < 0.5) { return vec4<f32>(camera.ambient.rgb * 0.025, 1.0); }
    return vec4<f32>(sh_irradiance(vec3<f32>(0.0, 0.0, -1.0)) * enh.exposure.x * 0.025, 1.0);
}

// The painted ground's albedo (`Renderer::encode_ground`): the terrain's base and its paint
// layers laid over each other into one picture of what the surface is made of - texture,
// detail and tint, blended by each layer's coverage - which the main pass then lights once
// (`from_ground` in `shade_enhanced`) instead of lighting every layer in full and blending
// the lit colours. The base writes it, the layers blend into it (the pipeline's blend: the
// coverage over what is there; the alpha channel is left as the base wrote it).
@fragment
fn fs_ground(in: FsIn) -> @location(0) vec4<f32> {
    let duv = in.uv * material.extra.z;
    let buv = tex_address(in.uv - in.params.zw);
    let mode = material.params.x;
    // (a brush mask that covers nothing here: the layer's texture need not be read)
    if (TERRAIN_PAINT && mode > 1.5 && material.params.z > 0.5 && material.params.w > 0.5) {
        let cover = smoothstep(0.32, 0.68, sample_transmap(buv).a);
        if (cover * material.color.a * in.params.x == 0.0) {
            discard;
        }
    }
    var tex = diffuse_border(textureSampleBias(mx_tex2d[mx_mat.diffuse], mx_samplers[mx_mat.sampler], duv, camera.lod.x), duv);
    if (material.extra.y > 0.0) {
        let det = textureSampleBias(mx_tex2d[mx_mat.light], mx_samplers[mx_mat.sampler], in.uv * material.extra.y, camera.lod.x);
        tex = vec4<f32>(clamp(tex.rgb * det.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), tex.a);
    }
    var cover = tex.a;
    if (material.params.z > 0.5) {
        cover = select(1.0, sample_transmap(buv).a, material.params.w > 0.5);
        if (mode > 1.5) {
            cover = smoothstep(0.32, 0.68, cover);
        }
    }
    var alpha = cover * material.color.a;
    if (mode < 0.5) {
        alpha = 1.0;
    }
    alpha = alpha * clamp(in.params.x, 0.0, 1.0);
    if (TERRAIN_PAINT && alpha == 0.0) {
        discard;
    }
    return vec4<f32>(tex.rgb * material.color.rgb, alpha);
}

// The bus's windows in the enhanced picture: the rain on their outer face, the snow that
// settles there and the condensation on their inner face. (The drops and the snow are the classic pictures' too: scene/08_rain_glass.wgsl.)

// The pane at a pixel, in the vehicle's own frame (the film's vertices as its mesh has them,
// `FsIn.local`): the drops are laid out there, so they stay on the glass however the body
// rolls and pitches and on a door that swings open. The modellers' texture coordinates of
// a film are no measure of the glass: on the windscreen of the NL202 they run half as fast
// again one way as the other and at sixty degrees to each other, from triangle to triangle
// anew, and every drop drawn in them came out a slanting streak. So a pane is laid flat by
// the axes of the vehicle (`pane_proj`): the windscreen and the rear window onto its
// across and up, the side windows onto its along and up, a roof light onto its plan. A
// raked or curved pane is foreshortened in that projection; a drop is drawn round all the
// same, its offsets taken back onto the glass (`m`) before its shape is worked out. The
// drops' grid alone stays in the projection: on a pane at an angle to it a cell covers
// more glass, and the drops are a little fewer there.

// (The drops on a pane, the runners, the blur's picture and the pane's own frame -
// `glass_water`, `glass_blur` - are the classic pictures' too: scene/08_rain_glass.wgsl.)

// The rain film on a pane (the enhanced pass's `emissive.w` 2 layer), drawn over the
// clean picture of the frame: the drops, each a small lens that shows the street behind
// it upside down, a shade darker at its rim where the light is bent away, with the sky's
// glint on it, and between them the soft blur of the wet glass.
fn glass_film(in: FsIn, water: f32) -> vec4<f32> {
    let v = camera.cam_pos.xyz - in.world;
    let dist = length(v);
    let vn = v / max(dist, 1e-4);
    let in_cab = inside_vehicle(camera.cam_pos.xyz) * near_player_vehicle(in.world) > 0.5;
    let g = glass_water(in.world, in.local, in.normal, water, camera.post.y, in_cab);
    if (g.cover <= 0.001 && g.mist <= 0.001 && g.veil <= 0.001) {
        return vec4<f32>(0.0);
    }
    let pre = enh.exposure.x;
    let has_picture = camera.flags.z < -0.5;
    let dims = vec2<f32>(textureDimensions(mx_tex2d[mx_mat.env]));
    let uv0 = in.clip.xy / dims;
    let mist_col = sh_irradiance(g.out) / PI * 0.9 * pre;
    // the view through the wet glass between the drops: softened by the film of water
    // (a running film scatters the light by a fraction of a degree: some 0.7 % of the
    // picture's height in a downpour, a third of that in a drizzle)
    let veil_lod = log2(max(dims.y * mix(0.002, 0.006, clamp(g.veil / 0.3, 0.0, 1.0)), 1.0));
    var behind = mist_col;
    if (has_picture) {
        behind = finite_or(glass_blur(uv0, veil_lod), mist_col);
    }
    // --- a drop: a small lens. Through it the street shows small and upside down (each
    // point of the drop looks out along its own bent way, `tilt` across the screen), the
    // more bent the further out on it; at its rim the light is bent so far that little of
    // it reaches the eye, a darker ring.
    var drop_col = behind;
    var a_drop = 0.0;
    if (g.cover > 0.001) {
        let tilt = g.n - g.out * dot(g.n, g.out);
        let right = camera.cam_right.xyz;
        let up = camera.cam_up.xyz;
        // (a drop is a wide lens: from its rim it looks out some 30 degrees aside, the sky
        // in the bottom of a drop against a wall)
        let o = vec2<f32>(dot(tilt, right), -dot(tilt, up)) * 0.3;
        var seen = behind;
        if (has_picture) {
            var uv = clamp(uv0 - o, vec2<f32>(0.001), vec2<f32>(0.999));
            // (only what lies beyond the glass: where the bent way lands on the cab's own
            // dashboard or pillar, in front of the pane, the drop shows what is straight
            // behind it instead)
            if (camera.clouds.w > 0.5 && camera.clouds.w < 1.5) {
                let asz = vec2<f32>(textureDimensions(t_ao));
                let z = textureLoad(t_ao, min(vec2<i32>(uv * asz), vec2<i32>(asz) - vec2<i32>(1)), 0).g;
                let zg = (camera.view_proj * vec4<f32>(in.world, 1.0)).w;
                let beyond = select(smoothstep(zg + 0.05, zg + 0.4, z), 1.0, z <= 0.0);
                uv = mix(uv0, uv, beyond);
            }
            seen = finite_or(glass_blur(uv, 0.8 + 1.5 * g.mist), behind);
        }
        let rim = smoothstep(0.55, 1.0, g.rim);
        let facing = dot(g.out, vn) > 0.0;
        let cosv = clamp(abs(dot(g.n, vn)), 0.0, 1.0);
        let f = select(0.02, 0.02 + 0.98 * pow(1.0 - cosv, 5.0), facing);
        let mirrored = rain_env_enhanced(reflect(-vn, g.n), 1.0) * pre;
        var c = seen * (1.0 - 0.45 * rim) * (1.0 - f) + mirrored * f;
        // the sun: a pinpoint where the dome mirrors it
        if (facing) {
            let r = reflect(-vn, g.n);
            c = c + enh.sun.rgb / PI * pre * pow(max(dot(r, camera.sun_dir.xyz), 0.0), 900.0) * f * 60.0;
        }
        // A drop of a pixel or less is no lens one can see: its many ways average to the
        // view behind it, a little dimmer for its rim - a faint grain, not a dark speck.
        let detail = smoothstep(0.5, 1.6, g.size_px);
        drop_col = mix(behind * 0.93, c, detail);
        a_drop = g.cover * mix(select(0.15, 0.5, has_picture), 0.95, detail);
    }
    // Drops a few pixels across are a pane's sparkle close up; seen from the street they
    // fade out over the first fifteen metres. The haze stays further.
    let aer = air(-vn, fog_distance(in.world), camera.cam_pos.z - enh.fog.z, in.world.z - enh.fog.z);
    a_drop = a_drop * (1.0 - smoothstep(5.0, 15.0, dist));
    // the fine mist of droplets: a faint grey scatter of the light about
    let a_mist = g.mist * 0.5;
    var a_veil = 0.0;
    if (g.veil > 0.001 && has_picture) {
        a_veil = g.veil * (1.0 - smoothstep(25.0, 60.0, dist));
    }
    let a_under = a_veil + a_mist * (1.0 - a_veil);
    let under = (behind * a_veil + mix(mist_col, behind, 0.9) * a_mist * (1.0 - a_veil)) / max(a_under, 1e-4);
    var a = a_drop + a_under * (1.0 - a_drop);
    if (a <= 1e-4) {
        return vec4<f32>(0.0);
    }
    let col = (drop_col * aer.a * a_drop + under * a_under * (1.0 - a_drop)) / a;
    // From inside, the rain is seen through the condensation on the glass's inner face
    // (drawn with the pane before this film): a misted pane shows the drops behind it as
    // soft shadows, not sharp lenses.
    if (in_cab) {
        a = a * (1.0 - 0.5 * glass_condensation(in.world, g.out));
    }
    return vec4<f32>(col, a);
}

// The snow a wiped pane holds while it snows (`is_snow_pane`; the mask's cover: 0 clear
// glass, up to 1 a pane closed by snow, above 1 the bank a blade pushed to the end of its
// sweep, below 0 glass a blade has just cleared). Crystals land one by one and stick, hold
// a while and melt into a little water - soon on the windscreen, whose glass the warm air
// of the defroster heats, slowly on a side window. As the cover grows they close into a
// milky layer, and it lies deepest along the windscreen's bottom edge and in its lower
// corners, where the blades and the airstream do not reach.
fn glass_snow(in: FsIn, cover: f32) -> vec4<f32> {
    // (the windscreen's defroster from the cabin's air, the height on the glass in the bus's
    // own frame with the body's tilt: see `glass_snow_with`, which the classic pictures
    // share)
    let l = pane_local(in.world);
    let fz = clamp(l.z / max(camera.inside_b.w, 0.1), -1.0, 1.0);
    return glass_snow_with(in, cover, enh.condensation.w, fz);
}

// A direction in the frame of the bus the camera is in (`pane_local` without the shift).
fn pane_dir(d: vec3<f32>) -> vec3<f32> {
    let sh = camera.inside_a.w;
    let ch = camera.inside_b.x;
    let v = vec3<f32>(d.x * ch - d.y * sh, d.x * sh + d.y * ch, d.z);
    let t = enh.pane_tilt;
    let p = vec3<f32>(v.x, v.y * t.y + v.z * t.x, v.z * t.y - v.y * t.x);
    return vec3<f32>(p.x * t.w - p.z * t.z, p.y, p.z * t.w + p.x * t.z);
}

// How close the edge of the pane at `world` (normal `n`) is: 1 against it, falling to 0
// some 25 cm in; -1 where it cannot be told. A pane's edge is where its frame begins - the
// rubber, the pillar, the trim, the dashboard under the windscreen: opaque and in the depth
// prepass, lying on the glass's own plane, where the glass is not. So points on the pane's
// plane round the pixel, in eight ways out to 26 cm, are looked up in the prepass's depth
// (the ambient occlusion's texture keeps it): where an opaque surface lies there on the
// plane - within a centimetre or two of it, not a wiper's arm a hand's breadth outside - the
// pane has ended. The nearest such end in each way, averaged over the ways: in a corner
// two ways find it close, along an edge one, in the middle of a pane none.
fn glass_frame_near(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (camera.clouds.w < 0.5 || camera.clouds.w > 1.5) {
        return -1.0;
    }
    let size = vec2<f32>(textureDimensions(t_ao));
    let nn = safe_normal(n);
    let up0 = vec3<f32>(0.0, 0.0, 1.0) - nn * nn.z;
    let a = select(vec3<f32>(1.0, 0.0, 0.0), normalize(up0), dot(up0, up0) > 0.04);
    let b = cross(nn, a);
    var sum = 0.0;
    for (var i = 0; i < 8; i = i + 1) {
        let ang = f32(i) * 0.7853982 + 0.2;
        let dir = a * cos(ang) + b * sin(ang);
        var near = 0.0;
        for (var k = 0; k < 6; k = k + 1) {
            let dist = 0.03 + 0.046 * f32(k);
            let c = camera.view_proj * vec4<f32>(world + dir * dist, 1.0);
            if (c.w <= 0.05) {
                break;
            }
            let uv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
            if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
                break;
            }
            let z = textureLoad(t_ao, min(vec2<i32>(uv * size), vec2<i32>(size) - vec2<i32>(1)), 0).g;
            if (z > 0.0 && abs(z - c.w) < 0.015 + 0.006 * c.w) {
                near = 1.0 - (dist - 0.03) / 0.26;
                break;
            }
        }
        sum = sum + near;
    }
    return clamp(sum / 3.0, 0.0, 1.0);
}

// How much of the light crossing the player's bus's pane at `world` (normal `n`) its
// condensation scatters (0..1). One cabin air mists all the glass much alike: each
// window's film (enh.condensation: the optical depths of the windscreen's, the side
// windows' and the rear window's, omsi-weather condensation.rs) is taken two thirds from
// their mean. The film starts where the glass is coldest and the air stillest, in a pane's
// corners and along its frame (and low down), and grows inward as it thickens, its front
// wandering with the glass's own patchiness, until it closes over the pane. A heavy film
// gathers into drops that trickle down and leave clear tracks; on the windscreen the
// defroster's warm air clears it from the bottom up.
fn glass_condensation(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (camera.inside_c.w < 0.5) {
        return 0.0;
    }
    let l = pane_local(world);
    let h = camera.inside_b.yzw;
    if (any(abs(l) >= h + vec3<f32>(0.25))) {
        return 0.0;
    }
    let nl = safe_normal(pane_dir(n));
    let gap = h - abs(l);
    // (the panes in the body's skin only, by the way they face: not the instruments' glass
    // behind the windscreen or a partition in the saloon)
    var own = -1.0;
    var front = false;
    if (abs(nl.y) > 0.55 && gap.y < 0.45) {
        front = l.y > 0.0;
        own = select(enh.condensation.z, enh.condensation.x, front);
    } else if (abs(nl.x) > 0.5 && gap.x < 0.35) {
        own = enh.condensation.y;
    }
    if (own < 0.0) {
        return 0.0;
    }
    let depth = mix((enh.condensation.x + enh.condensation.y + enh.condensation.z) / 3.0, own, 0.35);
    if (depth <= 0.001) {
        return 0.0;
    }
    // the pane's own frame in metres: across, and up the glass
    let up0 = vec3<f32>(0.0, 0.0, 1.0) - nl * nl.z;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), normalize(up0), dot(up0, up0) > 0.04);
    let s = vec2<f32>(dot(l, cross(up, nl)), dot(l, up));
    let fz = clamp(l.z / max(h.z, 0.1), -1.0, 1.0);
    let low = 1.0 - clamp(fz * 0.5 + 0.5, 0.0, 1.0);
    let edge_seen = glass_frame_near(world, n);
    let edge = select(edge_seen, 0.15 + 0.3 * low, edge_seen < 0.0);
    let n1 = snow_noise(s * 3.0 + vec2<f32>(11.0, 3.0));
    let n2 = snow_noise(s * 9.0 + vec2<f32>(3.0, 17.0));
    let patchy = n1 * 0.7 + n2 * 0.3;
    // (the front: into the corners and along the frame at first, the glass's patchiness
    // breaking its line, over the whole pane at a depth of two; a soft front, the film
    // thinning out over a few centimetres)
    let grow = 1.0 - exp(-depth * 1.3);
    let potential = 0.45 * edge + 0.15 * low + 0.45 * patchy;
    let front_line = 1.0 - grow * 1.3;
    let covered = smoothstep(front_line - 0.18, front_line + 0.18, potential);
    var amount = covered * (1.0 - exp(-depth * (0.9 + 0.5 * patchy)));
    // the fine grain of the droplets themselves
    amount = amount * (0.92 + 0.08 * snow_noise(s * 140.0));
    if (depth > 1.4) {
        // trickles: drops grown heavy run down a few centimetres a minute, each clearing a
        // narrow track below where it started
        let lane = floor(s.x / 0.03);
        let h1 = rain_hash(vec2<f32>(lane, 71.0));
        if (h1.x < clamp((depth - 1.4) * 0.35, 0.0, 0.6)) {
            let h2 = rain_hash(vec2<f32>(lane, 13.0));
            let x0 = (lane + 0.3 + 0.4 * h2.x) * 0.03 + 0.003 * sin(s.y * 40.0 + h2.y * 6.283185);
            let period = 90.0 + 120.0 * h1.y;
            let tau = fract(camera.post.y / period + h2.y) * period;
            let start = h.z * (0.75 - 0.9 * h2.x);
            let end = start - min(tau * 0.004, 0.6);
            let fade = smoothstep(0.0, 5.0, tau) * (1.0 - smoothstep(period - 10.0, period, tau));
            let dx = abs(s.x - x0);
            let in_track = step(end, s.y) * step(s.y, start) * (1.0 - smoothstep(0.0012, 0.0025, dx));
            let head = 1.0 - smoothstep(0.0018, 0.0028, length(vec2<f32>(dx, s.y - end)));
            amount = amount * (1.0 - 0.8 * max(in_track, head) * fade);
        }
    }
    if (front && enh.condensation.w > 0.02) {
        // the defroster's warm air clears the windscreen from its bottom edge up, first in
        // the middle over its outlets
        let u = clamp((fz + 0.35) / 1.2, 0.0, 1.0);
        let fx = l.x / max(h.x, 0.1);
        let reach = enh.condensation.w / 0.85 * 1.15 - 0.2 * fx * fx + 0.06 * (n1 - 0.5);
        amount = amount * smoothstep(reach - 0.12, reach, u);
    }
    return clamp(amount, 0.0, 1.0);
}

// The reflection a pane adds over what shows through it (pre-exposed `reflection`): the
// street and the sky it mirrors, seen from outside or on another vehicle; none on the inner
// face of the player's own bus's glass. (A box-projected cube of the cabin was mirrored
// there; it lagged behind the camera, shook with the head and showed a blurred picture, and
// was taken out on the user's word, 2026-11-22.)
fn glass_reflection(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, reflection: vec3<f32>) -> vec3<f32> {
    let inside = inside_vehicle(camera.cam_pos.xyz) * near_player_vehicle(world);
    return reflection * (1.0 - step(0.5, inside));
}
