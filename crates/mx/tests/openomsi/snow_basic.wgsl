// What the snowfall (snow.wgsl) and the rainfall (rain.wgsl) share: the camera, a hash of
// a particle's number, and the player's vehicle the weather is kept out of.
struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    // (kept in lockstep with `CameraUniform` in lib.rs, as sky.wgsl's)
    world_origin: vec4<f32>,
    sun_dir: vec4<f32>,
    ambient: vec4<f32>,
    fog: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    light_grid: vec4<f32>,
    sky: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    clouds: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    light_view_proj_far: mat4x4<f32>,
    shadow: vec4<f32>,
    post: vec4<f32>,
    inside_a: vec4<f32>,
    inside_b: vec4<f32>,
    inside_c: vec4<f32>,
    flags: vec4<f32>,
    light_view_proj_close: mat4x4<f32>,
    wind: vec4<f32>,
    lamp_view_proj: array<mat4x4<f32>, 56>,
    lamp_atlas: vec4<f32>,
    tree_wind: vec4<f32>,
    inside2_a: vec4<f32>,
    inside2_b: vec4<f32>,
    inside2_c: vec4<f32>,
    cookies: array<vec4<f32>, 8>,
    cabin_lamps: array<vec4<f32>, 4>,
    // x: the textures' mip bias (FSR 1: negative, see `CameraUniform::lod`)
    lod: vec4<f32>,
    // (the classic picture's headlight, `CameraUniform::spot`: unread here)
    spot: array<vec4<f32>, 4>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

fn hash4(n: u32) -> vec4<f32> {
    var x = vec4<u32>(n, n * 747796405u + 2891336453u, n ^ 0x9e3779b9u, n * 1664525u + 1013904223u);
    x = x * 1664525u + 1013904223u;
    x.x = x.x + x.y * x.w;
    x.y = x.y + x.z * x.x;
    x.z = x.z + x.x * x.y;
    x.w = x.w + x.y * x.z;
    x = x ^ (x >> vec4<u32>(16u));
    x.x = x.x + x.y * x.w;
    x.y = x.y + x.z * x.x;
    x.z = x.z + x.x * x.y;
    return vec4<f32>(x & vec4<u32>(0xffffffu)) / 16777216.0;
}

// The precipitation's roof map (passes/precip_map.rs): what stands over the ground round
// the camera, its depth seen from straight above.
@group(1) @binding(1) var roof_map: texture_depth_2d;
fn roof_depth(t: vec2<i32>) -> f32 { return textureLoad(roof_map, t, 0); }

// Whether something solid stands over `p` (the map's matrix `m`; `on` 0 without a map): a
// bus's or a car's roof, a shelter, a bridge, a house - no drop or flake falls there.
fn under_roof(p: vec3<f32>, m: mat4x4<f32>, on: f32) -> bool {
    let c = m * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    let side = f32(textureDimensions(roof_map).x);
    let t = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999)) * side);
    let top = roof_depth(t);
    // (5 cm under it: the map's depth reaches 300 m)
    return on > 0.5 && all(abs(c.xy) < vec2<f32>(1.0)) && c.z > top + 0.00017;
}

// 1 inside the player's own vehicle (its box, a little wider): no snow falls in the cab
fn in_own_vehicle(p: vec3<f32>) -> bool {
    // (the front and, of an articulated bus, the rear section: it snowed in the back, #1967;
    // without branches - two boxes tested with early returns and short-circuits brought
    // Apple's M4 shader compiler service down, XPC_ERROR_CONNECTION_INTERRUPTED, and the
    // game fell back to no snowfall)
    return any(vec2<bool>(in_box(p, camera.inside_a, camera.inside_b, camera.inside_c), in_box(p, camera.inside2_a, camera.inside2_b, camera.inside2_c)));
}

fn in_box(p: vec3<f32>, a: vec4<f32>, b: vec4<f32>, c: vec4<f32>) -> bool {
    let d = p - a.xyz;
    let sh = a.w;
    let ch = b.x;
    let q = vec3<f32>(d.x * ch - d.y * sh - c.x, d.x * sh + d.y * ch - c.y, d.z - c.z);
    let h = b.yzw + vec3<f32>(0.3);
    return select(false, all(abs(q) < h), c.w >= 0.5);
}

// Enhanced: the sun's far shadow map (the scene's camera group), which keeps every caster -
// a drop or a flake in the shade of a house or under the trees does not catch the sun.
@group(0) @binding(6) var s_shadow: sampler_comparison;
@group(0) @binding(7) var t_shadow_far: texture_depth_2d;

// Enhanced: the procedural flake atlas (precip.rs `FlakeAtlas`): eight cells of snow
// crystals and flakes, R how much of a texel they cover, G how bright it is.
@group(1) @binding(2) var t_flakes: texture_2d<f32>;
@group(1) @binding(3) var s_flakes: sampler;

// How far `p` is over whatever stands uppermost there by the roof map (m): below 0 under a
// roof, and far over everything where there is no map.
fn roof_clearance(p: vec3<f32>, m: mat4x4<f32>, on: f32) -> f32 {
    let c = m * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    let side = f32(textureDimensions(roof_map).x);
    let top = roof_depth(vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999)) * side));
    // (the map looks straight down: its depth goes against the height, m[2].z < 0)
    let h = (c.z - top) / m[2].z;
    return select(1e4, h, on > 0.5 && all(abs(c.xy) < vec2<f32>(1.0)) && top < 0.9999);
}

// The wind's gusts: it swells and slackens over some seconds, each swell running downwind
// over the ground as a wave some 70 m long, so that neighbouring drops and flakes are blown
// together and a gust is seen to sweep through the rain. x: the time's worth of the mean
// wind the gusts have added to a particle's way by `t` (the swell's integral, s), y: the
// swell now (a share of the mean wind).
fn precip_gust(x: vec2<f32>, t: f32, wind: vec2<f32>, strength: f32) -> vec2<f32> {
    let speed = length(wind);
    let along = dot(x, wind / max(speed, 1e-3)) * (6.2831853 / 70.0);
    let across = dot(x, vec2<f32>(-wind.y, wind.x) / max(speed, 1e-3)) * (6.2831853 / 160.0);
    let w1 = 0.42;
    let w2 = 1.17;
    let a1 = 0.38 * strength;
    let a2 = 0.17 * strength;
    let p1 = w1 * t - along + 0.6 * sin(across);
    let p2 = w2 * t - 2.3 * along + 1.7 + across;
    return vec2<f32>(-a1 * cos(p1) / w1 - a2 * cos(p2) / w2, a1 * sin(p1) + a2 * sin(p2));
}

// How much of the sun reaches `p` (Enhanced): one compare in the far cascade, lit beyond it
// and without shadow maps.
fn precip_sun(p: vec3<f32>) -> f32 {
    if (camera.shadow.x < 0.5 || enh.sun.r + enh.sun.g < 1e-4) {
        return 1.0;
    }
    let lp = camera.light_view_proj_far * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || lp.z < 0.0 || lp.z > 1.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(t_shadow_far, s_shadow, uv, lp.z - 0.0004);
}

// A hash of a cell of the plane (0..1 each), its integer coordinates taken modulo 4096 so
// that the world's large figures stay exact.
fn cell_hash(c: vec2<f32>) -> vec4<f32> {
    let w = c - 4096.0 * floor(c / 4096.0);
    let i = vec2<u32>(w);
    return hash4(i.x * 4099u + i.y * 7919u + 17u);
}

// The panels of a far veil of rain or snow (Enhanced): rings of upright panels round the
// camera, twelve a ring, each a strip of four vertices. Beyond the particles' layers the
// drops or flakes are too many and too small to draw one by one, and a heavy fall is seen
// there as what it is, a grey sheet streaked or speckled by them, sweeping by with the
// wind. `c`: the corner (0..1 each way); returns the corner's place (render frame) in xyz
// and its ring's radius in w.
fn veil_corner(panel: u32, c: vec2<f32>) -> vec4<f32> {
    var radii = array<f32, 3>(30.0, 48.0, 78.0);
    let ring = min(panel / 12u, 2u);
    let r = radii[ring];
    // (each panel a little wider than its twelfth, so that the chords close the ring; the
    // rings turned against each other, so their joins do not line up)
    let a = (f32(panel % 12u) + c.x * 1.04 - 0.02 + 0.37 * f32(ring)) * (6.2831853 / 12.0);
    let z = camera.cam_pos.z + mix(-6.0, 26.0 + 0.4 * r, c.y);
    return vec4<f32>(camera.cam_pos.xy + vec2<f32>(cos(a), sin(a)) * r, z, r);
}

// The snowfall, every mode: a field of flakes that lives on the GPU alone - each flake's
// place is worked out here from its number, the time, the wind and the camera, so that a
// heavy fall is a hundred and fifty thousand flakes for the cost of one draw (nothing is
// simulated or sent per flake). The field is the world's, repeating every few metres round
// the camera in three layers - dense close by, where every flake is a few pixels wide,
// thinner further out, where it is a speck; beyond them the snowfall's extinction takes the
// view (Enhanced). Flakes are a few millimetres across (an exponential spread, Gunn and
// Marshall 1958; a light fall single crystals of a millimetre), fall at about a metre a
// second (aggregates at 0.8 D^0.16 m/s, Locatelli and Hobbs 1974), sway as they go and
// drift with the wind; opaque and lit by what lights them - the classic picture's
// envir.cfg light, or the enhanced picture's sky, sun and lamps (`precip_light`).

struct SnowParams {
    // xyz the wind the flakes drift with (m/s), w the time (s)
    wind: vec4<f32>,
    // x how hard it snows (0..1), y the flakes' mean diameter (m), zw how many flakes the
    // first two layers hold (the rest are the third's)
    fall: vec4<f32>,
    // the precipitation's roof map: its matrix, and x 1 when there is one
    roof: mat4x4<f32>,
    roof_on: vec4<f32>,
    // Enhanced: x the instance the veil's panels begin at (the flakes' end), y how many
    // panels there are, z how strong the wind's gusts are (0..1), w the picture's height (px)
    ext: vec4<f32>,
};
@group(1) @binding(0) var<uniform> sp: SnowParams;

struct SnowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // the flake's cover (alpha), and its place (for its light)
    @location(1) cover: f32,
    @location(2) pos: vec3<f32>,
};

// The layers round the camera: half width, bottom and top (m).
const LAYER_H = array<vec3<f32>, 3>(vec3<f32>(4.0, -2.5, 5.5), vec3<f32>(10.0, -3.0, 9.0), vec3<f32>(24.0, -3.0, 16.0));

@vertex
fn vs_snow(@builtin(vertex_index) vid: u32, @builtin(instance_index) k: u32) -> SnowOut {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let c = corners[vid % 6u];
    var out: SnowOut;
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let n0 = u32(sp.fall.z);
    let n1 = u32(sp.fall.w);
    var layer = 2u;
    if (k < n0) {
        layer = 0u;
    } else if (k < n0 + n1) {
        layer = 1u;
    }
    let lh = LAYER_H[layer];
    let size = vec3<f32>(2.0 * lh.x, 2.0 * lh.x, lh.z - lh.y);
    let h1 = hash4(k * 2u + 1u);
    let h2 = hash4(k * 2u + 7919u);
    // the flake: its diameter, fall speed and sway
    let r = sp.fall.x;
    let d = min(0.0005 - sp.fall.y * log(max(1.0 - h1.w, 1e-6)), 0.015);
    let fall = max(0.8 * pow(d * 1000.0, 0.16) * (0.75 + 0.25 * r) * (0.85 + 0.3 * h2.x), 0.3);
    let t = sp.wind.w;
    let pace = 1.5 + 2.5 * h2.y;
    let sway = 0.05 + 0.15 * h2.z;
    let ph = h2.w * 6.2831853 + pace * t;
    // its place in the world: where it started, carried by the wind, swaying round, falling;
    // then the copy of the field nearest the camera (the field's figures in the world's
    // own, modulo a kilometre, so that it stays put while the render origin moves)
    let world_cam = camera.cam_pos.xyz + vec3<f32>(camera.world_origin.xy, 0.0);
    let start = h1.xyz * size;
    let drift = sp.wind.xyz * (0.6 * t) + vec3<f32>(cos(ph), sin(ph), 0.0) * sway - vec3<f32>(0.0, 0.0, fall * t);
    let lo = world_cam + vec3<f32>(-lh.x, -lh.x, lh.y);
    let q = lo + (((start + drift - lo) % size) + size) % size;
    let p = q - vec3<f32>(camera.world_origin.xy, 0.0);
    if (in_own_vehicle(p) || under_roof(p, sp.roof, sp.roof_on.x)) {
        return out;
    }
    // a denser layer fades out towards its edges, into the thinner one round it
    let rel = q - lo;
    let soft = lh.x * 0.3;
    var fade = 1.0;
    if (layer < 2u) {
        let e = min(rel, size - rel) / soft;
        fade = clamp(min(min(e.x, e.y), e.z), 0.0, 1.0);
    }
    let to_cam = camera.cam_pos.xyz - p;
    let dist = length(to_cam);
    let vdir = to_cam / max(dist, 0.001);
    // a flake close by is out of focus (its blur as wide as the pupil, some 2.5 mm), one far
    // off at least a pixel and a half wide; as bright as it is seen - a small point's
    // brightness goes with the square root of its light (Stevens' power law)
    let rad = d * 0.5;
    let rs = max(sqrt(rad * rad + 0.0025 * 0.0025), dist * 0.0016);
    out.cover = fade * sqrt(rad / rs);
    if (out.cover < 0.004) {
        return out;
    }
    var ax = cross(vec3<f32>(0.0, 0.0, 1.0), vdir);
    ax = select(vec3<f32>(1.0, 0.0, 0.0), normalize(ax), length(ax) > 1e-4);
    let ay = cross(vdir, ax);
    out.clip = camera.view_proj * vec4<f32>(p + (ax * c.x + ay * c.y) * rs, 1.0);
    out.uv = c;
    out.pos = p;
    return out;
}

fn flake_alpha(in: SnowOut) -> f32 {
    let r = length(in.uv);
    return clamp(in.cover * (1.0 - smoothstep(0.55, 1.0, r)), 0.0, 1.0);
}

// the classic picture: white, lit as the scene is (envir.cfg's ambient and sun light)
@fragment
fn fs_snow(in: SnowOut) -> @location(0) vec4<f32> {
    let a = flake_alpha(in);
    let light = min(camera.ambient.rgb + camera.sun_color.rgb * 0.5, vec3<f32>(1.0));
    return vec4<f32>(light * 0.9 * a, a);
}


// ---- The enhanced snowfall
//
// The same field of flakes, each drawn as what it is (precip.rs `FlakeAtlas`): in a light
// fall single crystals - stellar and fernlike dendrites, sectored and plain hexagonal
// plates - in a heavier one aggregates of many, a centimetre across. A flake turns as it
// falls and a plate tips over and back (seen edge-on it thins to a line), flutters on two
// periods of its own besides its slow sway, and is blown on by the wind's gusts with its
// neighbours. Close to the eye it is out of focus, a soft disc; far off a speck. A plate
// turned to the sun flashes for a moment. Lit as the classic snowfall is by the sky and the
// lamps, the sun's share shaded by the far shadow map and thrown a little forwards; beyond
// the flakes' layers a heavy fall is a speckled veil (`veil_corner`).

struct SnowOutE {
    @builtin(position) clip: vec4<f32>,
    // across the quad (-1..1 each way); the veil: the panel's corner (0..1)
    @location(0) uv: vec2<f32>,
    // x the cover, y the atlas cell (-1 the veil), z its mip level, w how far out of focus
    // (0 sharp .. 1 a soft disc)
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) pos: vec3<f32>,
    // the flake's turn (cos, sin), how far it faces the eye (1 face on), how much it glints
    @location(3) @interpolate(flat) turn: vec4<f32>,
    // x the quad's radius over the flake's, y the veil's radius (m), z its ring
    @location(4) @interpolate(flat) extra: vec4<f32>,
};

fn snow_none() -> SnowOutE {
    var out: SnowOutE;
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    out.pos = vec3<f32>(0.0);
    out.turn = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    out.extra = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    return out;
}

@vertex
fn vs_snow_enhanced(@builtin(vertex_index) vid: u32, @builtin(instance_index) k: u32) -> SnowOutE {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let c = corners[vid % 6u];
    if (k >= u32(sp.ext.x)) {
        return snow_veil(c * 0.5 + vec2<f32>(0.5), k - u32(sp.ext.x));
    }
    var out = snow_none();
    let n0 = u32(sp.fall.z);
    let n1 = u32(sp.fall.w);
    let layer = select(select(2u, 1u, k < n0 + n1), 0u, k < n0);
    let lh = LAYER_H[layer];
    let size = vec3<f32>(2.0 * lh.x, 2.0 * lh.x, lh.z - lh.y);
    let h1 = hash4(k * 2u + 1u);
    let h2 = hash4(k * 2u + 7919u);
    let h3 = hash4(k * 5u + 15485863u);
    // the flake: its diameter and fall speed as `vs_snow` has them - but a heavy fall's
    // aggregates bigger: half a centimetre across on the mean, the big wet flakes of a
    // thick fall up to two (the classic sizes, two millimetres, are those of a moderate
    // fall's single crystals, and a heavy fall showed as a fine dust)
    let r = sp.fall.x;
    let mean = max(sp.fall.y, 0.0012 + 0.0062 * smoothstep(0.2, 0.9, r));
    let d = min(0.0005 - mean * log(max(1.0 - h1.w, 1e-6)), 0.02);
    let fall = max(0.8 * pow(d * 1000.0, 0.16) * (0.75 + 0.25 * r) * (0.85 + 0.3 * h2.x), 0.3);
    let t = sp.wind.w;
    let world_cam = camera.cam_pos.xyz + vec3<f32>(camera.world_origin.xy, 0.0);
    let lo = world_cam + vec3<f32>(-lh.x, -lh.x, lh.y);
    let base = sp.wind.xyz * (0.6 * t) - vec3<f32>(0.0, 0.0, fall * t);
    let q0 = lo + (((h1.xyz * size + base - lo) % size) + size) % size;
    // its sway, a slow circle, and its flutter, quicker and smaller, up and down as well;
    // and the gusts
    let ph = h2.w * 6.2831853 + (1.5 + 2.5 * h2.y) * t;
    let ph2 = h3.x * 6.2831853 + (4.0 + 3.0 * h3.y) * t;
    let sway = vec3<f32>(cos(ph), sin(ph), 0.0) * (0.05 + 0.15 * h2.z);
    let flutter = vec3<f32>(cos(ph2), sin(1.3 * ph2), 0.5 * sin(0.7 * ph2)) * (0.015 + 0.03 * h3.z);
    let g = precip_gust(q0.xy, t, sp.wind.xy, sp.ext.z);
    let q = q0 + sway + flutter + vec3<f32>(sp.wind.xy * (0.6 * g.x), 0.0);
    let p = q - vec3<f32>(camera.world_origin.xy, 0.0);
    if (in_own_vehicle(p)) {
        return out;
    }
    // under a roof none; one about to land on the ground or a roof fades into it (the
    // surface would cut it in half)
    let clear = roof_clearance(p, sp.roof, sp.roof_on.x);
    if (clear < -0.05) {
        return out;
    }
    let landing = smoothstep(-0.03, 0.04, clear);
    // every layer fades out at its edges, where its flakes come round again
    let rel = q0 - lo;
    let e = min(rel, size - rel) / (lh.x * 0.3);
    let fade = clamp(min(min(e.x, e.y), e.z), 0.0, 1.0) * landing;
    let to_cam = camera.cam_pos.xyz - p;
    let dist = length(to_cam);
    let vdir = to_cam / max(dist, 0.001);
    let ch = camera.view_proj * vec4<f32>(p, 1.0);
    if (fade < 0.004 || ch.w < 0.05 || abs(ch.x) > 1.3 * ch.w || abs(ch.y) > 1.3 * ch.w) {
        return out;
    }
    let cu = camera.view_proj * vec4<f32>(p + camera.cam_up.xyz, 1.0);
    let px_per_m = max(abs(cu.y / cu.w - ch.y / ch.w) * 0.5 * sp.ext.w, 1e-3);
    // In focus a few metres off (the street the eye is on): nearer, a flake's blur grows to
    // a few millimetres (as a camera's lens draws it: a big flake right in front of the eye
    // is a soft blot, not a sharp star); it is a pixel and a half wide at the least, and as
    // bright as `vs_snow` has it.
    let rad = d * 0.5;
    let blur = 0.004 * clamp(1.0 - dist / 5.0, 0.0, 1.0);
    let rs = max(sqrt(rad * rad + blur * blur), 1.5 / px_per_m);
    let cover = fade * sqrt(rad / rs);
    if (cover < 0.004) {
        return out;
    }
    let sharp = rad / rs;
    // single crystals in a light fall, the bigger flakes aggregates of them
    let aggregate = h3.w < smoothstep(0.0012, 0.004, d);
    let cell = select(floor(h3.z * 3.99), 4.0 + floor(fract(h3.z * 7.0) * 3.99), aggregate);
    // turning, and tipping over and back (an aggregate hardly shows it)
    let spin = (h3.y - 0.5) * 5.0;
    let ang = h3.x * 6.2831853 + spin * t;
    let tip = cos((0.8 + 2.2 * h2.x) * t + h1.x * 6.2831853);
    let face = select(0.25 + 0.75 * abs(tip), 0.8 + 0.2 * abs(tip), aggregate);
    let glint = pow(0.5 + 0.5 * sin((2.0 + 3.0 * h2.x) * t + h2.w * 20.0), 60.0) * select(1.0, 0.25, aggregate);
    // the atlas's level: the flake's 58 texels over its pixels
    let level = clamp(log2(58.0 / max(2.0 * rad * px_per_m, 1e-3)), 0.0, 4.0);
    var ax = cross(vec3<f32>(0.0, 0.0, 1.0), vdir);
    ax = select(vec3<f32>(1.0, 0.0, 0.0), normalize(ax), length(ax) > 1e-4);
    let ay = cross(vdir, ax);
    out.clip = camera.view_proj * vec4<f32>(p + (ax * c.x + ay * c.y) * rs, 1.0);
    out.uv = c;
    out.shape = vec4<f32>(cover, cell, level, 1.0 - smoothstep(0.35, 0.9, sharp));
    out.pos = p;
    out.turn = vec4<f32>(cos(ang), sin(ang), face, glint);
    out.extra = vec4<f32>(1.0 / max(sharp, 1e-3), 0.0, 0.0, 0.0);
    return out;
}

fn snow_veil(c: vec2<f32>, panel: u32) -> SnowOutE {
    var out = snow_none();
    let v = veil_corner(panel, c);
    let ring = f32(panel / 12u);
    if (sp.fall.x < 0.08) {
        return out;
    }
    out.clip = camera.view_proj * vec4<f32>(v.xyz, 1.0);
    out.uv = c;
    out.shape = vec4<f32>(pow(sp.fall.x, 1.2) * 0.11 * (0.8 + 0.2 * ring), -1.0, 0.0, 0.0);
    out.pos = v.xyz;
    out.extra = vec4<f32>(1.0, v.w, ring, 0.0);
    return out;
}

fn flake_alpha_e(in: SnowOutE) -> vec2<f32> {
    // the flake's own figure, turned and tipped, the crystal's rim at 1
    let u = vec2<f32>(in.turn.x * in.uv.x - in.turn.y * in.uv.y, in.turn.y * in.uv.x + in.turn.x * in.uv.y);
    let f = vec2<f32>(u.x, u.y / max(in.turn.z, 0.05)) * in.extra.x;
    let cell = vec2<f32>(in.shape.y % 4.0, floor(in.shape.y / 4.0));
    let cuv = clamp(vec2<f32>(0.5) + f * 0.45, vec2<f32>(0.0), vec2<f32>(1.0));
    let tex = textureSampleLevel(t_flakes, s_flakes, (cell + cuv) * vec2<f32>(0.25, 0.5), in.shape.z);
    let crisp = select(0.0, tex.r, all(abs(f) < vec2<f32>(1.1)));
    // out of focus: a soft disc, a touch brighter at its rim, as a lens draws a point
    let rr = length(in.uv);
    let disc = (1.0 - smoothstep(0.55, 1.0, rr)) * (0.85 + 0.15 * smoothstep(0.3, 0.8, rr));
    let a = mix(crisp, disc, in.shape.w) * in.shape.x;
    return vec2<f32>(clamp(a, 0.0, 1.0), mix(tex.g, 1.0, in.shape.w));
}

fn veil_snow_alpha(in: SnowOutE) -> f32 {
    let r = in.extra.y;
    let ring = in.extra.z;
    let d = in.pos - camera.cam_pos.xyz;
    let theta = atan2(d.y, d.x);
    let t = sp.wind.w;
    let tang = vec2<f32>(-sin(theta), cos(theta));
    // The flakes beyond, each a speck a pixel or two across, falling a metre a second and
    // drifting with the wind: two lattices of specks, a fixed angle apart, each cell with a
    // flake of its own somewhere in it - more of them showing the harder it snows.
    var specks = 0.0;
    for (var j = 0u; j < 2u; j = j + 1u) {
        let fj = f32(j);
        let cols = round((900.0 + 150.0 * ring) * (1.0 + 0.41 * fj));
        let cw = 6.2831853 * r / cols;
        let drift = dot(sp.wind.xy, tang) * 0.6 * t;
        let x = vec2<f32>(theta / 6.2831853 * cols - drift / cw, (in.pos.z + (0.9 + 0.3 * fj) * t) / cw) + vec2<f32>(0.5 * fj);
        let ci = floor(x);
        let h = cell_hash(ci + vec2<f32>(311.0 * ring, 57.0 * fj));
        let centre = vec2<f32>(0.2) + 0.6 * h.xy + 0.08 * vec2<f32>(sin(t * 1.7 + h.z * 30.0), 0.0);
        let s = (1.0 - smoothstep(0.08, 0.22, length(fract(x) - centre))) * step(h.w, 0.35 + 0.5 * sp.fall.x);
        specks = max(specks, s * (0.5 + 0.5 * h.z));
    }
    let g = precip_gust(in.pos.xy + camera.world_origin.xy, t, sp.wind.xy, sp.ext.z);
    // (as the rain's veil, rain.wgsl `veil_rain_alpha`: gentler gusts, kept to the horizon, and
    // little flat grey under the specks - looked up at, the panels stood in the sky as a ring
    // of pale sheets)
    let sweep = clamp(1.0 + 0.6 * g.y, 0.7, 1.3);
    let up = in.pos.z - camera.cam_pos.z;
    let height = smoothstep(-3.0, 3.0, up) * (1.0 - smoothstep(0.08, 0.3, up / r));
    return clamp(in.shape.x * (0.2 + 0.8 * specks) * sweep * height, 0.0, 1.0);
}

// the enhanced picture: what the flake scatters of the sky's, the sun's and the lamps'
// light, at the albedo of snow
@fragment
fn fs_snow_enhanced(in: SnowOutE) -> @location(0) vec4<f32> {
    let veil = in.shape.y < -0.5;
    var a = 0.0;
    var bright = 1.0;
    if (veil) {
        a = veil_snow_alpha(in);
    } else {
        let fa = flake_alpha_e(in);
        a = fa.x;
        bright = fa.y;
    }
    if (a < 0.002) {
        discard;
    }
    let to_eye = normalize(camera.cam_pos.xyz - in.pos);
    let sun = precip_sun(in.pos);
    // the sun's share: `precip_light` takes it as even and unshaded; here it is shaded, and
    // thrown a little forwards (a fall lit from behind is brighter), and a tipping plate
    // flashes as it faces it
    let c = dot(-camera.sun_dir.xyz, to_eye);
    let lobe = 0.7 * hg_phase(c, 0.35) * 4.0 * PI + 0.3;
    let sun_part = enh.sun.rgb * (0.25 / PI * (sun * lobe - 1.0) + sun * in.turn.w * 0.6);
    let up = sh_irradiance(vec3<f32>(0.0, 0.0, 1.0)) / PI;
    var l = vec3<f32>(0.0);
    if (veil) {
        l = (enh.fog_color.rgb / 0.9 + enh.sun.rgb * 0.25 / PI + up) * 0.5 * 0.85 + sun_part * 0.425;
    } else {
        l = (up + up) * 0.5 * 0.85 + sun_part * 0.425;
    }
    l = max(l, vec3<f32>(0.0)) * mix(0.75, 1.05, bright);
    return vec4<f32>(min(l * enh.exposure.x, vec3<f32>(60.0)) * a, a);
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

struct PointLight {
    pos: vec4<f32>,
    color: vec4<f32>,
    dir: vec4<f32>,
    extra: vec4<f32>,
};
@group(0) @binding(3) var<storage, read> lights: array<PointLight>;
@group(0) @binding(4) var<storage, read> grid: array<u32>;
const CELL_CAP: u32 = 32u;

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

