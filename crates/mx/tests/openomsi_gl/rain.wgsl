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
@group(1) @binding(1) var roof_map: texture_2d<f32>;
fn roof_depth(t: vec2<i32>) -> f32 { return textureLoad(roof_map, t, 0).x; }

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

// The rainfall, every mode: a field of drops that lives on the GPU alone, as the snowfall
// does (snow.wgsl) - each drop's place is worked out here from its number, the time, the
// wind and the camera, so that a downpour is a quarter of a million drops for one draw.
// The field is the world's, repeating round the camera in three layers, dense close by,
// thinner further out; beyond them the rain's extinction takes the view (the fog).
//
// A drop is seen as the streak it draws in the eye's time (`ep.eye.w`): its way against
// the eye in that time, short and upright standing, long and slanting at a moving bus's
// windscreen. It is as wide as the drop is - a raindrop of two millimetres is less than a
// pixel wide a few metres off - so a streak is drawn a pixel wide at least, as transparent
// as the drop covers less of it: a downpour is a veil of many fine, faint lines, not a few
// thick ones. The drops are drawn as rain shows them, weighted by the light they catch
// (their cross-section) from Marshall and Palmer's sizes (a gamma of order 3 over the
// exponential), each falling at its own speed (Atlas, Srivastava and Sekhon 1973).

struct RainParams {
    // xyz the wind the drops drift with (m/s), w the time (s)
    wind: vec4<f32>,
    // xyz the eye's velocity (m/s, world), w the time the eye sees a drop as one streak (s)
    eye: vec4<f32>,
    // x how hard it rains (0..1), y the median volume diameter of its drops (m), zw how
    // many drops the first two layers hold (the rest are the third's)
    fall: vec4<f32>,
    // xy the picture's size (px), z how opaque a drop's streak is, w how many of the
    // instances are drops (the rest are their splashes)
    view: vec4<f32>,
    // the precipitation's roof map: its matrix, and x 1 when there is one
    roof: mat4x4<f32>,
    roof_on: vec4<f32>,
    // Enhanced: x the instance the veil's panels begin at (the drops' and the splashes' end),
    // y how many panels there are, z how strong the wind's gusts are (0..1), w the rain's
    // quality (0 low .. 2 high)
    ext: vec4<f32>,
};
@group(1) @binding(0) var<uniform> rp: RainParams;

struct RainOut {
    @builtin(position) clip: vec4<f32>,
    // across the streak and along it from its tail (px)
    @location(0) uv: vec2<f32>,
    // x the streak's half width, y its length (px), z its cover (alpha)
    @location(1) @interpolate(flat) shape: vec3<f32>,
    @location(2) pos: vec3<f32>,
};

// The layers round the camera: half width, bottom and top (m).
const RAIN_LAYERS = array<vec3<f32>, 3>(vec3<f32>(5.0, -2.5, 6.0), vec3<f32>(14.0, -3.0, 12.0), vec3<f32>(32.0, -3.0, 22.0));

// Where a drop lands: a crown of water a centimetre or two high that is gone in a tenth of
// a second - on whatever is uppermost there, the ground, a kerb, a bus's roof, as the
// precipitation's roof map has it (none without the map). Each splash takes a new place
// round the camera every time it comes round again; the most of them close by.
fn splash(vid: u32, s: u32) -> RainOut {
    // (a strip of four vertices)
    let c = vec2<f32>(f32(i32(vid & 1u) * 2 - 1), f32(i32(vid >> 1u) * 2 - 1));
    var out: RainOut;
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.shape = vec3<f32>(0.0);
    out.pos = vec3<f32>(0.0);
    let life = 0.1;
    let h0 = hash4(s * 3u + 11u);
    let cycle = rp.wind.w / life + h0.w;
    let age = fract(cycle);
    let h = hash4(s * 7919u + u32(floor(cycle)) * 2654435761u);
    // its place in the world: a field 28 m square laid over the world (in the world's own
    // figures), the copy of it round the camera - a splash stays where it landed however
    // the camera moves or turns, and only those the camera leaves behind come round again
    // ahead of it
    let size = 28.0;
    let world_cam = camera.cam_pos.xy + camera.world_origin.xy;
    let lo = world_cam - vec2<f32>(0.5 * size);
    let wxy = lo + (((h.xy * size - lo) % size) + size) % size;
    let xy = wxy - camera.world_origin.xy;
    // (fewer further off, faded in and out by the distance alone)
    let fade_far = 1.0 - smoothstep(6.0, 14.0, length(xy - camera.cam_pos.xy));
    let m = rp.roof;
    let mc = m * vec4<f32>(xy, 0.0, 1.0);
    let uv = vec2<f32>(mc.x * 0.5 + 0.5, 0.5 - mc.y * 0.5);
    let side = f32(textureDimensions(roof_map).x);
    let top = roof_depth(vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999)) * side));
    // (the map's depth back to a height: it looks straight down)
    let z = (top - mc.z) / m[2].z;
    let p = vec3<f32>(xy, z + 0.008 + 0.012 * age);
    let ch = camera.view_proj * vec4<f32>(p, 1.0);
    let cu = camera.view_proj * vec4<f32>(p + camera.cam_up.xyz, 1.0);
    if (rp.roof_on.x < 0.5 || top >= 0.9999 || ch.w < 0.3 || in_own_vehicle(p)) {
        return out;
    }
    let half_px = 0.5 * rp.view.xy;
    let px_per_m = length(cu.xy / cu.w * half_px - ch.xy / ch.w * half_px);
    // the crown widens as it rises and falls back, and fades
    let w = (0.006 + 0.02 * age) * px_per_m;
    let hw = max(0.5 * w, 0.5);
    let pad = hw + 1.0;
    let cover = min(w, 1.0) * (1.0 - age) * (1.0 - age) * 0.8 * rp.view.z * fade_far;
    out.clip = vec4<f32>(ch.xy + c * pad / half_px * ch.w, ch.zw);
    out.uv = c * pad;
    // (a round speck: no length, see `streak_alpha`)
    out.shape = vec3<f32>(hw, -1.0, cover);
    out.pos = p;
    return out;
}

@vertex
fn vs_rain(@builtin(vertex_index) vid: u32, @builtin(instance_index) k: u32) -> RainOut {
    if (k >= u32(rp.view.w)) {
        return splash(vid, k - u32(rp.view.w));
    }
    // across (-1, 1) and along (0 the tail, 1 the head)
    let c = vec2<f32>(f32(i32(vid & 1u) * 2 - 1), f32(vid >> 1u));
    var out: RainOut;
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.shape = vec3<f32>(0.0);
    out.pos = vec3<f32>(0.0);
    let n0 = u32(rp.fall.z);
    let n1 = u32(rp.fall.w);
    let layer = select(select(2u, 1u, k < n0 + n1), 0u, k < n0);
    let lh = RAIN_LAYERS[layer];
    let size = vec3<f32>(2.0 * lh.x, 2.0 * lh.x, lh.z - lh.y);
    let h1 = hash4(k * 2u + 3u);
    let h2 = hash4(k * 2u + 6007u);
    // the drop: its diameter (m), drawn by its cross-section from the rain's sizes, and
    // the speed it falls at
    let lambda = 3.67 / max(rp.fall.y, 1e-4);
    let d = clamp(-log(max(h2.x * h2.y * h2.z, 1e-12)) / lambda, 0.0003, 0.006);
    let fall = max(9.65 - 10.3 * exp(-600.0 * d), 0.5);
    // its place in the world: where it started, carried by the wind, falling; then the
    // copy of the field nearest the camera (in the world's own figures, so that it stays
    // put while the render origin moves)
    let t = rp.wind.w;
    let world_cam = camera.cam_pos.xyz + vec3<f32>(camera.world_origin.xy, 0.0);
    let vel = vec3<f32>(rp.wind.xy, -fall);
    let lo = world_cam + vec3<f32>(-lh.x, -lh.x, lh.y);
    // (the field repeats every `size`: the place wrapped into one copy of it)
    let wrapped = (h1.xyz * size + vel * t - lo) / size;
    let q = lo + (wrapped - floor(wrapped)) * size;
    let p = q - vec3<f32>(camera.world_origin.xy, 0.0);
    // a denser layer thins out with the distance from the eye, over the whole of its reach,
    // into the thinner one round it: faded only at its edges, it stood round the camera as a
    // box of dense rain that went wherever the camera went
    let rel = q - lo;
    let e = min(rel, size - rel) / (lh.x * 0.3);
    let edge = clamp(min(min(e.x, e.y), e.z), 0.0, 1.0);
    let reach = 1.0 - smoothstep(0.25 * lh.x, lh.x, length((q - world_cam).xy));
    let fade = select(1.0, edge * reach, layer < 2u);
    // a drop right at the eye is a blur too wide and faint to see, not a streak
    let near = smoothstep(0.6, 1.6, length(p - camera.cam_pos.xyz));
    // the streak: from where the drop was against the eye a streak's time ago to where it is
    let ch = camera.view_proj * vec4<f32>(p, 1.0);
    // (the field reaches all round the camera, most of it out of the picture: a drop well
    // beyond its edge - a streak is a few per cent of the picture long - is let go before
    // the rest of its work, and so is one too faint to see)
    if (fade * near < 0.01 || ch.w < 0.1 || abs(ch.x) > 1.5 * ch.w || abs(ch.y) > 1.5 * ch.w) {
        return out;
    }
    let tail = p - (vel - rp.eye.xyz) * rp.eye.w;
    let ct = camera.view_proj * vec4<f32>(tail, 1.0);
    if (ct.w < 0.1 || in_own_vehicle(p) || under_roof(p, rp.roof, rp.roof_on.x)) {
        return out;
    }
    let cu = camera.view_proj * vec4<f32>(p + camera.cam_up.xyz, 1.0);
    let half_px = 0.5 * rp.view.xy;
    let sh = ch.xy / ch.w * half_px;
    let st = ct.xy / ct.w * half_px;
    let along = sh - st;
    let len = length(along);
    let dir = select(vec2<f32>(0.0, 1.0), along / max(len, 1e-4), len > 1e-3);
    let across = vec2<f32>(-dir.y, dir.x);
    // as wide as the drop (out of focus close by: its blur as wide as the pupil, some
    // 2.5 mm), a pixel at least - and as much fainter as it is narrower than that
    let px_per_m = length(cu.xy / cu.w * half_px - sh);
    let blur = sqrt(d * d + 0.0025 * 0.0025);
    let w = blur * px_per_m;
    // and spread along its way: the drop lies on each pixel of its streak for the share of
    // the eye's time it takes to cross it - a long streak is a faint one (the drop's lens
    // gathers the sky's light, brighter than what is behind it: some six times its width)
    let spread = clamp(6.0 * max(w, 1.0) / max(len, 1.0), 0.2, 1.0);
    let cover = min(w, 1.0) * (d / blur) * spread * fade * near * rp.view.z * clamp(sqrt(d / 0.002), 0.5, 1.4);
    // (a hair under a pixel at the least: a picture drawn smaller and scaled up widened the
    // full-pixel lines into the thick white bars of the old sprites)
    let hw = max(0.5 * w, 0.4);
    let pad = hw + 1.0;
    // the quad round the streak (the head's and the tail's clip positions mixed, so that it
    // keeps the depth along it) widened in the picture's pixels
    let clip = mix(ct, ch, c.y);
    let off = across * c.x * pad + dir * (c.y * 2.0 - 1.0) * pad;
    out.clip = vec4<f32>(clip.xy + off / half_px * clip.w, clip.zw);
    out.uv = vec2<f32>(c.x * pad, c.y * (len + 2.0 * pad) - pad);
    out.shape = vec3<f32>(hw, len, cover);
    out.pos = p;
    return out;
}

fn streak_alpha(in: RainOut) -> f32 {
    let hw = in.shape.x;
    let len = in.shape.y;
    // a pixel's edge across and at both ends, and a little fainter towards the tail
    let a_across = clamp(hw + 0.5 - abs(in.uv.x), 0.0, 1.0);
    let a_along = clamp(0.5 + min(in.uv.y, len - in.uv.y), 0.0, 1.0);
    let tail = mix(0.55, 1.0, clamp(in.uv.y / max(len, 1.0), 0.0, 1.0));
    // (a splash's round speck)
    let speck = clamp(hw + 0.5 - length(in.uv), 0.0, 1.0);
    return clamp(in.shape.z * select(a_across * a_along * tail, speck, len < 0.0), 0.0, 1.0);
}

// the classic picture: water lit as the scene is (envir.cfg's ambient and sun light)
@fragment
fn fs_rain(in: RainOut) -> @location(0) vec4<f32> {
    let a = streak_alpha(in);
    let light = min(camera.ambient.rgb + camera.sky_color.rgb * 0.5 + camera.sun_color.rgb * 0.3, vec3<f32>(1.0)) * vec3<f32>(0.68, 0.7, 0.75);
    return vec4<f32>(light * a, a);
}


// ---- The enhanced rainfall
//
// The same field of drops, drawn as rain photographs: each drop the streak it leaves in the
// eye's time, its length and slant its way against the eye (the bus's own motion, the wind
// and its gusts), a soft line as wide as the drop with rounded ends, beaded along a long
// streak where the oscillating drop flickers. A drop is a small ball lens: it carries the
// light of what lies round it turned over - the sky in its lower half, the street in its
// upper - so that by day it is a silvery line, faint against the bright sky and clear
// against a dark street, glows when the sun shines through it from behind, colours the
// rainbow's ring round the antisolar point, and flashes in the lamps and headlights at
// night (`precip_light`). Where the drops land, crowns of water rise and throw droplets;
// beyond the drops' layers the rain is a veil of streaks (`veil_corner`).

struct RainOutE {
    @builtin(position) clip: vec4<f32>,
    // a drop: across its streak and along it from the tail (px); a splash: across it and up
    // it (-1..1, 0..1 and over); the veil: the panel's corner (0..1)
    @location(0) uv: vec2<f32>,
    // a drop: x the streak's half width, y its length (px); a splash: x one pixel in its
    // own measure, y its age (0..1); z the cover, w which it is (0 a drop, 1 a splash, 2 the
    // veil)
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) pos: vec3<f32>,
    // x a number of the particle's own (0..1), y the veil's radius (m), z its ring
    @location(3) @interpolate(flat) seed: vec4<f32>,
};

fn rain_none() -> RainOutE {
    var out: RainOutE;
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    out.pos = vec3<f32>(0.0);
    out.seed = vec4<f32>(0.0);
    return out;
}

@vertex
fn vs_rain_enhanced(@builtin(vertex_index) vid: u32, @builtin(instance_index) k: u32) -> RainOutE {
    if (k >= u32(rp.ext.x)) {
        return rain_veil(vid, k - u32(rp.ext.x));
    }
    if (k >= u32(rp.view.w)) {
        return rain_crown(vid, k - u32(rp.view.w));
    }
    return rain_drop(vid, k);
}

fn rain_drop(vid: u32, k: u32) -> RainOutE {
    // across (-1, 1) and along (0 the tail, 1 the head)
    let c = vec2<f32>(f32(i32(vid & 1u) * 2 - 1), f32(vid >> 1u));
    var out = rain_none();
    let n0 = u32(rp.fall.z);
    let n1 = u32(rp.fall.w);
    let layer = select(select(2u, 1u, k < n0 + n1), 0u, k < n0);
    let lh = RAIN_LAYERS[layer];
    let size = vec3<f32>(2.0 * lh.x, 2.0 * lh.x, lh.z - lh.y);
    let h1 = hash4(k * 2u + 3u);
    let h2 = hash4(k * 2u + 6007u);
    // the drop's diameter and speed, as `vs_rain` draws them
    let lambda = 3.67 / max(rp.fall.y, 1e-4);
    let d = clamp(-log(max(h2.x * h2.y * h2.z, 1e-12)) / lambda, 0.0003, 0.006);
    let fall = max(9.65 - 10.3 * exp(-600.0 * d), 0.5);
    let t = rp.wind.w;
    let world_cam = camera.cam_pos.xyz + vec3<f32>(camera.world_origin.xy, 0.0);
    let lo = world_cam + vec3<f32>(-lh.x, -lh.x, lh.y);
    let wrapped = (h1.xyz * size + vec3<f32>(rp.wind.xy, -fall) * t - lo) / size;
    let q0 = lo + (wrapped - floor(wrapped)) * size;
    // the gusts carry it on from its place in the field, and slant it as they blow
    let g = precip_gust(q0.xy, t, rp.wind.xy, rp.ext.z);
    let q = q0 + vec3<f32>(rp.wind.xy * g.x, 0.0);
    let vel = vec3<f32>(rp.wind.xy * (1.0 + g.y), -fall);
    let p = q - vec3<f32>(camera.world_origin.xy, 0.0);
    // every layer fades out at its edges (where its drops come round again, the gusts'
    // way at the far side of the field differs from the near side's), the denser ones
    // over their whole reach into the thinner round them
    let rel = q0 - lo;
    let e = min(rel, size - rel) / (lh.x * 0.3);
    let edge = clamp(min(min(e.x, e.y), e.z), 0.0, 1.0);
    let reach = 1.0 - smoothstep(0.25 * lh.x, lh.x, length((q - world_cam).xy));
    let fade = edge * select(1.0, reach, layer < 2u);
    let near = smoothstep(0.5, 1.4, length(p - camera.cam_pos.xyz));
    let ch = camera.view_proj * vec4<f32>(p, 1.0);
    if (fade * near < 0.01 || ch.w < 0.1 || abs(ch.x) > 1.5 * ch.w || abs(ch.y) > 1.5 * ch.w) {
        return out;
    }
    let tail = p - (vel - rp.eye.xyz) * rp.eye.w;
    let ct = camera.view_proj * vec4<f32>(tail, 1.0);
    if (ct.w < 0.1 || in_own_vehicle(p) || under_roof(p, rp.roof, rp.roof_on.x)) {
        return out;
    }
    let cu = camera.view_proj * vec4<f32>(p + camera.cam_up.xyz, 1.0);
    let half_px = 0.5 * rp.view.xy;
    let sh = ch.xy / ch.w * half_px;
    let st = ct.xy / ct.w * half_px;
    let along = sh - st;
    let len = length(along);
    let dir = select(vec2<f32>(0.0, 1.0), along / max(len, 1e-3), len > 1e-3);
    let across = vec2<f32>(-dir.y, dir.x);
    // as wide as the drop, a millimetre's blur at the least (crisp close by: the eye is
    // focused on the street, a few metres off, and a drop there is sharp), and spread over
    // its way as `vs_rain` spreads it
    let px_per_m = length(cu.xy / cu.w * half_px - sh);
    // (a millimetre and a half at the least: at one millimetre the near drops drew as opaque
    // grey lines)
    let blur = sqrt(d * d + 0.0015 * 0.0015);
    let w = blur * px_per_m;
    let spread = clamp(6.0 * max(w, 1.0) / max(len, 1.0), 0.1, 1.0);
    let cover = min(w, 1.0) * (d / blur) * spread * fade * near * rp.view.z * clamp(sqrt(d / 0.002), 0.5, 1.4);
    let hw = max(0.5 * w, 0.4);
    let pad = hw + 1.5;
    let clip = mix(ct, ch, c.y);
    let off = across * c.x * pad + dir * (c.y * 2.0 - 1.0) * pad;
    out.clip = vec4<f32>(clip.xy + off / half_px * clip.w, clip.zw);
    out.uv = vec2<f32>(c.x * pad, c.y * (len + 2.0 * pad) - pad);
    out.shape = vec4<f32>(hw, len, cover, 0.0);
    out.pos = p;
    out.seed = vec4<f32>(h2.w, 0.0, 0.0, 0.0);
    return out;
}

// Where a drop lands (`splash`'s places): a crown of water rises round the impact for a
// few hundredths of a second, its rim breaks into points, and the points fling droplets
// out on short arcs that fall back - on the street, a kerb, a bus's roof. Upright, it is
// drawn facing the eye.
fn rain_crown(vid: u32, s: u32) -> RainOutE {
    // across (-1, 1) and up (0, 1)
    let c = vec2<f32>(f32(i32(vid & 1u) * 2 - 1), f32(vid >> 1u));
    var out = rain_none();
    let life = 0.24;
    let h0 = hash4(s * 3u + 11u);
    let cycle = rp.wind.w / life + h0.w;
    let age = fract(cycle);
    let h = hash4(s * 7919u + u32(floor(cycle)) * 2654435761u);
    let size = 28.0;
    let world_cam = camera.cam_pos.xy + camera.world_origin.xy;
    let lo = world_cam - vec2<f32>(0.5 * size);
    let wxy = lo + (((h.xy * size - lo) % size) + size) % size;
    let xy = wxy - camera.world_origin.xy;
    let fade_far = 1.0 - smoothstep(7.0, 16.0, length(xy - camera.cam_pos.xy));
    let m = rp.roof;
    let mc = m * vec4<f32>(xy, 0.0, 1.0);
    let uv = vec2<f32>(mc.x * 0.5 + 0.5, 0.5 - mc.y * 0.5);
    let side = f32(textureDimensions(roof_map).x);
    let top = roof_depth(vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999)) * side));
    let p = vec3<f32>(xy, (top - mc.z) / m[2].z + 0.002);
    let ch = camera.view_proj * vec4<f32>(p, 1.0);
    if (rp.roof_on.x < 0.5 || top >= 0.9999 || fade_far <= 0.0 || ch.w < 0.3 || in_own_vehicle(p)) {
        return out;
    }
    let cu = camera.view_proj * vec4<f32>(p + camera.cam_up.xyz, 1.0);
    let half_px = 0.5 * rp.view.xy;
    let px_per_m = length(cu.xy / cu.w * half_px - ch.xy / ch.w * half_px);
    // a few centimetres across, the bigger the drops the bigger, and at least a few pixels
    // (the crown drawn smaller inside it, its parts no thinner than a pixel)
    let scale = clamp(rp.fall.y / 0.0018, 0.6, 1.6) * (0.7 + 0.6 * h.z);
    let half_w = 0.045 * scale;
    let grow = max(1.0, 2.0 / max(half_w * px_per_m, 1e-3));
    let to_cam = camera.cam_pos.xyz - p;
    let flat_side = vec3<f32>(-to_cam.y, to_cam.x, 0.0);
    let ax = select(vec3<f32>(1.0, 0.0, 0.0), normalize(flat_side), length(flat_side) > 1e-4);
    let corner = p + ax * (c.x * half_w * grow) + vec3<f32>(0.0, 0.0, (c.y * 1.2 * grow - 0.05) * half_w);
    out.clip = camera.view_proj * vec4<f32>(corner, 1.0);
    out.uv = vec2<f32>(c.x * grow, (c.y * 1.2 * grow - 0.05));
    out.shape = vec4<f32>(1.0 / max(half_w * px_per_m, 1e-3), age, 0.9 * rp.view.z * fade_far, 1.0);
    out.pos = p;
    out.seed = vec4<f32>(h.w, 0.0, 0.0, 0.0);
    return out;
}

fn rain_veil(vid: u32, panel: u32) -> RainOutE {
    let c = vec2<f32>(f32(vid & 1u), f32(vid >> 1u));
    var out = rain_none();
    let v = veil_corner(panel, c);
    let ring = f32(panel / 12u);
    // (none in a light rain, whose drops are not seen that far)
    if (rp.fall.x < 0.15) {
        return out;
    }
    out.clip = camera.view_proj * vec4<f32>(v.xyz, 1.0);
    out.uv = c;
    // (the rain's streaks are seen beyond the drops only in a fair rain, and as a veil in a
    // heavy one; the further rings stand for more rain behind them)
    out.shape = vec4<f32>(0.0, 0.0, pow(rp.fall.x, 1.5) * 0.085 * (0.8 + 0.2 * ring), 2.0);
    out.pos = v.xyz;
    out.seed = vec4<f32>(0.0, v.w, ring, 0.0);
    return out;
}

fn drop_alpha(in: RainOutE) -> f32 {
    let hw = in.shape.x;
    let len = in.shape.y;
    // across: a soft line, as a lens draws a thin one - at least a pixel's blur
    let s = max(hw * 0.85, 0.6);
    let across = exp(-0.5 * in.uv.x * in.uv.x / (s * s));
    // along: the drop's way, its ends rounded as wide as the drop
    let e = hw + 0.7;
    let along = smoothstep(-e, e, in.uv.y) * smoothstep(-e, e, len - in.uv.y);
    // a little brighter at the head (where the drop is now), and a long streak beaded: a
    // falling drop oscillates a few hundred times a second, its glint flickering on its way
    let f = clamp(in.uv.y / max(len, 1.0), 0.0, 1.0);
    let beads = 1.0 + 0.22 * sin(in.uv.y * (0.35 + 0.3 * in.seed.x) + in.seed.x * 40.0) * smoothstep(12.0, 40.0, len);
    return clamp(in.shape.z * across * along * mix(0.75, 1.0, f) * beads, 0.0, 1.0);
}

fn crown_alpha(in: RainOutE) -> f32 {
    let px = in.shape.x;
    let age = in.shape.y;
    let seed = in.seed.x;
    let uv = in.uv;
    // the crown: a thin sheet rising round the impact, a cylinder seen from the side - most
    // of it at its outline, where the eye looks along the sheet - its rim pulled into points
    let rise = smoothstep(0.0, 0.18, age) * (1.0 - smoothstep(0.3, 0.6, age));
    let r = 0.35 + 0.5 * sqrt(age);
    let x = uv.x / r;
    let points = 7.0 + floor(seed * 5.0);
    let ang = asin(clamp(x, -1.0, 1.0));
    let tips = pow(abs(cos(ang * points * 0.5 + seed * 6.2831853)), 6.0);
    let top = rise * (0.5 + 0.35 * tips) * (0.8 + 0.4 * fract(seed * 7.3));
    let outline = 0.3 + 0.7 * smoothstep(0.5, 1.0, abs(x));
    let side = 1.0 - smoothstep(1.0 - px / r, 1.0 + px / r, abs(x));
    let below = 1.0 - smoothstep(top - px, top + px, uv.y);
    // (a sheet thinner than a pixel is as faint as it is thin)
    let crown = side * below * outline * step(0.0, uv.y) * 0.75 * min(1.0, 0.04 / px);
    // the droplets the points fling off, on arcs out and up and back down
    var drops = 0.0;
    let fly = max(age - 0.14, 0.0);
    let rad = max(0.07, px);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let hi = hash4(u32(seed * 16777216.0) + i * 977u);
        let vx = (hi.x * 2.0 - 1.0) * 2.4;
        let vy = 2.2 + 2.2 * hi.y;
        let q = vec2<f32>(sign(vx) * r * 0.9 + vx * fly, 0.5 * rise + vy * fly - 14.0 * fly * fly);
        let a = (1.0 - smoothstep(rad - px, rad + px, length(uv - q))) * step(0.0, q.y) * step(0.05, fly);
        drops = max(drops, a * (0.07 / rad) * (0.07 / rad) * step(hi.z, 0.8));
    }
    return clamp(in.shape.z * max(crown, drops) * (1.0 - age * age), 0.0, 1.0);
}

fn veil_rain_alpha(in: RainOutE) -> f32 {
    let r = in.seed.y;
    let ring = in.seed.z;
    let d = in.pos - camera.cam_pos.xyz;
    let theta = atan2(d.y, d.x);
    let t = rp.wind.w;
    let tang = vec2<f32>(-sin(theta), cos(theta));
    let fall = 7.0;
    // (the wind slants the streaks: so many metres sideways for each metre of fall)
    let slant = dot(rp.wind.xy, tang) / fall;
    // Columns of streaks, a fixed angle apart (a few pixels) and a whole number of them to
    // the ring, so that its seam does not show; each column's drops fall at their own
    // speed, a streak a stretch of it.
    var streaks = 0.0;
    for (var j = 0u; j < 2u; j = j + 1u) {
        let fj = f32(j);
        let cols = round((1400.0 + 216.0 * ring) * (1.0 + 0.37 * fj));
        let cw = 6.2831853 * r / cols;
        let x = theta / 6.2831853 * cols - slant * in.pos.z / cw + 0.5 * fj;
        let ci = floor(x);
        let h = cell_hash(vec2<f32>(ci, 97.0 * fj + 13.0 * ring));
        let fx = fract(x) - 0.5 - (h.x - 0.5) * 0.4;
        let speed = fall * (0.85 + 0.3 * h.y);
        let period = r * (0.07 + 0.05 * h.z);
        let v = fract((in.pos.z + speed * t) / period + h.w);
        let len = 0.25 + 0.35 * h.x;
        let along = smoothstep(0.0, 0.1, v) * (1.0 - smoothstep(len - 0.1, len, v));
        let across = 1.0 - smoothstep(0.06, 0.16, abs(fx));
        streaks = max(streaks, along * across * (0.4 + 0.6 * h.y));
    }
    // the gusts sweep through it as a little denser sheets (at the full swing of 0.3 to 1.9 they
    // stood in the sky as broad pale bands, sliding across it)
    let g = precip_gust(in.pos.xy + camera.world_origin.xy, t, rp.wind.xy, rp.ext.z);
    let sweep = clamp(1.0 + 0.6 * g.y, 0.7, 1.3);
    // (thinned out towards the ground, where the panel meets it in a line, and above the
    // horizon: the veil is the far rain seen edgewise along the ground - looked up at, its
    // panels and their tops stood round the eye as a ring of grey sheets in the sky; what the
    // rain takes from the view up there is its extinction, the fog's)
    let up = in.pos.z - camera.cam_pos.z;
    let height = smoothstep(-3.0, 3.0, up) * (1.0 - smoothstep(0.08, 0.3, up / r));
    // (the streaks themselves: a flat grey under them read as a sheet of frosted glass)
    return clamp(in.shape.z * (0.12 + 0.88 * streaks) * sweep * height, 0.0, 1.0);
}

// The light a drop sends to the eye besides the lamps': the sky's and the street's through
// it, and the sun's - refracted forwards (a shower lit from behind shines) and, at 42 and
// 51 degrees round the antisolar point, split into the rainbow's colours.
fn rain_surround(to_eye: vec3<f32>, sun: f32) -> vec3<f32> {
    let up = sh_irradiance(vec3<f32>(0.0, 0.0, 1.0)) / PI;
    let down = sh_irradiance(vec3<f32>(0.0, 0.0, -1.0)) / PI;
    // (a ball lens gathers the light of half the sky into the eye: a drop's streak is a
    // little brighter than the overcast behind it - a silvery line - and much brighter
    // than the street, the trees and the walls)
    let l = mix(down, up, 0.8) * 1.0;
    let c = dot(-camera.sun_dir.xyz, to_eye);
    let forward = 0.85 * hg_phase(c, 0.82) + 0.15 / (4.0 * PI);
    let a = degrees(acos(clamp(-c, -1.0, 1.0)));
    let b1 = (a - vec3<f32>(42.3, 41.5, 40.7)) / 0.75;
    let b2 = (a - vec3<f32>(50.3, 51.2, 52.4)) / 1.0;
    let bow = exp(-b1 * b1) * 0.12 + exp(-b2 * b2) * 0.045;
    return l + enh.sun.rgb * sun * (forward + bow);
}

@fragment
fn fs_rain_enhanced(in: RainOutE) -> @location(0) vec4<f32> {
    let kind = in.shape.w;
    var a = 0.0;
    if (kind < 0.5) {
        a = drop_alpha(in);
    } else if (kind < 1.5) {
        a = crown_alpha(in);
    } else {
        a = veil_rain_alpha(in);
    }
    if (a < 0.002) {
        discard;
    }
    let to_eye = normalize(camera.cam_pos.xyz - in.pos);
    var l = rain_surround(to_eye, precip_sun(in.pos));
    // (the lamps for the drops and the crowns alone: the veil's panels cover much of the
    // picture, and the lamps' light in the air is the fog's). A drop sends a lamp's light
    // every way - the glint of its surface, a small mirror ball, besides what it refracts on
    // forwards - so the streaks shine in a street lamp's light however the eye looks at
    // them, as rain is seen at night (`precip_light`'s flake lobe, less the light round it,
    // which `rain_surround` gives already)
    if (kind < 1.5) {
        let round = enh.fog_color.rgb / 0.9 + enh.sun.rgb * (0.25 / PI);
        l = max(precip_light(in.pos, to_eye, true) - round, vec3<f32>(0.0)) * 1.5 + l;
    }
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
fn lights_at(i: u32) -> PointLight { return PointLight(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)); }
fn grid_at(i: u32) -> u32 { return 0xffffffffu; }
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
                let li = grid_at(base + j);
                if (li == 0xffffffffu) {
                    break;
                }
                let l = lights_at(li);
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
                let li = grid_at(base + j);
                if (li == 0xffffffffu) {
                    break;
                }
                let l = lights_at(li);
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

