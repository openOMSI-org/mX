// Enhanced graphics: the light the air scatters towards the camera that the main pass
// leaves out - the lamps' in the weather's fog, rain or falling snow (`lamp_airlight` in
// lamp_air.wgsl) and the sun's shafts between the shadows (`sun_shafts`) - added onto the
// drawn picture before its glow and its metering. Once per pixel over the depth prepass: worked out in
// the main pass it was paid again for every layer drawn over a pixel (painted ground,
// decals), several milliseconds a frame in a foggy night.
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

struct FogLampParams {
    // clip space (reversed Z) back to the render-origin relative world
    inv_view_proj: mat4x4<f32>,
    // xy the picture's size in pixels, z 1: the sun's glare is added (2: it is all that is
    // added, the half-size result holds only the sun's row), w the share of the weather's
    // extinction that is mist and fog (not rain)
    size: vec4<f32>,
};
@group(1) @binding(0) var<uniform> fp: FogLampParams;
@group(1) @binding(1) var t_depth: texture_depth_2d;
// the half-size result (rgb pre-exposed light, a the distance it was worked out to)
@group(1) @binding(2) var t_fog: texture_2d<f32>;

struct FogVsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_fog_lamps(@builtin(vertex_index) i: u32) -> FogVsOut {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    var out: FogVsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// The camera, the direction through a full-size pixel position and the distance to what
// the prepass holds there (3 km for the sky).
fn pixel_ray(at: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t_depth));
    let px = clamp(vec2<i32>(at), vec2<i32>(0), dims - vec2<i32>(1));
    let d = textureLoad(t_depth, px, 0);
    let ndc = vec2<f32>(at.x / fp.size.x * 2.0 - 1.0, 1.0 - at.y / fp.size.y * 2.0);
    let c = camera.cam_pos.xyz;
    let mid = fp.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    let dir = normalize(mid.xyz / mid.w - c);
    var len = 3000.0;
    if (d > 0.0) {
        let h = fp.inv_view_proj * vec4<f32>(ndc, d, 1.0);
        len = distance(h.xyz / h.w, c);
    }
    return vec4<f32>(dir, len);
}

// The sun's shadow maps (shader.wgsl's bindings and constants).
@group(0) @binding(5) var t_shadow: texture_depth_2d;
@group(0) @binding(6) var s_shadow: sampler_comparison;
@group(0) @binding(7) var t_shadow_far: texture_depth_2d;
const SHAFT_DEPTH_RANGE: f32 = 2199.0;
const SHAFT_STEPS: u32 = 16u;
// how far the shafts are followed in the dense steps (the near cascade's reach, a little less)
const SHAFT_NEAR: f32 = 180.0;
// and, past it, in a few sparse ones to the far map's reach (the beams of a misty avenue, or
// a low sun between the blocks, reach farther than the first hundred metres)
const SHAFT_FAR_STEPS: u32 = 4u;
const SHAFT_REACH: f32 = 420.0;
// how far the sky's gaps are followed (the haze beyond that is the table's, lit as a whole) and in
// how many steps growing by a fixed ratio
const SKY_SHAFT_REACH: f32 = 7000.0;
const SKY_SHAFT_STEPS: u32 = 8u;

// Whether the sun reaches a point in the air (1) or a house or a crown stands between (0).
fn sun_reaches(x: vec3<f32>) -> f32 {
    let lp = camera.light_view_proj * vec4<f32>(x, 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (max(abs(uv.x - 0.5), abs(uv.y - 0.5)) < 0.48 && lp.z >= 0.0 && lp.z <= 1.0) {
        return textureSampleCompareLevel(t_shadow, s_shadow, vec2<f32>(uv.x * 0.5, uv.y), lp.z - 0.15 / SHAFT_DEPTH_RANGE);
    }
    let fp2 = camera.light_view_proj_far * vec4<f32>(x, 1.0);
    let fuv = vec2<f32>(fp2.x * 0.5 + 0.5, 0.5 - fp2.y * 0.5);
    if (fuv.x < 0.0 || fuv.x > 1.0 || fuv.y < 0.0 || fuv.y > 1.0 || fp2.z < 0.0 || fp2.z > 1.0) {
        return 1.0;
    }
    // (the far map takes the top of its texture, see shader.wgsl `shadow_pcf_far`)
    return textureSampleCompareLevel(t_shadow_far, s_shadow, vec2<f32>(fuv.x, fuv.y * (1.0 / (camera.shadow.y * camera.lamp_atlas.z))), fp2.z - 0.6 / SHAFT_DEPTH_RANGE);
}

fn sun_reaches_close(x: vec3<f32>) -> f32 {
    let lp = camera.light_view_proj_close * vec4<f32>(x, 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (camera.flags.w <= 0.0 || max(abs(uv.x - 0.5), abs(uv.y - 0.5)) >= 0.48 || lp.z < 0.0 || lp.z > 1.0) {
        return sun_reaches(x);
    }
    let s = select(1.0, camera.post.w, camera.post.w > 0.0);
    return textureSampleCompareLevel(t_shadow, s_shadow, vec2<f32>(0.5 + uv.x * s * 0.5, uv.y * s), lp.z - 0.05 / SHAFT_DEPTH_RANGE);
}

// Shafts of sunlight - crepuscular rays: the air between the camera and what it sees
// scatters the sun towards the eye wherever the sun reaches it, and not in the shadow of a
// house or a tree. The picture's air (`air` in enhanced_common.wgsl) holds that light as
// if the sun reached the air everywhere; along the first stretch of the ray, which the
// shadow maps cover, what lies in shadow is taken out again here (a negative radiance):
// lit and dark shafts through a hazy evening, a misty morning wood, a dusty street. The
// air scatters the way `air` has it - the weather's fog with g 0.55, the clear air's haze
// and dust forward with g 0.7 (its Rayleigh part is small near the ground).
fn sun_shafts(c: vec3<f32>, d: vec3<f32>, len: f32, jitter: f32) -> vec3<f32> {
    let s = camera.sun_dir.xyz;
    let e_sun = enh.sun.rgb;
    // (the clouds' shadow field: the shafts through a broken sky, followed to kilometres)
    let sky_gaps = enh.cloud_f1.w > 0.5 && s.z >= 0.06;
    let shadows = camera.shadow.x > 0.5;
    // (with the volumetric fog on, its march has the shadows in the air already:
    // volumetric_fog.wgsl, enh.vol_fog.x)
    if (enh.vol_fog.x > 0.5 || (!shadows && !sky_gaps) || s.z <= 0.0 || max(e_sun.r, e_sun.g) < 1e-4) {
        return vec3<f32>(0.0);
    }
    let end = min(len, SHAFT_NEAR);
    let dt = end / f32(SHAFT_STEPS);
    let far_len = max(min(len, SHAFT_REACH) - SHAFT_NEAR, 0.0);
    let dt_far = far_len / f32(SHAFT_FAR_STEPS);
    let cos_t = dot(s, d);
    let p_fog = hg_phase(cos_t, 0.55) * 0.9;
    let p_air = hg_phase(cos_t, 0.7);
    let h0 = c.z - enh.fog.z;
    // the clouds' part reaches on, out to the haze's own horizon: from the sparse steps'
    // end, in steps growing by a fixed ratio
    let sky_end = select(SHAFT_REACH, min(len, SKY_SHAFT_REACH), sky_gaps);
    // (the most the shadows could take out along the way, on the screen: where that is
    // too little to see - most of the picture, looking away from the sun - nothing is done)
    let most = (enh.fog.x * p_fog + enh.fog.w * p_air) * max(e_sun.r, e_sun.g) * max(end, select(0.0, sky_end, sky_gaps)) * enh.exposure.x;
    if (most < 0.004) {
        return vec3<f32>(0.0);
    }
    var shade = vec3<f32>(0.0);
    for (var i = 0u; i < SHAFT_STEPS + SHAFT_FAR_STEPS; i = i + 1u) {
        let far = i >= SHAFT_STEPS;
        if (far && far_len <= 0.0) {
            break;
        }
        let t = select((f32(i) + jitter) * dt, SHAFT_NEAR + (f32(i - SHAFT_STEPS) + jitter) * dt_far, far);
        let w = select(dt, dt_far, far);
        let x = c + d * t;
        var lit = 1.0;
        if (shadows) {
            lit = sun_reaches(x);
        }
        if (sky_gaps) {
            lit = lit * cloud_shadow_ratio(x);
        }
        // (a point the cloud lets more sun through than the camera's own place gets adds light)
        if (lit >= 0.999 && lit <= 1.001) {
            continue;
        }
        let h = max(x.z - enh.fog.z, 0.0);
        let s_fog = enh.fog.x * exp(-enh.fog.y * h);
        let s_air = enh.fog.w * exp(-h / 1500.0);
        let tau = layer_depth(enh.fog.x, enh.fog.y, h0, h, t) + layer_depth(enh.fog.w, 1.0 / 1500.0, h0, h, t);
        shade = shade + vec3<f32>((1.0 - lit) * (s_fog * p_fog + s_air * p_air) * exp(-tau) * w);
    }
    // the part beyond the shadow maps: only the clouds' shadow is known there
    if (sky_gaps && sky_end > SHAFT_REACH) {
        let r = pow(sky_end / SHAFT_REACH, 1.0 / f32(SKY_SHAFT_STEPS));
        var t0 = SHAFT_REACH;
        for (var i = 0u; i < SKY_SHAFT_STEPS; i = i + 1u) {
            let t1 = t0 * r;
            let t = t0 + (t1 - t0) * jitter;
            let x = c + d * t;
            let lit = cloud_shadow_ratio(x);
            if (lit < 0.999 || lit > 1.001) {
                let h = max(x.z - enh.fog.z, 0.0);
                let s_fog = enh.fog.x * exp(-enh.fog.y * h);
                let s_air = enh.fog.w * exp(-h / 1500.0);
                let tau = layer_depth(enh.fog.x, enh.fog.y, h0, h, t) + layer_depth(enh.fog.w, 1.0 / 1500.0, h0, h, t);
                shade = shade + vec3<f32>((1.0 - lit) * (s_fog * p_fog + s_air * p_air) * exp(-tau) * (t1 - t0));
            }
            t0 = t1;
        }
    }
    return -shade * e_sun;
}

// at half size: one ray per four pixels
@fragment
fn fs_fog_lamps(in: FogVsOut) -> @location(0) vec4<f32> {
    // (the texture's last row: how much of the sun is seen, worked out once for the frame
    // in its first texel)
    // (t_fog is not this pass's target: the row is the one under the half picture)
    if (i32(in.clip.y) >= i32(ceil(fp.size.y * 0.5))) {
        var seen = 0.0;
        if (i32(in.clip.x) == 0 && fp.size.z > 0.5 && camera.sun_dir.z > -0.02) {
            seen = sun_seen(camera.cam_pos.xyz, camera.sun_dir.xyz);
        }
        return vec4<f32>(seen, 0.0, 0.0, 0.0);
    }
    let r = pixel_ray(floor(in.clip.xy) * 2.0 + vec2<f32>(0.5));
    let q = vec2<i32>(in.clip.xy) & vec2<i32>(1);
    let ign = (f32(q.x * 2 + (q.x ^ q.y)) + 0.5) * 0.25;
    // (the lamps' halo and cones only from the mist's and fog's droplets: fp.size.w)
    let l = (lamp_airlight(camera.cam_pos.xyz, r.xyz, 0.0, r.w, ign, fp.size.w) + sun_shafts(camera.cam_pos.xyz, r.xyz, r.w, ign)) * enh.exposure.x;
    return vec4<f32>(clamp(l, vec3<f32>(-4000.0), vec3<f32>(4000.0)), r.w);
}

// How much of the sun's disc is seen (0..1): houses, trees and the bus in front of it
// are in the depth prepass (the clouds and the fog are in `enh.sun` already).
fn sun_seen(c: vec3<f32>, s: vec3<f32>) -> f32 {
    let r = enh.sun.w;
    let up0 = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(s.z) > 0.9);
    let a = normalize(cross(s, up0));
    let b = cross(s, a);
    var seen = 0.0;
    var n = 0.0;
    for (var k = 0; k < 13; k = k + 1) {
        var o = vec2<f32>(0.0);
        if (k > 0) {
            let ang = f32(k) * 2.3999632;
            o = vec2<f32>(cos(ang), sin(ang)) * sqrt(f32(k) / 12.0) * 0.9;
        }
        let dir = normalize(s + (a * o.x + b * o.y) * r);
        let clip = camera.view_proj * vec4<f32>(c + dir * 1000.0, 1.0);
        let ndc = clip.xy / max(clip.w, 1e-6);
        n = n + 1.0;
        if (clip.w <= 0.0 || abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
            // Beyond the picture's edge the prepass knows nothing: whether the eye stands in
            // the shadow of anything the sun's shadow maps hold (without shadows: seen).
            // What stood at the edge was taken before - a house or a pole reaching the edge
            // as the camera turned put out the whole veil at once, and sky put it back, a
            // sheet of light over the picture switched off and on.
            if (camera.shadow.x > 0.5) {
                seen = seen + sun_reaches_close(c + dir * 0.3);
            } else {
                seen = seen + 1.0;
            }
            continue;
        }
        let px = vec2<f32>((ndc.x * 0.5 + 0.5) * fp.size.x, (0.5 - ndc.y * 0.5) * fp.size.y);
        let dims = vec2<i32>(textureDimensions(t_depth));
        let q = clamp(vec2<i32>(px), vec2<i32>(0), dims - vec2<i32>(1));
        seen = seen + select(0.0, 1.0, textureLoad(t_depth, q, 0) <= 0.0);
    }
    return seen / n;
}

// Glare: the light a lens scatters round a bright source - the sun seen directly washes out
// what is right beside it. The picture is a camera's, and the player compares it with
// photographs: a good lens' stray light falls off as the near term of the CIE's glare
// function, 10/theta^3 per unit of the source's illuminance (theta in degrees, CIE
// 146:2002), a glow of a few degrees. The eye's own share - the wide veil of 5/theta^2 and
// more that the vitreous and the eye's wall add, its ciliary streaks and its lenticular
// halo - was drawn as well and made a white blot ten degrees round the sun of a clear day,
// "the sun is too large" (#2400): a photograph shows none of it.
fn sun_glare(dir: vec3<f32>) -> vec3<f32> {
    let s = camera.sun_dir.xyz;
    if (max(enh.sun.r, enh.sun.g) < 1e-5 || s.z < -0.02) {
        return vec3<f32>(0.0);
    }
    let cos_t = clamp(dot(dir, s), -1.0, 1.0);
    // (cos 100 degrees: past it there is no veil, and the arc cosine is not needed)
    if (cos_t < -0.17364818) {
        return vec3<f32>(0.0);
    }
    let theta = max(degrees(acos(cos_t)), 0.25);
    let seen = textureLoad(t_fog, vec2<i32>(0, i32(textureDimensions(t_fog).y) - 1), 0).r;
    if (seen <= 0.0) {
        return vec3<f32>(0.0);
    }
    return enh.sun.rgb * seen * (10.0 / (theta * theta * theta));
}

// at full size, added onto the picture: the four nearest half-size rays, each as far as
// it was worked out to a distance like this pixel's (the light in front of a near pole
// is not the light in front of the houses behind it)
@fragment
fn fs_fog_composite(in: FogVsOut) -> @location(0) vec4<f32> {
    let ray = pixel_ray(in.clip.xy);
    var air_light = vec3<f32>(0.0);
    if (fp.size.z < 1.5) {
        air_light = gathered_air_light(in.clip.xy, ray.w);
    }
    // the eye's glare from the sun, on top of everything (fp.size.z: on)
    var glare = vec3<f32>(0.0);
    if (fp.size.z > 0.5) {
        glare = sun_glare(ray.xyz) * enh.exposure.x;
    }
    return vec4<f32>(air_light + min(glare, vec3<f32>(4000.0)), 0.0);
}

fn gathered_air_light(at: vec2<f32>, len: f32) -> vec3<f32> {
    let hp = at * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(hp));
    let f = hp - floor(hp);
    // (the last row is the sun's, see `fs_fog_lamps`)
    let hd = vec2<i32>(textureDimensions(t_fog)) - vec2<i32>(1, 2);
    // (not one of the sixteen half-size rays holds any light: no lamp's cone and no shaft
    // here - most of a picture in clear air - and the weighted sum below would only add up
    // zeros, so it is not worked out. All sixteen, the whole footprint of the tent: a lit
    // ray at its edge still lights the pixel)
    var lit = false;
    for (var k = 0; k < 16; k = k + 1) {
        let o = vec2<i32>(k & 3, k >> 2) - vec2<i32>(1);
        lit = lit || any(textureLoad(t_fog, clamp(base + o, vec2<i32>(0), hd), 0).rgb != vec3<f32>(0.0));
    }
    if (!lit) {
        return vec3<f32>(0.0);
    }
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    // (a twig or a wire none of the four rays met: the one nearest in distance)
    var best = vec3<f32>(0.0);
    var best_d = 1e9;
    // (a tent over the 4 x 4 nearest: the edge of a headlight's beam in the haze, worked
    // out at half size, showed its 2 x 2 blocks under a plain bilinear one)
    for (var k = 0; k < 16; k = k + 1) {
        let o = vec2<i32>(k & 3, k >> 2) - vec2<i32>(1);
        let s = textureLoad(t_fog, clamp(base + o, vec2<i32>(0), hd), 0);
        let dx = abs(f32(o.x) - f.x);
        let dy = abs(f32(o.y) - f.y);
        let bil = max(1.0 - dx * 0.5, 0.0) * max(1.0 - dy * 0.5, 0.0);
        let dd = abs(s.a - len);
        // (a Gaussian of the distance difference, not an exponential: a road's rows differ by
        // a few per cent of its length from one half-size row to the next, and an exponential
        // weighted every row differently - the offsets of the staggered steps, which cancel
        // over the tent exactly when its weights are even, then came out as a stripe along
        // every other row of a bright beam. A step in depth, a pole against what is behind it,
        // still keeps the taps of the far side out.)
        let tol = (dd / (0.05 * len + 0.5)) * 1.2;
        let w = (bil + 1e-3) * exp(-tol * tol);
        sum = sum + s.rgb * w;
        wsum = wsum + w;
        if (dd < best_d) {
            best_d = dd;
            best = s.rgb * min(len / max(s.a, 0.1), 1.0);
        }
    }
    if (wsum < 1e-4) {
        return best;
    }
    return sum / max(wsum, 1e-6);
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

// The shadows in a lamp's beam (the fog pass: its camera uniform has the lamp shadow tiles and
// the far shadow map's texture, see `fog_lamps.wgsl`). The other shaders that share
// `lamp_air.wgsl` take `beam_shadow_none.wgsl`'s: no map to look up, the beam goes on.

// Which of the lamp shadow tiles (under the far shadow map, see `lamp_shadow_at` in the
// scene's lamps) belongs to lamp `li`: its record names it in `pos.w` (a multiple of 32 of the
// marks below zero, `light_marks`), or 0xffffffff for a lamp with none.
fn beam_tile(li: u32, l: PointLight) -> u32 {
    let tile = floor(max(-l.pos.w, 0.0) / 32.0);
    return select(0xffffffffu, u32(tile - 0.5), tile > 0.5);
}

// How much of a beam reaches a point of the air, in tile `k` of its lamp's map: 0 where a
// pole, a car or a wall stands between the lamp and the point, so that the fog shows the
// shadow in its beam as it shows the sun's in a shaft. A tap of the hardware's filtered
// compare; off the tile, or without a map, the beam goes on.
fn beam_reaches(k: u32, x: vec3<f32>) -> f32 {
    if (k >= 44u) {
        return 1.0;
    }
    let q = camera.lamp_view_proj[k] * vec4<f32>(x, 1.0);
    if (q.w <= 0.0) {
        return 1.0;
    }
    let ndc = q.xyz / q.w;
    if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0 || ndc.z >= 1.0 || ndc.z <= 0.0) {
        return 1.0;
    }
    let dims = vec2<f32>(1.0 / camera.shadow.y, camera.lamp_atlas.z);
    let tile = camera.lamp_atlas.x;
    let cols = u32(camera.lamp_atlas.y);
    let tuv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let tx = clamp(tuv * tile, vec2<f32>(1.5), vec2<f32>(tile - 1.5));
    let at = (vec2<f32>(f32(k % cols) * tile, dims.x + f32(k / cols) * tile) + tx) / dims;
    // (a little more off the occluder than a surface takes: the air has no surface to
    // acne, and the lamp's own housing must not shade its beam)
    return textureSampleCompareLevel(t_shadow_far, s_shadow, at, ndc.z - 0.0002);
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

