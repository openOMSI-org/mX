fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}

// Sky dome: the envir.cfg gradient textures (day / twilight / night), u = azimuth relative
// to the sun, v = elevation (0 zenith … 1 horizon), blended by the sun altitude.
struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    // Kept in lockstep with `CameraUniform` in lib.rs.  The sky does not use the
    // floating-origin reconstruction itself, but omitting this member shifts every
    // following field one vec4 earlier: vanilla then reads the light-grid as `sky`
    // weights (usually all zero) and clears/draws a black sky.
    world_origin: vec4<f32>,
    sun_dir: vec4<f32>,
    ambient: vec4<f32>,
    fog: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    light_grid: vec4<f32>,
    sky: vec4<f32>,          // x sun azimuth (rad), y day weight, z twilight weight, w night weight
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    clouds: vec4<f32>,       // x density 0..1, yz texture offset (wind drift)
    light_view_proj: mat4x4<f32>,
    light_view_proj_far: mat4x4<f32>,
    shadow: vec4<f32>,
    post: vec4<f32>,         // x enhanced, y time
    inside_a: vec4<f32>,
    inside_b: vec4<f32>,
    inside_c: vec4<f32>,
    flags: vec4<f32>,
    light_view_proj_close: mat4x4<f32>,
    wind: vec4<f32>,
    // Enhanced: the lamps' shadow maps (the tiles under the far map; zero where nothing is
    // drawn) and where the tiles lie: x a tile's side, y tiles to a row, z the far map's
    // texture height, w the number of tiles (`LampAtlas::uniform`)
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

// A hash of a lattice point, from its integer bits.
//
// It was the old `fract(sin(dot(q, k)) * 43758)` trick, and that is what put flat slabs across
// the top of the sky. `cloud_fbm` below walks `q` up by 2.1 an octave from a world coordinate
// that is kilometres wide, and once the argument of the `sin` reaches a few times 1e5 a
// 32-bit float can no longer say where the fractional part of it lands: the hash comes out
// constant over a whole lattice cell, so the noise reads as flat squares with hard edges in a
// grid that turns with the camera - and only where the layer is thin enough to see through,
// which is why it showed as a "seam" in the sky "only when there are clouds": the high thin
// layer is the only thing up there in a clear sky, and it is faint. Integer bits do not lose
// precision with distance.
fn hash2(q: vec2<f32>) -> f32 {
    let x = bitcast<u32>(i32(floor(q.x)));
    let y = bitcast<u32>(i32(floor(q.y)));
    // (integer arithmetic in WGSL wraps, so these multiplications are modular already)
    var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffu) / 65535.0;
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// Cloud density field: the map's cloud texture as the large shape, fractal noise as the
// small one, two layers at different heights drifting at different speeds.
fn cloud_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * vnoise(q);
        q = q * 2.1 + vec2<f32>(17.0, 9.0);
        a = a * 0.5;
    }
    return v;
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var t_day: texture_2d<f32>;
@group(1) @binding(1) var t_twilight: texture_2d<f32>;
@group(1) @binding(2) var t_night: texture_2d<f32>;
@group(1) @binding(3) var s_sky: sampler;
@group(1) @binding(4) var t_clouds: texture_2d<f32>;
@group(1) @binding(5) var s_repeat: sampler;

/// How far the cloud field reaches before it repeats (m): a cumulus is then half a
/// kilometre to two across.
const CLOUD_FIELD_TILE: f32 = 14000.0;

// The ground point under the sky at distance `t` along the view ray `d`, in the clouds' own
// frame (world coordinates modulo 70 km, lib.rs `CLOUD_ORIGIN_PERIOD`), so that the cloud
// field stays where it is when the floating render origin moves on.
fn cloud_ground(d: vec3<f32>, t: f32) -> vec2<f32> {
    return camera.cam_pos.xy + camera.world_origin.zw + d.xy * t;
}

// The cloud cover over ground point p (x, 0..1, its edges frayed by the billows), how tall
// the cloud there grows (y, 0..1) and the cover without the billows (z: the enhanced sky
// raises its rounded tops on that, the billows would stand on them as spikes). From the
// cloud field: its equalised shape (G) cut at 1 - the cover, the weather's own picture (R)
// nudging where the clouds gather, billows (B) fraying the edges near by.
fn cloud_cover_at(p: vec2<f32>, lod: f32) -> vec3<f32> {
    let uv = p / CLOUD_FIELD_TILE + camera.clouds.yz * (2500.0 / CLOUD_FIELD_TILE);
    let t = textureSampleLevel(t_clouds, s_repeat, uv, lod);
    let thr = 1.0 - clamp(camera.clouds.x, 0.0, 1.0);
    let smooth_shape = t.g + (t.r - 0.5) * 0.15;
    var shape = smooth_shape;
    // the billows only where they are big enough to see
    let fray = 1.0 - smoothstep(2.0, 5.0, lod);
    if (fray > 0.0) {
        let detail = textureSampleLevel(t_clouds, s_repeat, uv * 3.7 + vec2<f32>(0.31, 0.73), lod + 1.9).b;
        shape = shape + (detail - 0.5) * 0.2 * fray;
    }
    return vec3<f32>(clamp((shape - thr) / 0.14, 0.0, 1.0), t.a, clamp((smooth_shape - thr) / 0.14, 0.0, 1.0));
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    // the dome rides with the camera; pushed onto the far plane (0 with reversed Z), the
    // depth the pass starts from: its depth test (equal or nearer than that) keeps it to the
    // pixels nothing was drawn into, when it is drawn after the scenery's depth was laid
    // (main_pass.rs `encode_sky`)
    let wp = camera.cam_pos.xyz + pos * 4000.0;
    var clip = camera.view_proj * vec4<f32>(wp, 1.0);
    clip.z = 0.0;
    out.clip = clip;
    out.dir = pos;
    return out;
}

// The classic sky's weather (lib.rs `VanillaSky`): the cloud type's own texture and
// cloud: x height H (m over the map's zero), y the type's size (m a tile; 0: no clouds),
// zw the wind's drift (m); haze: x the weather's fog range (m), y the visibility (m),
// z 1 for an overcast type, w the render origin's height.
struct VanillaSkyUniform {
    cloud: vec4<f32>,
    haze: vec4<f32>,
};
@group(1) @binding(9) var t_vclouds: texture_2d<f32>;
@group(1) @binding(10) var<uniform> vsky: VanillaSkyUniform;
@group(1) @binding(11) var t_sun: texture_2d<f32>;

fn vanilla_sun(d: vec3<f32>) -> vec3<f32> {
    let s = normalize(camera.sun_dir.xyz);
    let c = dot(d, s);
    let right = normalize(select(vec3<f32>(s.y, -s.x, 0.0), vec3<f32>(1.0, 0.0, 0.0), length(s.xy) < 1e-4));
    let up = cross(right, s);
    let p = d * (3.0 / max(c, 1e-4));
    let uv = vec2<f32>((dot(p, right) + 0.0803) / 0.1606, (0.0821 - dot(p, up)) / 0.1606);
    let t = srgb_encode(textureSampleLevel(t_sun, s_sky, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb);
    let inside = c > 0.0 && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    return select(vec3<f32>(0.0), t * vec3<f32>(1.0, 1.0, 0.0), inside);
}

// The v of the sky textures towards elevation `e` (rad), as Omsi.exe's dome
// (helper\skybox.x) has it: rings at the horizon, at 45 degrees and at the zenith with v
// 0.9833, 0.4868 and 0.0236, flat bands between them along which v runs linearly - along
// the chord, not with the angle - and the horizon's v all the way down below it.
fn dome_v(e: f32) -> f32 {
    if (e <= 0.0) {
        return 0.9833;
    }
    let upper = e > 0.7853982;
    let e0 = select(0.0, 0.7853982, upper);
    let e1 = select(0.7853982, 1.5707963, upper);
    let v0 = select(0.9833, 0.4868, upper);
    let v1 = select(0.4868, 0.0236, upper);
    // (where the ray at e meets the chord from the ring at e0 to the one at e1)
    let t = sin(e - e0) / ((sin(e1) - sin(e0)) * cos(e) - (cos(e1) - cos(e0)) * sin(e));
    return mix(v0, v1, clamp(t, 0.0, 1.0));
}

// The haze Omsi.exe lays over its sky (0x5d8e98): the dome again with Texture\nebel.tga,
// white with an alpha running linearly over its 32 rows from 0 at the zenith to 1 at the
// horizon, in the fog colour, its alpha moved by the fog range f and the cloud height H:
// down by 1 - 4 asin(H/f)/pi where f > H sqrt 2 (no haze at all under a clear sky, H 0),
// up by 1 - 4 acos(H/f)/pi below that, and the whole sky fog where f <= H.
fn vanilla_haze(v: f32) -> f32 {
    let f = vsky.haze.x;
    let h = vsky.cloud.x;
    if (f <= 0.0) {
        return 0.0;
    }
    if (f <= h) {
        return 1.0;
    }
    let tex_a = clamp((v * 32.0 - 0.5) / 31.0, 0.0, 1.0);
    let q = h / f;
    if (q < 0.70710678) {
        return clamp(tex_a - (1.0 - 4.0 * asin(q) / 3.14159265), 0.0, 1.0);
    }
    return clamp(tex_a + (1.0 - 4.0 * acos(q) / 3.14159265), 0.0, 1.0);
}

// The classic cloud layer towards d: rgb (in the sky's own terms, `fog` its fog colour) and
// how much it covers. Omsi.exe (0x754e44) draws helper\clouds.o3d, a flat cone over the
// camera - its apex H over the map's zero, its rim on the zero 9.363 H away - with the cloud
// type's texture laid on the ground plan, one tile every `size` metres, moved by the wind;
// its colour is the texture times the fog colour, its alpha the texture's (an `ovc` deck is
// opaque), and the fixed-function fog takes it into the fog colour by its depth in the
// view over the visibility.
fn vanilla_clouds(d: vec3<f32>, fog: vec3<f32>, tex: vec4<f32>, r: f32, encoded: bool) -> vec4<f32> {
    let hl = max(length(d.xy), 1e-4);
    // (the hit point lies along d, r / hl from the camera; its depth along the view's axis)
    let fwd = normalize(cross(camera.cam_up.xyz, camera.cam_right.xyz));
    let depth = r / hl * dot(d, fwd);
    let vis = max(vsky.haze.y, 1.0);
    let k = clamp(1.0 - depth / vis, 0.0, 1.0);
    var rgb = select(tex.rgb, srgb_encode(tex.rgb), encoded) * fog;
    rgb = mix(fog, rgb, k);
    return vec4<f32>(rgb, select(tex.a, 1.0, vsky.haze.z > 0.5));
}

// How far (over the ground plan) the ray d meets the cloud cone, and whether it does.
fn vanilla_cloud_reach(d: vec3<f32>) -> vec2<f32> {
    let h = vsky.cloud.x;
    let h0 = camera.cam_pos.z + vsky.haze.w;
    let hl = max(length(d.xy), 1e-4);
    let rim = 9.363 * h;
    let den = d.z / hl + 1.0 / 9.363;
    if (vsky.cloud.y <= 0.0 || h <= h0 || h <= 0.0) {
        return vec2<f32>(rim, 0.0);
    }
    // Below the rim the cone's skirt (helper\clouds.o3d goes on down and in under the
    // horizon) lies so far away that the fog takes it whole: nothing of the clouds shows
    // there. Every ray below the rim drawn at the rim instead smeared the texture's one row
    // into upright streaks above the horizon, wherever the ground did not reach that far.
    if (den <= 0.0) {
        return vec2<f32>(rim, 0.0);
    }
    let r = (h - h0) / den;
    if (r > rim) {
        return vec2<f32>(rim, 0.0);
    }
    return vec2<f32>(r, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = normalize(in.dir);
    // the dome turned so that the texture's column 0.5036 faces the sun, u growing
    // anticlockwise (from above) away from it (0x5d92a0: the dome rotated by the sun's
    // azimuth)
    let az = atan2(d.x, d.y);
    let u = 0.503645 + (camera.sky.x - az) / 6.2831853;
    let elev = asin(clamp(d.z, -1.0, 1.0));
    let v = dome_v(elev);
    let uv = vec2<f32>(u, v);
    // (Omsi.exe blends two of its pictures by the sun's height, see `Daylight::sky_weights`;
    // the sky textures have no mip levels, so the seam of u behind the sun is not seen)
    let sky = textureSample(t_day, s_sky, uv).rgb * camera.sky.y + textureSample(t_twilight, s_sky, uv).rgb * camera.sky.z + textureSample(t_night, s_sky, uv).rgb * camera.sky.w;
    // the cloud texture on the ground plan where the ray meets the cone, sampled outside
    // any branch (its mip levels by the screen's derivatives)
    let reach = vanilla_cloud_reach(d);
    let hl = max(length(d.xy), 1e-4);
    let ground = camera.cam_pos.xy + camera.world_origin.zw + d.xy / hl * reach.x;
    let size = max(vsky.cloud.y, 1.0);
    let cuv = (ground + vsky.cloud.zw) / size + vec2<f32>(0.5, -0.5) * (vsky.cloud.x / size);
    let ctex = textureSample(t_vclouds, s_repeat, vec2<f32>(cuv.x, cuv.y));
    // Vanilla blends in the 8-bit sRGB terms of Omsi.exe's fixed-function stages,
    // Vanilla+ in linear light
    let encoded = camera.sky_color.w > 0.5;
    let fog = camera.fog.xyz;
    let lit = min(srgb_encode(sky) + vanilla_sun(d), vec3<f32>(1.0));
    var col = select(srgb_decode(lit), lit, encoded);
    col = mix(col, fog, vanilla_haze(v));
    if (reach.y > 0.5) {
        let c = vanilla_clouds(d, fog, ctex, reach.x, encoded);
        col = mix(col, c.rgb, c.a);
    }
    if (encoded) {
        return vec4<f32>(srgb_decode(col), 1.0);
    }
    return vec4<f32>(col, 1.0);
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

// Enhanced graphics: the sky - the physical sky table, clouds lit by the sun and the sky,
// the sun's disc and the air - and the reflection probe drawn from it.

// Clouds: a ray-marched volume on the curved Earth, built the way Guerrilla's Horizon Zero
// Dawn and Nubis clouds are (and Frostbite's and Unreal's after them):
// * the layer is a spherical shell over the ground of an Earth of 6371 km (the weather
//   model's condensation level up to where its convection stops; 1400-2800 m for the stock
//   weather), so the clouds sink to the horizon and thin out into the distance;
// * the weather map (clouds.rs) says where the clouds stand - its field cut by the
//   weather's cover, so a cover of 0.3 leaves some 30 % of the sky under cloud - what type
//   they are there and how tall they grow; the weather's type (heaps, the flat sheet of an
//   overcast deck, a storm's towers with their anvil) gives each its height profile;
// * the base volume (Perlin-Worley) is the body of every cloud, cut by that profile and
//   the local coverage; the detail volume, twisted by the curl map, erodes its edges - into
//   wisps under a cloud, into cauliflower on its top;
// * the light: the key light (the sun, by night the moon) through the cloud towards it,
//   in growing steps, with Hillaire's octaves of multiple scattering, a two-lobe
//   Henyey-Greenstein phase (a silver lining towards the sun, a little back-scatter away
//   from it) and Schneider's powder term (the dark creases of a cloud seen away from the
//   sun); the sky's light from above and the ground's from below by height; integrated per
//   step the energy-conserving Frostbite way; far clouds take on the air's colour.
// The window's picture draws them per pixel at a quarter of its size and accumulates them
// over the frames (clouds_ss.wgsl); the sky cube (the reflection probe's source) marches
// a cheaper version of them.
fn cloud_bottom() -> f32 { return enh.layers.x; }
fn cloud_top() -> f32 { return enh.layers.y; }
// The storm's activity: the layer is a cumulonimbus, dense, dark, with its anvil.
fn storm_cb() -> f32 { return enh.layers.w; }
const EARTH_R: f32 = 6371000.0;
// Extinction per metre of the densest cloud (a cumulus' 0.05 - an optical depth of some
// 20 through one a few hundred metres deep).
const CLOUD_SIGMA: f32 = 0.05;
// How far the pictures reach before they repeat (m): all of them divide the clouds' frame
// of 70 km (lib.rs `CLOUD_ORIGIN_PERIOD`), or the clouds would jump where it wraps.
const WEATHER_PERIOD: f32 = 35000.0;
const BASE_PERIOD: f32 = 5000.0;
const DETAIL_PERIOD: f32 = 625.0;
const CURL_PERIOD: f32 = 3500.0;
// How far the curl map twists the detail (in the detail's periods).
const CURL_STRENGTH: f32 = 0.35;
// How much faster than the weather map the base volume and the detail drift: the clouds
// change their shape slowly as they go (cloud_field.rs `BASE_DRIFT` alike).
const BASE_DRIFT: f32 = 1.12;
const DETAIL_DRIFT: f32 = 1.4;
// How much of a cloud the detail erodes at most, and its mean (which the cheap march takes
// off instead).
const DETAIL_ERODE: f32 = 0.32;
const DETAIL_MEAN: f32 = 0.5;
// How far the weather map's field spreads the cover from place to place.
const COVER_SPREAD: f32 = 1.1;
// The droplets' asymmetry: how far forward they scatter (Mie, cumulus droplets, visible light).
const CLOUD_G: f32 = 0.85;
const CLOUD_MAX_DIST: f32 = 60000.0;
// Single scattering and three octaves hold only part of the light a cloud scatters on
// inside it; a cumulus's sunlit side is about as bright as white paper in the sun, E/π,
// which this factor brings it to.
const CLOUD_MS_GAIN: f32 = 2.4;
// The powder term's depth scale (1/extinction): how far into a cloud its sun-facing edge
// keeps its full brightness seen from the side away from the sun.
const CLOUD_POWDER: f32 = 140.0;

@group(1) @binding(6) var t_cloud_weather: texture_2d<f32>;
@group(1) @binding(7) var t_cloud_detail: texture_3d<f32>;
@group(1) @binding(8) var s_cloud: sampler;
// (the passes that march the clouds only: the sky cube's and the screen's)
@group(0) @binding(20) var t_cloud_base: texture_3d<f32>;
@group(0) @binding(21) var t_cloud_curl: texture_2d<f32>;

fn linearstep(a: f32, b: f32, v: f32) -> f32 {
    return clamp((v - a) / (b - a), 0.0, 1.0);
}

// Height over the curved ground of the point t metres along d from the camera.
fn cloud_height(d: vec3<f32>, t: f32) -> f32 {
    return camera.cam_pos.z + d.z * t + t * t * (1.0 - d.z * d.z) / (2.0 * EARTH_R);
}

// Distance along d at which the ray reaches height h over the curved ground (the first
// crossing ahead; -1 when it never does).
fn cloud_shell(d: vec3<f32>, h: f32) -> f32 {
    let a = (1.0 - d.z * d.z) / (2.0 * EARTH_R);
    let b = d.z;
    let c = camera.cam_pos.z - h;
    if (a < 1e-12) {
        return select(-1.0, -c / b, abs(b) > 1e-6 && -c / b > 0.0);
    }
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return -1.0;
    }
    let q = -0.5 * (b + select(-1.0, 1.0, b >= 0.0) * sqrt(disc));
    let r0 = q / a;
    let r1 = c / q;
    let lo = min(r0, r1);
    let hi = max(r0, r1);
    if (lo > 0.0) {
        return lo;
    }
    return select(-1.0, hi, hi > 0.0);
}

// Where along d the ray is inside the layer: from x to y (x < 0: it never is).
fn cloud_span(d: vec3<f32>) -> vec2<f32> {
    let z = camera.cam_pos.z;
    let tb = cloud_shell(d, cloud_bottom());
    let tt = cloud_shell(d, cloud_top());
    var t0 = -1.0;
    var t1 = -1.0;
    if (z < cloud_bottom()) {
        t0 = tb;
        t1 = tt;
    } else if (z < cloud_top()) {
        t0 = 0.0;
        t1 = select(tt, tb, d.z < 0.0 && tb > 0.0);
    } else {
        t0 = tt;
        t1 = select(t0 + 8000.0, tb, tb > 0.0);
    }
    if (t0 < 0.0 || t0 > CLOUD_MAX_DIST) {
        return vec2<f32>(-1.0, -1.0);
    }
    if (t1 <= t0) {
        t1 = t0 + 8000.0;
    }
    return vec2<f32>(t0, min(t1, t0 + 15000.0));
}

// The sun's irradiance at height z (as `cloud_height` gives it) in the cloud layer:
// between the base, middle and top of `enh.cloud_sun`.
fn cloud_sun_at(z: f32) -> vec3<f32> {
    let t = clamp((z - cloud_bottom()) / (cloud_top() - cloud_bottom()), 0.0, 1.0) * 2.0;
    return select(mix(enh.cloud_sun[1].rgb, enh.cloud_sun[2].rgb, t - 1.0), mix(enh.cloud_sun[0].rgb, enh.cloud_sun[1].rgb, t), t < 1.0);
}

// The weather's clouds as a whole (cloud_field.rs `CloudParams` alike): the cover; how much
// of it is a closed deck (a full cover, or a sky that rain falls from); the storm; the type
// (1 heaps, a deck's flat sheet ~0.3, thicker as a raining nimbostratus, a storm's towers);
// how much the detail erodes (less on a deck: a textured grey underside, not torn rags) and
// how dense they are (a raining deck and a storm hold more water).
struct CloudSky {
    cover: f32,
    deck: f32,
    storm: f32,
    kind: f32,
    erode: f32,
    dens: f32,
};

fn cloud_sky() -> CloudSky {
    let cover = clamp(camera.clouds.x, 0.0, 1.0);
    let deck = max(smoothstep(0.8, 1.0, cover), smoothstep(0.55, 1.0, enh.weather.w));
    let rain = enh.weather.z;
    let storm = storm_cb();
    let kind = max(mix(1.0, 0.3 + 0.35 * rain, deck), storm);
    return CloudSky(cover, deck, storm, kind, DETAIL_ERODE * (1.0 - 0.3 * deck), 1.0 + 1.2 * max(deck * (0.4 + 0.6 * rain), storm));
}

fn cloud_drift() -> vec2<f32> {
    return camera.clouds.yz * 2500.0;
}

// The weather's own cloud picture over tens of kilometres: a little more or less cover.
fn cloud_picture_bias(p: vec2<f32>) -> f32 {
    let uv = p / CLOUD_FIELD_TILE + camera.clouds.yz * (2500.0 / CLOUD_FIELD_TILE);
    return (textureSampleLevel(t_clouds, s_repeat, uv * 0.35, 5.0).g - 0.5) * 0.16;
}

// How much of its layer a cloud type fills at height fraction h (cloud_field.rs
// `cloud_profile`): a sheet low in the layer with a flat top; heaps with a flat base and a
// rounded top that narrows (the coverage cut takes ever more of a heap as the profile falls);
// as tall as the weather map's `tall` lets them grow, a storm's to the top.
fn cloud_profile(h: f32, kind: f32, tall: f32) -> f32 {
    let top = mix(0.16, 1.0, kind) * mix(0.6, 1.0, tall);
    let bottom = linearstep(0.0, 0.05 + 0.05 * kind, h);
    return bottom * (1.0 - linearstep(top * mix(0.75, 0.35, kind), top, h));
}

// The cloud's density (0..1) at p (h: 0 at the base, 1 at the top of the layer); `lod` the
// base volume's level for the pixel's footprint; without `fine` the detail is taken off on
// average (the sky cube, and the light's far steps).
fn cloud_density(p: vec3<f32>, h: f32, lod: f32, sky: CloudSky, fine: bool) -> f32 {
    if (h <= 0.0 || h >= 1.0) {
        return 0.0;
    }
    let w = textureSampleLevel(t_cloud_weather, s_cloud, (p.xy + cloud_drift()) / WEATHER_PERIOD, max(lod - 0.5, 0.0));
    let f = w.r * 0.7 + w.a * 0.3;
    var cov = clamp(sky.cover + (f - 0.5) * COVER_SPREAD + cloud_picture_bias(p.xy), 0.0, 1.0);
    cov = mix(cov, 1.0, sky.deck);
    if (cov <= 0.001) {
        return 0.0;
    }
    let kind = clamp(sky.kind + (w.g - 0.5) * 0.5 * (1.0 - sky.deck), 0.0, 1.0);
    let prof = cloud_profile(h, kind, max(w.b, sky.storm));
    if (prof <= 0.0 && sky.storm < 0.05) {
        return 0.0;
    }
    let bxy = p.xy + cloud_drift() * BASE_DRIFT;
    let bz = h * (cloud_top() - cloud_bottom());
    let b = textureSampleLevel(t_cloud_base, s_cloud, vec3<f32>(bxy, bz) / BASE_PERIOD, lod).r;
    // (a deck keeps the base volume's picture in its underside, but no holes)
    let base = mix(b, 0.5 + 0.5 * b, sky.deck);
    var m = clamp((base * prof - (1.0 - cov)) / max(cov, 1e-3), 0.0, 1.0) * cov;
    if (sky.storm > 0.05) {
        // the storm's anvil: under the tropopause the tower spreads into a flat sheet of ice,
        // wide as the coarse picture of the weather round it, which the upper wind pulls
        // downwind; thin and torn at its edges
        let wide = textureSampleLevel(t_cloud_weather, s_cloud, (p.xy + cloud_drift() + vec2<f32>(1500.0, 800.0) * (h - 0.8)) / WEATHER_PERIOD, lod + 3.0).r;
        let spread = smoothstep(0.35, 0.7, mix(wide, f, 0.5));
        let anvil = sky.storm * smoothstep(0.74, 0.86, h) * (1.0 - smoothstep(0.9, 1.0, h)) * spread * spread;
        m = max(m, anvil * (0.15 + 0.85 * b));
    }
    if (m <= 0.0) {
        return 0.0;
    }
    var e = DETAIL_MEAN * sky.erode;
    if (fine) {
        // the detail, twisted by the curl map (more under a cloud, where the wind tears it)
        let curl = textureSampleLevel(t_cloud_curl, s_cloud, bxy / CURL_PERIOD, max(lod + 0.5, 0.0)).rg * 2.0 - vec2<f32>(1.0);
        let q = vec3<f32>(p.xy + cloud_drift() * DETAIL_DRIFT, bz) / DETAIL_PERIOD + vec3<f32>(curl * CURL_STRENGTH * (1.0 - h), 0.0);
        // (the detail texel is half the base's: one level coarser; read too sharp, a
        // repeating 625 m volume drew moire arcs over the sky)
        let dl = max(lod + 1.0, 0.0);
        let dn = textureSampleLevel(t_cloud_detail, s_cloud, q, dl).r;
        // (wispy at the base, billowy higher up; its mean where it can no longer be resolved)
        let ed = mix(dn, 1.0 - dn, clamp(h * 4.0, 0.0, 1.0));
        e = mix(ed, DETAIL_MEAN, smoothstep(1.5, 3.0, dl)) * sky.erode;
    }
    return clamp((m - e) / (1.0 - e), 0.0, 1.0);
}

// Extinction (1/m) at p.
fn cloud_sigma(p: vec3<f32>, h: f32, lod: f32, sky: CloudSky, fine: bool) -> f32 {
    return cloud_density(p, h, lod, sky, fine) * CLOUD_SIGMA * sky.dens;
}

// The clouds along d: the light they scatter towards the camera (radiance, the air's
// colour taken on in the distance from `below`, the sky behind them), how much of the sky
// behind gets through, and how far they are seen (their transmittance-weighted depth).
struct CloudMarch {
    light: vec3<f32>,
    trans: f32,
    dist: f32,
};

// The march: `steps` through the layer, starting `jitter` (0..1) into the first; `pix` a
// pixel's angular size; without `fine` the detail is left out.
fn cloud_march(d: vec3<f32>, below: vec3<f32>, steps: i32, jitter: f32, pix: f32, fine: bool) -> CloudMarch {
    var out = CloudMarch(vec3<f32>(0.0), 1.0, 1e6);
    let sky = cloud_sky();
    if (sky.cover <= 0.001) {
        return out;
    }
    let span = cloud_span(d);
    if (span.x < 0.0) {
        return out;
    }
    let t0 = span.x;
    let t1 = span.y;
    out.dist = t0;
    // (more steps for the long, low rays: a fixed count across up to 15 km slices the layer
    // into horizontal planes, drawn as rings round the zenith and bands at the horizon)
    let n = i32(clamp(f32(steps) * (1.0 + 2.0 * (1.0 - abs(d.z))), f32(steps), f32(steps) * 3.0));
    let ds = (t1 - t0) / f32(n);
    // the key light: the sun, and the moon once the sun has gone from the clouds too
    let night = camera.sun_dir.z < -0.12;
    let ld = select(normalize(camera.sun_dir.xyz), normalize(enh.moon.xyz), night);
    let cos_l = dot(d, ld);
    // (the phase of each octave: weaker and flatter, Hillaire 2016)
    var phase: array<f32, 3>;
    var c = 1.0;
    for (var o = 0; o < 3; o = o + 1) {
        phase[o] = mix(hg_phase(cos_l, 0.8 * c), hg_phase(cos_l, -0.3 * c), 0.3);
        c = c * 0.5;
    }
    let sky_top = sh_irradiance(vec3<f32>(0.0, 0.0, 1.0)) / PI;
    let ground = sh_irradiance(vec3<f32>(0.0, 0.0, -1.0)) / PI;
    let deep = max(sky.deck, sky.storm);
    let shade = mix(0.20, 0.03, deep);
    let tau_deck = max(enh.clouds_ext2.x, 1.0);
    // (a texel of the base volume, m)
    let texel = BASE_PERIOD / f32(textureDimensions(t_cloud_base).x);
    var trans = 1.0;
    var acc = vec3<f32>(0.0);
    var hit = 0.0;
    var hit_w = 0.0;
    var t = t0 + ds * jitter;
    for (var i = 0; i < n; i = i + 1) {
        let p = vec3<f32>(cloud_ground(d, t), cloud_height(d, t));
        let h = (p.z - cloud_bottom()) / (cloud_top() - cloud_bottom());
        // (the step's length counts as the footprint too: a long step through a sharp volume
        // reads it as contour lines)
        let lod = log2(max(max(t * pix, 0.5 * ds) / texel, 1.0));
        let sigma = cloud_sigma(p, h, lod, sky, fine);
        if (sigma > 1e-6) {
            // the key light reaching p through the cloud towards it, in steps growing 1.9
            // times (2 km in all; the detail in the first three only)
            var od = 0.0;
            var ls = 40.0;
            var lt = 0.0;
            for (var k = 0; k < 6; k = k + 1) {
                let q = p + ld * (lt + fract(jitter + 0.61803399 * f32(k + 1)) * ls);
                let hq = (q.z - cloud_bottom()) / (cloud_top() - cloud_bottom());
                if (hq >= 1.0 || hq <= 0.0) {
                    break;
                }
                od = od + cloud_sigma(q, hq, lod + 0.5 + f32(k) * 0.5, sky, fine && k < 3) * ls;
                lt = lt + ls;
                ls = ls * 1.9;
            }
            let key = select(cloud_sun_at(p.z), enh.moon_disc.rgb, night);
            var direct = vec3<f32>(0.0);
            var a = 1.0;
            var b = 1.0;
            for (var o = 0; o < 3; o = o + 1) {
                direct = direct + key * (a * phase[o] * exp(-od * b));
                a = a * 0.5;
                b = b * 0.4;
            }
            // the powder term: seen away from the light, a cloud's thin outer edge facing
            // it has scattered less light back out than its depths
            let powder = 1.0 - exp(-sigma * CLOUD_POWDER);
            direct = direct * mix(powder, 1.0, 0.35 + 0.65 * clamp(0.5 + 0.5 * cos_l, 0.0, 1.0));
            // The sky's light from above (and the ground's from below, low in the layer)
            // gets down through the cloud over this point by scattering: the two-stream
            // 1 / (1 + 3/4 (1 - g) od) rather than Beer's beam, so that a deck stays twice
            // as bright where it is half as deep and its underside keeps a picture of its
            // own; the depth towards the light stands for the depth overhead. The floor is
            // what comes in sideways through the gaps. Under a shut deck the sky's light is
            // already what the deck lets through (the sky model's, `clouds_ext2.x` its depth):
            // there the depth over this point only makes it brighter or darker than the mean.
            let through = 1.0 / (1.0 + 0.75 * (1.0 - CLOUD_G) * od);
            let rel = clamp((1.0 + 0.75 * (1.0 - CLOUD_G) * tau_deck) / (1.0 + 0.75 * (1.0 - CLOUD_G) * od * max(ld.z, 0.15)), 0.35, 2.0);
            let amb = mix(ground * 0.45 + sky_top * 0.55, sky_top * 1.1, clamp(h * 1.4, 0.0, 1.0)) * mix(shade + (1.0 - shade) * through, rel, deep);
            let light = direct * CLOUD_MS_GAIN + amb;
            // Frostbite: the light scattered over the step, dimmed by the cloud before it
            let dt = exp(-sigma * ds);
            acc = acc + trans * light * (1.0 - dt);
            hit = hit + t * trans * (1.0 - dt);
            hit_w = hit_w + trans * (1.0 - dt);
            trans = trans * dt;
            if (trans < 0.005) {
                trans = 0.0;
                break;
            }
        }
        t = t + ds;
    }
    // far clouds take on the colour of the air in front of them, and beyond 40 km fade out
    let dist = select(t0, hit / max(hit_w, 1e-4), hit_w > 1e-4);
    let aerial = 1.0 - exp(-dist / 22000.0);
    let fade = (1.0 - smoothstep(40000.0, CLOUD_MAX_DIST, dist)) * smoothstep(-0.01, 0.02, d.z + select(0.0, 1.0, camera.cam_pos.z > cloud_bottom()));
    out.light = mix(acc, below * (1.0 - trans), aerial) * fade;
    out.trans = 1.0 - (1.0 - trans) * fade;
    out.dist = dist;
    return out;
}

// Fractal value noise with its octaves faded out where they get finer than the pixel's
// footprint `fp` (in p's units): left in, the fine octaves of a distant layer alias into a
// marbled pattern of contour lines; faded to their mean, the layer softens into the haze it
// is at that distance.
fn hc_fbm(p: vec2<f32>, fp: f32, octaves: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    var f = fp;
    for (var i = 0; i < octaves; i = i + 1) {
        let keep = 1.0 - smoothstep(0.1, 0.35, f);
        v = v + a * mix(0.5, vnoise(q), keep);
        q = q * 2.03 + vec2<f32>(17.0, 9.0);
        f = f * 2.03;
        a = a * 0.5;
    }
    return v / (1.0 - 2.0 * a);
}

// The high layers over the cumulus, analytic (drawn at the picture's own resolution, `pix`
// a pixel's angular size): altocumulus, a sheet of cells at 4 km (the middle levels'
// moisture ahead of a front), and cirrus, thin wisps of ice at 8 km drawn out by the upper
// wind. Both are sparse: cirrus in patches tens of kilometres apart, many skies without
// any; thicker and wider spread as a front's veil comes (the veil's milky light itself is in
// the sky table, atmosphere.rs). rgb: the sky `bg` with them over it, a: how much they
// cover. A shut deck hides them.
fn high_clouds(d: vec3<f32>, bg: vec3<f32>, pix: f32) -> vec4<f32> {
    let sky = cloud_sky();
    let closed = max(sky.deck, smoothstep(0.85, 1.0, sky.cover));
    var col = bg;
    var cover = 0.0;
    if (d.z <= 0.0 || closed >= 0.999) {
        return vec4<f32>(col, 0.0);
    }
    let sd = normalize(camera.sun_dir.xyz);
    let cos_sun = dot(d, sd);
    let sky_top = sh_irradiance(vec3<f32>(0.0, 0.0, 1.0)) / PI;
    let drift = cloud_drift();
    // altocumulus
    let alto_cover = enh.layers.z * (1.0 - closed) * smoothstep(0.03, 0.2, d.z);
    if (alto_cover > 0.01 && camera.cam_pos.z < 4000.0) {
        let t_alto = (4000.0 - camera.cam_pos.z) / max(d.z, 0.03);
        let pa = cloud_ground(d, t_alto) + drift * 1.4;
        // (a pixel's footprint on the sheet, m: stretched along the view towards the horizon)
        let fa = t_alto * pix / max(d.z, 0.05);
        // (its cells: the detail volume's billows on a slice, at the mip level of the
        // footprint, in patches the weather map's small field lays out)
        let rolls = textureSampleLevel(t_cloud_weather, s_cloud, pa / 9000.0, log2(max(fa / (9000.0 / 512.0), 1.0))).a;
        let cl = log2(max(fa / (1400.0 / 32.0), 1.0));
        let cells = 1.0 - textureSampleLevel(t_cloud_detail, s_cloud, vec3<f32>(pa / 1400.0, 0.37), cl).r;
        let dens = rolls * 0.6 + cells * 0.4;
        let edge = 0.1 + 0.1 * min(cl, 3.0);
        let thick = smoothstep(1.0 - alto_cover * 0.9 - edge, 1.0 - alto_cover * 0.9 + edge, dens);
        let sun = mix(enh.cloud_sun[2].rgb, enh.cloud_sun[3].rgb, 0.4);
        let lit = sun * (0.10 + 0.55 * hg_phase(cos_sun, 0.7)) / PI;
        let under = mix(1.0, 0.55, thick * thick);
        let alto_col = (lit * mix(0.6, 1.0, cells) + sky_top * 0.85) * under;
        let alto_a = thick * 0.75 * (1.0 - smoothstep(25000.0, 70000.0, t_alto));
        col = mix(col, alto_col, alto_a);
        cover = alto_a;
    }
    // cirrus: how much of the sky it may take - the veil of a coming front, and a few
    // wisps of fair weather over the heaps (none in a clear sky's dry air)
    let veil = enh.cloud_sun[0].w;
    let amount = clamp(0.7 * veil + 0.45 * smoothstep(0.1, 0.45, sky.cover) * (1.0 - sky.deck), 0.0, 1.0) * (1.0 - closed);
    var hi_cover = 0.0;
    if (amount > 0.01 && camera.cam_pos.z < 8000.0) {
        let t_hi = (8000.0 - camera.cam_pos.z) / max(d.z, 0.02);
        let ph = cloud_ground(d, t_hi) + drift * 1.7;
        let fp = t_hi * pix / max(d.z, 0.05);
        // (where: patches of some 20 km)
        let field = hc_fbm(ph / 22000.0 + vec2<f32>(3.1, 7.7), fp / 22000.0, 3);
        let patchy = smoothstep(0.62 - 0.3 * amount, 0.85 - 0.2 * amount, field);
        if (patchy > 0.001) {
            // (the wind aloft lays the fibres out; it turns slowly across the sky)
            let turn = 0.6 + 1.2 * (hc_fbm(ph / 60000.0, 0.0, 2) - 0.5);
            let ax = vec2<f32>(cos(turn), sin(turn));
            let r = vec2<f32>(dot(ph, ax), dot(ph, vec2<f32>(-ax.y, ax.x)));
            // (the wisps: soft elongated tufts some kilometres long, their edges fading over
            // hundreds of metres - real cirrus has no hard rim)
            let warp = hc_fbm(r / vec2<f32>(12000.0, 6000.0) + vec2<f32>(1.7, 4.2), fp / 6000.0, 2) - 0.5;
            let body = hc_fbm((r + vec2<f32>(0.0, 1500.0 * warp)) / vec2<f32>(14000.0, 2600.0) + vec2<f32>(5.3, 1.9), fp / 2600.0, 4);
            // (its edge kept some pixels wide: low towards the horizon a wisp seen across its
            // width is only a few pixels high, and an edge sharpened to the noise drew thin
            // bright lines there; further down it is taken at its mean, an even sheen)
            let we = 2.0 * fp / 2600.0;
            let wisp = mix(smoothstep(0.42 - we, 0.8 + we, body), 0.25, smoothstep(0.04, 0.16, fp / 2600.0));
            // (the fibres in them: fine streaks along the wind, combed and bent by the same
            // slow field, that only shade the wisp - a hair-like grain of brighter and
            // fainter ice, never a line of its own. A threshold on them drew sharp parallel
            // contour lines over the sky.)
            // (a finer field of its own lets the strands part and gather like hair, not run
            // side by side as one bent comb)
            let comb = hc_fbm(r / vec2<f32>(3500.0, 1400.0) + vec2<f32>(8.1, 2.6), fp / 1400.0, 2) - 0.5;
            let f = vec2<f32>(r.x / 2600.0, (r.y + 1500.0 * warp + 500.0 * comb) / 450.0);
            let fib = hc_fbm(f, 2.0 * fp / 450.0, 4);
            // (what the footprint has blurred out: the grain evens out into the wisp's sheen
            // well before a strand narrows to a pixel)
            let lost = smoothstep(0.05, 0.45, fp / 450.0);
            let he = 2.0 * fp / 450.0;
            let hair = mix(1.0, 0.45 + 1.1 * smoothstep(0.25 - he, 0.85 + he, fib), 0.4 * (1.0 - lost));
            let dens = patchy * wisp * hair;
            hi_cover = clamp(dens, 0.0, 1.0) * mix(0.3, 0.6, clamp(veil, 0.0, 1.0)) * smoothstep(0.02, 0.15, d.z);
            // (lit by the sun at its own height, through thin ice: forward scattering
            // towards the sun, a soft grey-white away from it)
            let hi_sun = enh.cloud_sun[3].rgb * (0.2 + 0.4 * hg_phase(cos_sun, 0.6) * PI);
            // (never darker than the sky behind, but kept in its own colour: a per-channel
            // max took the sky's blue and the sun's red and tinted the fibres pink)
            let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
            let veiled = mix(col, hi_sun / PI + sky_top, hi_cover);
            col = veiled * max(1.0, dot(col, lw) / max(dot(veiled, lw), 1e-6));
        }
    }
    return vec4<f32>(col, 1.0 - (1.0 - cover) * (1.0 - hi_cover));
}

// The lightning's colour: a channel of plasma at some 30 000 K, scattered by the cloud
const FLASH_COLOUR: vec3<f32> = vec3<f32>(0.80, 0.90, 1.0);

fn flash_dir() -> vec3<f32> {
    return enh.flash2.xyz;
}

// The light a flash puts into the cloud at p (as radiance towards the camera): the cloud
// lights up round the flash, brightest next to its channel, fading with the distance from
// it through the cloud (a cell's size is some ten kilometres; the light spreads by
// multiple scattering far beyond the flash itself).
fn flash_in_cloud(p: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if (enh.flash.x <= 0.0) {
        return vec3<f32>(0.0);
    }
    let fd = flash_dir();
    let ang = acos(clamp(dot(d, fd), -1.0, 1.0));
    let near = exp(-ang * ang / (2.0 * 0.22 * 0.22));
    let broad = exp(-ang * ang / (2.0 * 0.9 * 0.9));
    return FLASH_COLOUR * enh.flash.x * (8.0 * near + 1.0 * broad);
}

// The channel of a cloud-to-ground stroke: a jagged line from the cloud's base to the
// ground (a fractal in the real thing; sines of three scales with the flash's seed make the
// same crooked path, one branch breaking off a third of the way down), seen as a thin hot
// core in a wider halo.
fn flash_channel(d: vec3<f32>) -> vec3<f32> {
    let k = enh.flash2.w;
    if (k <= 0.01) {
        return vec3<f32>(0.0);
    }
    let az = vec2<f32>(enh.flash.y, enh.flash.z);
    // angle sideways from the flash's vertical plane, and the elevation, of this pixel
    let across = (d.x * az.y - d.y * az.x);
    let along = d.x * az.x + d.y * az.y;
    if (along <= 0.0) {
        return vec3<f32>(0.0);
    }
    let el = atan2(d.z, along);
    let top = enh.flash.w + 0.14;
    let bottom = 0.0;
    if (el > top || el < bottom - 0.02) {
        return vec3<f32>(0.0);
    }
    let u = clamp((top - el) / (top - bottom), 0.0, 1.0);
    let seed = enh.flash3.x;
    let amp = (top - bottom) * 0.22;
    let path = amp * (0.5 * sin(6.0 * u + seed) + 0.28 * sin(17.0 * u + 2.3 * seed) + 0.14 * sin(41.0 * u + 3.1 * seed));
    var dist = abs(across - path);
    // the branch: leaves the channel at a third of its way down and bends away from it
    let ub = 0.33;
    if (u > ub && u < ub + 0.4) {
        let pb = amp * (0.5 * sin(6.0 * ub + seed) + 0.28 * sin(17.0 * ub + 2.3 * seed) + 0.14 * sin(41.0 * ub + 3.1 * seed));
        let side = select(-1.0, 1.0, fract(seed * 7.0) > 0.5);
        let branch = pb + side * (u - ub) * (top - bottom) * 0.45 + amp * 0.12 * sin(33.0 * (u - ub) + seed);
        dist = min(dist, abs(across - branch) * 1.6);
    }
    let core = exp(-dist * dist / (2.0 * 0.0009 * 0.0009));
    let halo = exp(-dist * dist / (2.0 * 0.008 * 0.008));
    return FLASH_COLOUR * k * (60.0 * core + 1.5 * halo) * (1.0 - 0.5 * u);
}

// --- the dome

// Where the screen's clouds were traced from (cloud_pass.rs `CloudView`): the inverse of the
// view-projection without its translation (clip to direction), that matrix this frame and
// for the picture in the history, how far the camera and the clouds moved since (xyz, m;
// w the history's weight, 0 without one), the trace's and the history's sizes, the
// picture's own size, x the frame's number (with this pass's round added, `probe.p.x`) and
// w the march's steps, and x 1 when the dome may look the clouds up, y the old table
// scale over the new one (the history is kept divided by it), z a traced pixel's angular
// size (rad).
struct CloudView {
    inv_vp: mat4x4<f32>,
    vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    moved: vec4<f32>,
    size: vec4<f32>,
    full: vec4<f32>,
    frame: vec4<f32>,
    params: vec4<f32>,
};
@group(1) @binding(12) var<uniform> cview: CloudView;
// The screen's clouds, accumulated (clouds_ss.wgsl `fs_enhanced_cloud_resolve`): rgb the
// light they scatter (divided by the table scale), a what they let through of the sky behind.
@group(0) @binding(17) var t_clouds_ss: texture_2d<f32>;
// The reflection probe (its sharpest level is the sky cube: the sky, the clouds, and their
// cover in alpha): the sky of a view the screen's clouds do not cover.
@group(0) @binding(12) var t_env_probe: texture_cube<f32>;

// The weather's fog towards the sky from height h0 over its base, 30 km of it. The weather's
// fog range is the game's draw distance as much as the air's haze (a fair day's two
// kilometres, a rainy one's two hundred metres: the game's sky stayed blue or showed its
// clouds over them), so the sky takes a sixth of its density: the horizon is lost in it
// and the sky low over it whitens, but a fair sky stays blue overhead and an overcast one
// keeps the shapes of its deck - a uniform grey sheet hid them; only a real fog (some tens
// of metres) still hides the sky.
fn sky_air(d: vec3<f32>, h0: f32) -> vec4<f32> {
    let k = 0.16;
    let tau_fog = layer_depth(enh.fog.x * k, enh.fog.y, h0, h0 + 30000.0 * max(d.z, 0.0), 30000.0);
    if (tau_fog < 1e-5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let t = exp(-tau_fog);
    let fog_in = enh.fog_color.rgb + enh.sun.rgb * hg_phase(dot(d, camera.sun_dir.xyz), 0.55) * 0.9;
    return vec4<f32>(fog_in * (1.0 - t), t);
}

// The screen's clouds towards d, and whether there are any (a < 0: none - not traced, or
// outside the traced view).
fn screen_clouds(d: vec3<f32>) -> vec4<f32> {
    if (cview.params.x < 0.5) {
        return vec4<f32>(-1.0);
    }
    let c = cview.vp * vec4<f32>(d, 1.0);
    if (c.w <= 1e-6) {
        return vec4<f32>(-1.0);
    }
    let ndc = c.xy / c.w;
    if (abs(ndc.x) > 1.002 || abs(ndc.y) > 1.002) {
        return vec4<f32>(-1.0);
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return clouds_ss_bicubic(uv);
}

// The screen's clouds at uv, filtered up from their picture of half the window's size with
// a Catmull-Rom spline (nine bilinear taps): a plain bilinear lookup blurred their edges
// over four of the window's pixels. Clamped to the four nearest texels, so the spline's
// overshoot leaves no dark or bright rims.
fn clouds_ss_bicubic(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(t_clouds_ss));
    let sp = uv * size;
    let tc1 = floor(sp - vec2<f32>(0.5)) + vec2<f32>(0.5);
    let f = sp - tc1;
    let w0 = f * (vec2<f32>(-0.5) + f * (vec2<f32>(1.0) - 0.5 * f));
    let w1 = vec2<f32>(1.0) + f * f * (vec2<f32>(-2.5) + 1.5 * f);
    let w2 = f * (vec2<f32>(0.5) + f * (vec2<f32>(2.0) - 1.5 * f));
    let w3 = f * f * (vec2<f32>(-0.5) + 0.5 * f);
    let w12 = w1 + w2;
    let tc0 = (tc1 - vec2<f32>(1.0)) / size;
    let tc3 = (tc1 + vec2<f32>(2.0)) / size;
    let tc12 = (tc1 + w2 / w12) / size;
    var c = textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc0.x, tc0.y), 0.0) * (w0.x * w0.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc12.x, tc0.y), 0.0) * (w12.x * w0.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc3.x, tc0.y), 0.0) * (w3.x * w0.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc0.x, tc12.y), 0.0) * (w0.x * w12.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc12.x, tc12.y), 0.0) * (w12.x * w12.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc3.x, tc12.y), 0.0) * (w3.x * w12.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc0.x, tc3.y), 0.0) * (w0.x * w3.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc12.x, tc3.y), 0.0) * (w12.x * w3.y);
    c = c + textureSampleLevel(t_clouds_ss, s_lin, vec2<f32>(tc3.x, tc3.y), 0.0) * (w3.x * w3.y);
    let i = vec2<i32>(tc1 - vec2<f32>(0.5));
    let mx = vec2<i32>(size) - vec2<i32>(1);
    let a = textureLoad(t_clouds_ss, clamp(i, vec2<i32>(0), mx), 0);
    let b = textureLoad(t_clouds_ss, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), mx), 0);
    let e = textureLoad(t_clouds_ss, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), mx), 0);
    let g = textureLoad(t_clouds_ss, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), mx), 0);
    return clamp(c, min(min(a, b), min(e, g)), max(max(a, b), max(e, g)));
}

@fragment
fn fs_enhanced(in: VsOut) -> @location(0) vec4<f32> {
    return enhanced_sky(in, true);
}

// The sky towards the dome's pixel: the table, the high layers and the screen's clouds over
// it, the weather's fog, the sun's disc, the moon and the stars, the lightning; where the
// screen's clouds were not traced (`screen` false: a mirror, or another view beyond the
// window's) the reflection probe's sky cube instead.
fn enhanced_sky(in: VsOut, screen: bool) -> vec4<f32> {
    let d = normalize(in.dir);
    let pre = enh.exposure.x;
    // (a pixel's angular size, for the high layers' filtering)
    let pix = max(length(fwidth(d)), 1e-5);
    var ss = vec4<f32>(-1.0);
    if (screen) {
        ss = screen_clouds(d);
    }
    var col = vec3<f32>(0.0);
    var cover = 0.0;
    if (ss.a >= 0.0) {
        let hi = high_clouds(d, sky_table(d), pix);
        col = hi.rgb * ss.a + ss.rgb * enh.ground.w;
        cover = 1.0 - ss.a * (1.0 - hi.a);
        // the weather's fog along the whole way to the horizon (the clear air is in the
        // table already: taken again, it whitened the whole sky)
        let h0 = camera.cam_pos.z - enh.fog.z;
        let a = sky_air(d, h0);
        col = col * a.a + a.rgb;
    } else {
        let p = textureSampleLevel(t_env_probe, s_lin, vec3<f32>(d.x, d.z, d.y), 0.0);
        col = p.rgb * enh.fog_color.w;
        cover = clamp(p.a, 0.0, 1.0);
    }
    var sun_px = vec3<f32>(0.0);
    // the sun: a limb-darkened disc as bright as its irradiance spread over its size,
    // behind whatever cloud there is
    let sd = normalize(camera.sun_dir.xyz);
    let r = enh.sun.w;
    let cosang = dot(d, sd);
    if (cosang > cos(r * 1.5) && d.z > -0.02) {
        let x = clamp(acos(clamp(cosang, -1.0, 1.0)) / r, 0.0, 1.5);
        let disc = 1.0 - smoothstep(0.85, 1.1, x);
        let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - min(x, 1.0) * min(x, 1.0), 0.0)));
        let h0 = camera.cam_pos.z - enh.fog.z;
        let t = sky_air(d, h0).a;
        sun_px = enh.sun_disc.rgb / (PI * r * r) * disc * limb * (1.0 - cover) * enh.lights.w * t;
    }
    col = col + night_sky(d, cover, fwidth(d));
    // lightning (not part of the accumulated clouds: a flash lasts a tenth of a second):
    // the cloud lit from inside, the haze lit a little, and the channel
    if (enh.flash.x > 0.0) {
        col = col + flash_in_cloud(vec3<f32>(0.0), d) * (0.12 + 0.88 * cover);
        col = col + flash_channel(d);
    }
    // the dome is drawn pre-exposed; the disc only a little over white: the eye's glare round
    // it is drawn from its light (fog_lamps.wgsl `sun_glare`), and from the disc's pixels
    // as well the glow would have counted it twice
    return vec4<f32>(min(col * pre, vec3<f32>(4000.0)) + min(sun_px * pre, vec3<f32>(24.0)), 1.0);
}

// The mirrors of an Enhanced picture are drawn with the plain shading (lib.rs, the cost of
// the enhanced shader in a small picture), but their sky is this one, tone-mapped here
// with the window's curve (post.wgsl `natural_tone`): the plain envir.cfg sky showed a blue
// dusk in the mirrors over the window's grey one, or the other way round (#1754, #1449).
// The meter's correction is left out (by day a few tenths of a stop at most).
fn mirror_tone(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    let knee = 0.66;
    let x = max(color, vec3<f32>(0.0));
    let y = 0.18 * pow(x / 0.18 + vec3<f32>(1e-7), vec3<f32>(contrast));
    let peak = max(y.r, max(y.g, y.b));
    if (peak <= knee) {
        return y;
    }
    let d = 1.0 - knee;
    let shoulder = vec3<f32>(knee) + d * (vec3<f32>(1.0) - exp(-(y - vec3<f32>(knee)) / d));
    let soft = select(y, shoulder, y > vec3<f32>(knee));
    let np = knee + d * (1.0 - exp(-(peak - knee) / d));
    let o = mix(y * (np / peak), soft, 0.4);
    let g = 1.0 - 1.0 / (0.3 * (peak - np) + 1.0);
    return mix(o, vec3<f32>(np), g);
}

@fragment
fn fs_enhanced_mirror(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = enhanced_sky(in, false).rgb;
    return vec4<f32>(clamp(mirror_tone(hdr, max(enh.debug.w, 1.0)), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

fn star_hash(c: vec3<i32>) -> vec4<f32> {
    var x = vec4<u32>(bitcast<vec3<u32>>(c), 0x9e3779b9u).xyzw;
    x = x * 1664525u + 1013904223u;
    x.x = x.x + x.y * x.w;
    x.y = x.y + x.z * x.x;
    x.z = x.z + x.x * x.y;
    x.w = x.w + x.y * x.z;
    x = x ^ (x >> vec4<u32>(16u));
    x.x = x.x + x.y * x.w;
    x.y = x.y + x.z * x.x;
    x.z = x.z + x.x * x.y;
    x.w = x.w + x.y * x.z;
    return vec4<f32>(x & vec4<u32>(0xffffffu)) / 16777216.0;
}

// The moon and the stars towards d (radiance), behind what cloud covers it. The moon is a
// lit sphere: its lit side faces the sun, its phase is what the sun's direction makes of
// it, and its grey seas darken parts of the disc. The stars are a few thousand points
// fixed in the sky, of a few magnitudes, slightly coloured; they show only where the sky
// behind them is dark - a lit city's glow and the haze leave the brightest few.
fn night_sky(d: vec3<f32>, cover: f32, pix: vec3<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    if (d.z < -0.01) {
        return col;
    }
    let h0 = camera.cam_pos.z - enh.fog.z;
    let through = sky_air(d, h0).a * (1.0 - cover);
    if (through < 0.01) {
        return col;
    }
    let m = enh.moon.xyz;
    let r = enh.moon.w;
    let cm = dot(d, m);
    if (cm > cos(r * 1.3) && max(enh.moon_disc.r, enh.moon_disc.g) > 0.0) {
        // the point on the moon's sphere seen in direction d
        let right = normalize(cross(m, vec3<f32>(0.0, 0.0, 1.0)) + vec3<f32>(1e-5, 0.0, 0.0));
        let up = cross(right, m);
        let q = vec2<f32>(dot(d, right), dot(d, up)) / r;
        let q2 = dot(q, q);
        let rim = 1.0 - smoothstep(0.92, 1.05, sqrt(q2));
        let nz = sqrt(max(1.0 - q2, 0.0));
        let n = normalize(right * q.x + up * q.y - m * nz);
        let sun_side = smoothstep(-0.04, 0.08, dot(n, normalize(camera.sun_dir.xyz)));
        // the seas: a few broad dark patches fixed on the face
        let sea = smoothstep(0.45, 0.7, cloud_fbm(q * 1.7 + vec2<f32>(3.1, 7.7)));
        let albedo = 1.0 - 0.38 * sea;
        // (as bright as the lit part's share of the irradiance spread over the disc; the
        // dark part keeps a trace of earthshine)
        let lit_share = max(0.5 + 0.5 * dot(-m, normalize(camera.sun_dir.xyz)), 0.03);
        let l = enh.moon_disc.rgb / (PI * r * r * lit_share) * (sun_side + 0.004) * albedo * rim;
        col = col + l * through;
    }
    // the stars: one candidate in some of the cells of a grid on the sphere
    let vis = enh.moon_disc.w;
    if (vis > 0.0 && d.z > 0.0) {
        let cells = 140.0;
        let g = d * cells;
        let c = vec3<i32>(floor(g));
        let h = star_hash(c);
        let density = 0.014;
        if (h.x < density) {
            let p = (vec3<f32>(c) + vec3<f32>(0.25) + h.yzw * 0.5) / cells;
            let sd = normalize(p);
            let ang = acos(clamp(dot(d, sd), -1.0, 1.0));
            // a point of light about a pixel across
            let size = clamp(length(pix) * 0.6, 1e-5, 0.004);
            let spot = exp(-ang * ang / (size * size));
            // brightness: many faint ones, a few bright (a power law of the magnitudes)
            let b = pow(1.0 - h.x / density, 4.0);
            let tint = mix(vec3<f32>(1.0, 0.86, 0.72), vec3<f32>(0.8, 0.88, 1.0), h.w);
            // the faintest that shows against the sky behind it (in the picture's own
            // units): a dark country sky shows thousands, a city's glow the few dozen
            // brightest, a twilight none
            let sky_l = dot(sky_table(d), vec3<f32>(0.2126, 0.7152, 0.0722)) * enh.exposure.x;
            let limit = clamp(sky_l * 90.0 + (1.0 - vis), 0.0, 1.2);
            let shows = smoothstep(limit, limit + 0.15, b) * (1.0 - smoothstep(0.03, 0.1, sky_l));
            col = col + tint * (0.04 + 1.4 * b) * spot * shows * through / max(enh.exposure.x, 1e-6) * smoothstep(0.0, 0.15, d.z);
        }
    }
    return col;
}

// --- the reflection probe: a cube map of the sky seen from the camera, drawn now and then
// and blurred into its mip levels for rough surfaces (divided by the table scale, which
// the main pass multiplies back in).
struct ProbeParams {
    // x first face of this pass (0 or 3), y roughness, z face size of level 0, w level
    p: vec4<f32>,
};
@group(0) @binding(15) var<uniform> probe: ProbeParams;
@group(0) @binding(16) var t_probe_src: texture_cube<f32>;

// Direction of a texel of cube face f (u, v in -1..1, v down), in cube coordinates.
fn face_dir(f: i32, u: f32, v: f32) -> vec3<f32> {
    var c = vec3<f32>(-u, -v, -1.0);
    switch f {
        case 0: { c = vec3<f32>(1.0, -v, -u); }
        case 1: { c = vec3<f32>(-1.0, -v, u); }
        case 2: { c = vec3<f32>(u, 1.0, v); }
        case 3: { c = vec3<f32>(u, -1.0, -v); }
        case 4: { c = vec3<f32>(u, -v, 1.0); }
        default: {}
    }
    return normalize(c);
}

fn cube_to_world(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x, c.z, c.y);
}

struct ProbeOut {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) c: vec4<f32>,
};

struct FsIn {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_probe(@builtin(vertex_index) i: u32) -> FsIn {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    return FsIn(vec4<f32>(x, y, 0.0, 1.0));
}

fn probe_sky(f: i32, uv: vec2<f32>) -> vec4<f32> {
    let c = face_dir(f, uv.x, uv.y);
    let d = cube_to_world(c);
    var col: vec3<f32>;
    if (d.z >= 0.0) {
        // the sky cube (lib.rs binds it as level 0's source), already divided by the table
        // scale, with the clouds' cover in alpha (the dome hides the sun's disc by it where
        // it reads the probe)
        return textureSampleLevel(t_probe_src, s_lin, c, 0.0);
    } else {
        // below the horizon: the table's distant ground, with the houses and trees of a
        // town darkening the band just under the horizon
        col = sky_table(d);
        let band = smoothstep(0.02, 0.25, -d.z);
        col = mix(col, enh.ground.rgb * 0.8, 1.0 - band * 0.6);
    }
    return vec4<f32>(col / max(enh.ground.w, 1e-8), 0.0);
}

@fragment
fn fs_probe_sky(in: FsIn) -> ProbeOut {
    let size = probe.p.z;
    let uv = in.clip.xy / size * 2.0 - vec2<f32>(1.0);
    let f0 = i32(probe.p.x);
    return ProbeOut(probe_sky(f0, uv), probe_sky(f0 + 1, uv), probe_sky(f0 + 2, uv));
}

fn radical_inverse(i: u32) -> f32 {
    var b = i;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10;
}

// GGX-filtered radiance around n (the split-sum assumption: view = normal), sampling the
// sharper levels at a lod that matches each sample's footprint.
fn prefilter(n: vec3<f32>) -> vec3<f32> {
    let rough = probe.p.y;
    let a = rough * rough;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    let count = 32u;
    let texel_sa = 4.0 * PI / (6.0 * probe.p.z * probe.p.z);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let xi = vec2<f32>(f32(i) / f32(count), radical_inverse(i));
        let phi = 2.0 * PI * xi.x;
        let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        let sin_t = sqrt(1.0 - cos_t * cos_t);
        let h = tx * (cos(phi) * sin_t) + ty * (sin(phi) * sin_t) + n * cos_t;
        let l = 2.0 * dot(n, h) * h - n;
        let nl = dot(n, l);
        if (nl <= 0.0) {
            continue;
        }
        let nh = max(dot(n, h), 0.0);
        let d2 = nh * nh * (a * a - 1.0) + 1.0;
        let dd = a * a / (PI * d2 * d2);
        let pdf = dd * 0.25;
        let sample_sa = 1.0 / (f32(count) * pdf + 1e-4);
        let lod = clamp(0.5 * log2(sample_sa / texel_sa) + 1.0, 0.0, probe.p.w - 1.0);
        sum = sum + textureSampleLevel(t_probe_src, s_lin, l, lod).rgb * nl;
        wsum = wsum + nl;
    }
    return sum / max(wsum, 1e-4);
}

@fragment
fn fs_probe_filter(in: FsIn) -> ProbeOut {
    let size = probe.p.z / pow(2.0, probe.p.w);
    let uv = in.clip.xy / size * 2.0 - vec2<f32>(1.0);
    let f0 = i32(probe.p.x);
    return ProbeOut(
        vec4<f32>(prefilter(face_dir(f0, uv.x, uv.y)), 1.0),
        vec4<f32>(prefilter(face_dir(f0 + 1, uv.x, uv.y)), 1.0),
        vec4<f32>(prefilter(face_dir(f0 + 2, uv.x, uv.y)), 1.0),
    );
}

// The steps of the sky cube's clouds: the cube is the reflections' sky and the mirrors',
// redrawn a slice a frame and averaged over its rounds.
const CUBE_CLOUD_STEPS: i32 = 40;

// One face of the sky cube: the sky radiance (the table, the clouds, the air) divided by the
// table scale, and the cloud cover in alpha (the sun's disc is hidden by it).
@fragment
fn fs_sky_cube(in: FsIn) -> @location(0) vec4<f32> {
    let size = probe.p.z;
    let uv = in.clip.xy / size * 2.0 - vec2<f32>(1.0);
    let d = cube_to_world(face_dir(i32(probe.p.x), uv.x, uv.y));
    let pix = 1.4 / size;
    // a new start point for the steps each redraw (p.y counts the redraws): interleaved
    // gradient noise over the face, shifted by the golden ratio
    let ign = fract(52.9829189 * fract(dot(in.clip.xy, vec2<f32>(0.06711056, 0.00583715))));
    let jitter = fract(ign + (probe.p.y + 8.0 * f32(probe.p.x)) * 0.61803399);
    let bg = sky_table(d);
    let hi = high_clouds(d, bg, pix);
    let c = cloud_march(d, bg, CUBE_CLOUD_STEPS, jitter, pix, false);
    var col = hi.rgb * c.trans + c.light;
    let h0 = camera.cam_pos.z - enh.fog.z;
    let a = sky_air(d, h0);
    col = col * a.a + a.rgb;
    return vec4<f32>(col / max(enh.ground.w, 1e-8), 1.0 - c.trans * (1.0 - hi.a));
}

// Enhanced graphics: the window's clouds, drawn per pixel the way games draw volumetric
// clouds (Horizon Zero Dawn, Frostbite, Unreal):
// * `fs_enhanced_cloud_trace` marches them (sky_enhanced.wgsl `cloud_march`) for a pixel
//   in four each way, its ray through another quarter of the pixel each frame and its
//   steps started at another point (interleaved gradient noise moved on by the golden
//   ratio), and keeps how far the clouds are seen; where the scene's geometry covers the pixel and every
//   pixel round it (the depth prepass) it marches nothing;
// * `fs_enhanced_cloud_resolve` takes that into a picture at half the window's size: each
//   pixel finds where its clouds were in the last frame's picture (their depth and the last
//   frame's view, the camera's and the clouds' own movement), takes the history from there
//   clamped to what the traced pixels round it allow (so nothing ghosts when a cloud or
//   the view changes), and blends in the new trace - most where a traced ray landed on it
//   this frame. Turning the view, driving and the clouds' drift move nothing but what moves;
// * the dome (`enhanced_sky`) looks the result up only where the sky shows: the geometry is
//   drawn first, so the clouds never cover it.

// The scene's depth (the prepass, reversed: 0 where nothing was drawn); a texel when there
// is none.
@group(2) @binding(0) var t_scene_depth: texture_depth_2d;
// The trace (`fs_enhanced_cloud_trace`): its light and transmittance, and the clouds' depth
// (m; < 0 where the geometry covers it); the accumulated picture of the frame before.
@group(0) @binding(23) var t_trace: texture_2d<f32>;
@group(0) @binding(24) var t_trace_depth: texture_2d<f32>;
@group(0) @binding(25) var t_cloud_hist: texture_2d<f32>;

// This pass's frame number: the frame's, with the round of a picture drawn on its own
// (`probe.p.x`) added.
fn cloud_frame() -> f32 {
    return cview.frame.x + probe.p.x;
}

// Where in its pixel the trace's ray goes this frame (in traced pixels): the four quarters
// in turn, each the centre of one of the resolved picture's pixels.
fn trace_jitter(frame: f32) -> vec2<f32> {
    let k = u32(frame) % 4u;
    return vec2<f32>(select(-0.25, 0.25, k == 1u || k == 2u), select(-0.25, 0.25, k == 1u || k == 3u));
}

// The view's direction through the point uv (0..1, y down) of the picture.
fn cloud_view_dir(uv: vec2<f32>) -> vec3<f32> {
    let p = cview.inv_vp * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    return normalize(p.xyz / p.w);
}

// Whether the scene's geometry covers the traced pixel at `px` and a traced pixel round it
// (what the resolve and the dome's filter read), every other pixel of it looked at.
fn cloud_covered(px: vec2<f32>) -> bool {
    let dims = vec2<i32>(textureDimensions(t_scene_depth));
    if (dims.x < 4) {
        return false;
    }
    let scale = vec2<f32>(dims) / cview.size.xy;
    let c = vec2<i32>((floor(px) + vec2<f32>(0.5)) * scale);
    let n = i32(ceil(max(scale.x, scale.y) * 1.5));
    for (var y = -n; y <= n; y = y + 2) {
        for (var x = -n; x <= n; x = x + 2) {
            let q = clamp(c + vec2<i32>(x, y), vec2<i32>(0), dims - vec2<i32>(1));
            if (textureLoad(t_scene_depth, q, 0) <= 0.0) {
                return false;
            }
        }
    }
    return true;
}

struct TraceOut {
    @location(0) col: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_enhanced_cloud_trace(in: FsIn) -> TraceOut {
    if (cloud_covered(in.clip.xy)) {
        return TraceOut(vec4<f32>(0.0, 0.0, 0.0, 1.0), -1.0);
    }
    let frame = cloud_frame();
    let uv = (in.clip.xy + trace_jitter(frame)) / cview.size.xy;
    let d = cloud_view_dir(uv);
    let ign = fract(52.9829189 * fract(dot(floor(in.clip.xy), vec2<f32>(0.06711056, 0.00583715))));
    let jitter = fract(ign + frame * 0.61803399);
    let bg = sky_table(d);
    let c = cloud_march(d, bg, i32(cview.frame.w), jitter, cview.params.z, true);
    return TraceOut(vec4<f32>(c.light / max(enh.ground.w, 1e-8), c.trans), c.dist);
}

@fragment
fn fs_enhanced_cloud_resolve(in: FsIn) -> @location(0) vec4<f32> {
    let frame = cloud_frame();
    let ts = cview.size.xy;
    let tsi = vec2<i32>(ts);
    let uv = in.clip.xy / cview.size.zw;
    // where this pixel lies among the traced ones (their centres at i + 0.5 this frame)
    let x = uv * ts - trace_jitter(frame);
    let ic = clamp(vec2<i32>(floor(x)), vec2<i32>(0), tsi - vec2<i32>(1));
    let fresh = all(abs(fract(x) - vec2<f32>(0.5)) < vec2<f32>(0.15));
    var cur = textureSampleLevel(t_trace, s_lin, x / ts, 0.0);
    if (fresh) {
        cur = textureLoad(t_trace, ic, 0);
    }
    // the traced pixels round it: their mean and spread (the geometry's left out)
    var m1 = vec4<f32>(0.0);
    var m2 = vec4<f32>(0.0);
    var n = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let q = clamp(ic + vec2<i32>(dx, dy), vec2<i32>(0), tsi - vec2<i32>(1));
            if (textureLoad(t_trace_depth, q, 0).r >= 0.0) {
                let s = textureLoad(t_trace, q, 0);
                m1 = m1 + s;
                m2 = m2 + s * s;
                n = n + 1.0;
            }
        }
    }
    if (n < 0.5) {
        return cur;
    }
    let mean = m1 / n;
    let spread = sqrt(max(m2 / n - mean * mean, vec4<f32>(0.0)));
    // where the clouds this pixel sees were in the last frame's picture
    let depth = textureLoad(t_trace_depth, ic, 0).r;
    let p = cloud_view_dir(uv) * select(1e6, depth, depth > 0.0) + cview.moved.xyz;
    let pc = cview.prev_vp * vec4<f32>(p, 1.0);
    let manual = probe.p.y >= 0.0;
    var w = select(cview.moved.w, probe.p.y, manual);
    if (pc.w <= 1e-6) {
        w = 0.0;
    }
    let puv = vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
    if (any(puv < vec2<f32>(0.0)) || any(puv > vec2<f32>(1.0))) {
        w = 0.0;
    }
    if (w <= 0.0) {
        return cur;
    }
    var hist = textureSampleLevel(t_cloud_hist, s_lin, puv, 0.0);
    hist = vec4<f32>(hist.rgb * cview.params.y, hist.a);
    if (!manual) {
        // (the history only within what the neighbourhood allows; a fresh ray replaces
        // more of it than the upsampled trace between them)
        let g = 1.25;
        hist = clamp(hist, mean - g * spread - vec4<f32>(1e-4), mean + g * spread + vec4<f32>(1e-4));
        w = w * select(0.985, 0.9, fresh);
    }
    return mix(cur, hist, w);
}
