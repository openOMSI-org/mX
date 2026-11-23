// Light coronas ([light_enh]): camera-facing additive sprites.
struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    // This remains part of the shared camera layout even though coronas only use
    // `view_proj` and `cam_pos` today.  Keeping the prefix aligned prevents later
    // corona effects from reading every lighting field one vec4 too early.
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
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var t_corona: texture_2d<f32>;
@group(1) @binding(1) var s_corona: sampler;

struct CoronaIn {
    @location(0) pos: vec3<f32>,      // relative to the render origin
    @location(1) size: f32,           // world radius
    @location(2) color: vec4<f32>,    // rgb, alpha = brightness (0..2)
    @location(3) dir: vec4<f32>,      // xyz facing direction (0 = omni), w cos of the outer cone
    @location(4) up: vec4<f32>,       // xyz up / rotation axis, w rotating mode (0 flat, 1 about up, 2 billboard)
    @location(5) extra: vec4<f32>,    // x cos of the inner cone, y z offset (<0 default), z parameter bits
    @builtin(vertex_index) vid: u32,
};
struct CoronaOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // 1 for what is drawn at the scene's level (a precipitation streak, lit fog: a cone or
    // a halo), 0 for a light's glare
    @location(2) kind: f32,
    // 1 when the light is drawn with a star (`[light_enh_2]` parameter bit 1)
    @location(3) star: f32,
    // 1 for a light's cone in the fog (OMSI's fan, see below)
    @location(4) beam: f32,
    // the cone's inner and outer half angles (radians)
    @location(5) cone: vec2<f32>,
    // a smoke puff's: how high this point is over the ground it fades into (m), and how
    // high the fade reaches (0: not faded)
    @location(6) ground: vec2<f32>,
    // precipitation (Enhanced, see `precip_light`): the particle's place, w 1 for rain, 2
    // for snow, 0 for anything else
    @location(7) wpos: vec4<f32>,
};

// How many times wider than the light's own sprite the Enhanced picture draws a lamp's quad, for
// the faint glare round it (`fs_enhanced`) - with *Glow of lights* on: 64 times the pixels.
const GLARE_K: f32 = 8.0;
// the glare's strength at the sprite's rim, against the lamp's core
const GLARE_TAIL: f32 = 0.4;

// How many times its sprite a lamp's quad is drawn: `GLARE_K` with *Glow of lights* on (enh.led.z),
// and in mist, fog and rain (which have a halo of their own, in the bitmap) only the sprite.
fn glare_scale() -> f32 {
    let wet = clamp(enh.fog.x * 400.0 + enh.weather.z * 0.7, 0.0, 1.0);
    return select(1.0, GLARE_K, enh.led.z > 0.5 && wet < 1.0);
}

@vertex
fn vs_main(in: CoronaIn) -> CoronaOut {
    return corona_vertex(in, 1.0);
}

@vertex
fn vs_enhanced(in: CoronaIn) -> CoronaOut {
    return corona_vertex(in, glare_scale());
}

fn corona_vertex(in: CoronaIn, glare: f32) -> CoronaOut {
    var out: CoronaOut;
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let c = corners[in.vid % 6u];
    if (in.extra.w > 2.5) {
        // a [smoke] puff (`smoke_sprite`): rauch.tga facing the viewer as Omsi.exe lays it
        // out (0x5a2b5c), turned about the line of sight by its own angle (dir: its cosine
        // and sine) and moved up.z towards the eye in depth only - its size on the screen
        // stays (Omsi.exe's 0.1 m: the view depth less 0.1 put through the projection's
        // depth terms)
        let to_cam = camera.cam_pos.xyz - in.pos;
        let dist = length(to_cam);
        let view_dir = to_cam / max(dist, 0.001);
        var side = cross(vec3<f32>(0.0, 0.0, 1.0), view_dir);
        if (length(side) < 0.0001) {
            // (seen from straight above or below)
            side = camera.cam_right.xyz;
        }
        let right0 = normalize(side);
        let up0 = cross(view_dir, right0);
        let right = right0 * in.dir.x + up0 * in.dir.y;
        let up = up0 * in.dir.x - right0 * in.dir.y;
        let size = max(in.size, dist * 0.002) * 0.9;
        let wp = in.pos + (right * c.x + up * c.y) * size;
        var sout: CoronaOut;
        sout.clip = camera.view_proj * vec4<f32>(wp, 1.0);
        if (in.up.z > 0.0) {
            let nearer = camera.view_proj * vec4<f32>(wp + view_dir * min(in.up.z, dist * 0.9), 1.0);
            sout.clip.z = nearer.z / nearer.w * sout.clip.w;
        }
        sout.uv = c * 0.5 + 0.5;
        let a = min(in.color.a, 1.0);
        sout.color = vec4<f32>(in.color.rgb, a);
        // Omsi.exe lets a puff sink on into the road, which cuts it off in a straight line
        // (under a wheel's spray: bright bands across the road); here it fades out over
        // its lowest part into the ground it was thrown up from (up.x, when up.y is 1),
        // from up.w above it (`smoke_ground_fade`) down to nothing
        if (in.up.y > 0.5) {
            sout.ground = vec2<f32>(wp.z - in.up.x, in.up.w);
        }
        if (!(a > 0.001) || (bitcast<u32>(a) & 0x7f800000u) == 0x7f800000u) {
            sout.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        }
        return sout;
    }
    if (in.extra.w > 1.5) {
        // the halo round a light in fog: licht.bmp turned to the
        // viewer, pulled towards them by its size, seen from in front of the light (for a
        // directional one the more, the further out of its cone's side the viewer stands)
        let to_cam = camera.cam_pos.xyz - in.pos;
        let dist = length(to_cam);
        let vdir = to_cam / max(dist, 0.001);
        var k = in.color.a;
        if (length(in.dir.xyz) > 0.5) {
            let axis = normalize(in.dir.xyz);
            let inner = in.extra.x;
            let outer = in.dir.w;
            let ang = acos(clamp(dot(axis, vdir), -1.0, 1.0));
            var from_side = select(0.0, 1.0, ang > inner);
            if (outer > inner + 0.0001) {
                from_side = clamp((ang - inner) / (outer - inner), 0.0, 1.0);
            }
            let side = min(from_side, clamp(3.0 - ang, 0.0, 1.0));
            let front = (1.0 - side) * clamp(1.0 - (ang - 2.0) * 0.5, 0.5, 1.0);
            k = k * front * front * select(0.0, 1.0, ang < 1.5705);
        }
        let fwd = normalize(cross(camera.cam_up.xyz, camera.cam_right.xyz));
        let on_screen = 2.0 * dot(fwd, -vdir) - 1.0;
        let far = clamp(1.0 - dist / max(in.extra.y, 1.0), 0.0, 1.0);
        k = k * max(on_screen, 0.0) * far;
        let right = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), vdir));
        let up = cross(vdir, right);
        var hout: CoronaOut;
        let centre = in.pos + vdir * min(in.size, dist * 0.9);
        hout.clip = camera.view_proj * vec4<f32>(centre + (right * c.x + up * c.y) * in.size, 1.0);
        hout.uv = c * 0.5 + 0.5;
        hout.color = vec4<f32>(in.color.rgb, k);
        hout.kind = 1.0;
        hout.star = 0.0;
        hout.beam = 0.0;
        hout.cone = vec2<f32>(0.0, 0.0);
        if (k <= 0.001 || on_screen <= 0.01) {
            hout.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        }
        return hout;
    }
    if (in.extra.w > 0.5) {
        // OMSI's cone in the fog: a flat
        // fan from the lamp, `size` in radius, spreading to the outer half angle either side
        // of the light's axis and turned about that axis to face the eye.
        // It is drawn here as a square in that plane; the fragment cuts the fan out of it
        // and maps light_cone.bmp onto it as the fan's vertices do.
        let axis = normalize(in.dir.xyz);
        let inner = in.extra.x;
        let outer = in.dir.w;
        let to_cam = camera.cam_pos.xyz - in.pos;
        let dist = length(to_cam);
        let vdir = to_cam / max(dist, 0.001);
        var side = cross(vdir, axis);
        if (length(side) < 0.001) {
            side = cross(vec3<f32>(0.0, 0.0, 1.0), axis);
        }
        side = normalize(side);
        // seen from the side it shows, from within its inner cone the glow does instead,
        // and from behind (2..3 rad off its axis) it fades out
        let ang = acos(clamp(dot(axis, vdir), -1.0, 1.0));
        var from_side = select(0.0, 1.0, ang > inner);
        if (outer > inner + 0.0001) {
            from_side = clamp((ang - inner) / (outer - inner), 0.0, 1.0);
        }
        let behind = clamp(3.0 - ang, 0.0, 1.0);
        // towards the edge of the view it fades (2 cos - 1 of the angle off the view's axis)
        let fwd = normalize(cross(camera.cam_up.xyz, camera.cam_right.xyz));
        let on_screen = 2.0 * dot(fwd, -vdir) - 1.0;
        // and it goes with the lamp's distance through the fog
        let far = clamp(1.0 - dist / max(in.extra.y, 1.0), 0.0, 1.0);
        let k = in.color.a * min(from_side, behind) * max(on_screen, 0.0) * far;
        var bout: CoronaOut;
        bout.clip = camera.view_proj * vec4<f32>(in.pos + (axis * c.x + side * c.y) * in.size, 1.0);
        bout.uv = c;
        bout.color = vec4<f32>(in.color.rgb, k);
        bout.kind = 1.0;
        bout.star = 0.0;
        bout.beam = 1.0;
        bout.cone = vec2<f32>(inner, outer);
        if (k <= 0.001 || on_screen <= 0.01) {
            bout.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        }
        return bout;
    }
    let to_cam = camera.cam_pos.xyz - in.pos;
    let dist = length(to_cam);
    let view_dir = to_cam / max(dist, 0.001);
    // (a cone of -2 marks a raindrop, of -3 a snowflake: rain.rs)
    let streak = in.dir.w < -1.5 && in.dir.w > -2.5;
    let flake = in.dir.w <= -2.5;
    let directed = length(in.dir.xyz) > 0.5 && !streak;
    var brightness = in.color.a;
    if (directed) {
        // a directional light is seen inside its outer cone, in full inside the inner one,
        // linearly in the angle between them
        let ang = acos(clamp(dot(normalize(in.dir.xyz), view_dir), -1.0, 1.0));
        let outer = acos(clamp(in.dir.w, -1.0, 1.0));
        let inner = select(0.0, acos(clamp(in.extra.x, -1.0, 1.0)), in.extra.x >= -1.0);
        brightness = brightness * clamp((outer - ang) / max(outer - inner, 0.0001), 0.0, 1.0);
    }
    // the light that reaches the eye through the fog: Beer-Lambert's exp(-density x
    // distance), the same law the scene's own fog follows (`shader.wgsl`). Left out, the
    // street lamps and signals shone at full strength to the end of the view in a fog that
    // had hidden their poles 50 m away, a fan of dots at the horizon (#1212).
    if (!streak) {
        brightness = brightness * exp(-dist * max(camera.fog.w, 0.0));
    }
    // a star grows with the light's strength; every sprite's strength stops at 1
    let star_sprite = (u32(in.extra.z + 0.5) & 8u) != 0u && !streak;
    let grow = select(1.0, brightness, star_sprite);
    if (!streak) {
        brightness = min(brightness, 1.0);
    }
    // the sprite's axes: flat and facing its direction (rotating 0), turned to the viewer
    // about its up axis (1), or turned to the viewer entirely (2)
    let mode = u32(in.up.w + 0.5);
    var right = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), view_dir));
    var up = cross(view_dir, right);
    let axis = select(vec3<f32>(0.0, 0.0, 1.0), normalize(in.up.xyz), length(in.up.xyz) > 0.5);
    if (!streak && mode == 0u && directed) {
        let n = normalize(in.dir.xyz);
        var u = axis - n * dot(axis, n);
        if (length(u) < 0.01) {
            u = vec3<f32>(0.0, 0.0, 1.0) - n * n.z;
        }
        up = normalize(u);
        right = cross(up, n);
    } else if (!streak && mode == 1u) {
        let r = cross(axis, view_dir);
        if (length(r) > 0.01) {
            right = normalize(r);
            up = axis;
        }
    }
    // coronas keep a minimum on-screen size in the distance like the original;
    // precipitation particles (cone < -1.5) follow their motion relative to the eye.
    // A sprite's radius, from the light's size (its diameter) and that distance floor. The
    // 0.9 is measured against Omsi.exe: at the size the game files ask for, every glow reads
    // a shade too wide beside the original, which draws the sprite a little inside the
    // diameter its `size` names. (Streaks keep their size: they are rain, not a light.)
    let size = select(max(in.size * grow, dist * 0.002) * 0.9 * select(glare, 1.0, flake), in.size, streak);
    let stretch = select(vec2<f32>(1.0, 1.0), vec2<f32>(0.06, 4.0), streak);
    var upv = up;
    if (streak) {
        upv = vec3<f32>(0.0, 0.0, 1.0);
        if (length(in.dir.xyz) > 0.5) { upv = normalize(in.dir.xyz); }
        let across = cross(upv, view_dir);
        if (length(across) > 0.001) { right = normalize(across); }
    }
    // the spot moved towards the viewer by its z offset, so that a lamp inside its housing
    // shows (without one: half its size, at most half a metre)
    let pull = select(min(size * 0.5, 0.5), in.extra.y, in.extra.y >= 0.0);
    let wp = in.pos + (right * c.x * stretch.x + upv * c.y * stretch.y) * size + select(view_dir * min(pull, dist * 0.9), vec3<f32>(0.0), streak);
    out.clip = camera.view_proj * vec4<f32>(wp, 1.0);
    // A lamp's glare quad (Enhanced, *Glow of lights*) is `GLARE_K` times its sprite and flat,
    // facing the eye. Left at the lamp's own depth it cuts into whatever stands within its
    // reach - the bumper, the road in front of the bus, the lens itself - along a straight
    // line, and fights with the lens's own depth: a bright plane standing in the air, seen
    // the plainer the lower the camera. A glare is the eye's and the lens's, not a thing in
    // the world: it is moved towards the eye in depth only (as a smoke puff is, above; its
    // size on the screen stays) by the quad's own radius, so that only what truly stands
    // between the eye and the lamp, further out than that, covers it.
    if (!streak && !flake && glare > 1.0) {
        let push = min(size, dist * 0.9);
        let nearer = camera.view_proj * vec4<f32>(wp + view_dir * push, 1.0);
        out.clip.z = nearer.z / nearer.w * out.clip.w;
    }
    out.uv = c * 0.5 + 0.5;
    out.color = vec4<f32>(in.color.rgb, brightness);
    out.kind = select(0.0, 1.0, streak);
    out.wpos = vec4<f32>(in.pos, select(select(0.0, 2.0, flake), 1.0, streak));
    out.star = 0.0;
    out.beam = 0.0;
    out.cone = vec2<f32>(0.0, 0.0);
    // (not `<= 0.001`: a NaN brightness is not smaller, and Direct3D drew it as a black
    // square where Metal's fast maths had dropped it)
    if (!(brightness > 0.001) || (bitcast<u32>(brightness) & 0x7f800000u) == 0x7f800000u) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    return out;
}

// The sprite's picture at `uv`, interpolated between its texels with a quintic ease instead
// of linearly. The pictures are small (light_cone.bmp is 64 x 64) and a lamp's sprite or a
// fog cone spans hundreds of pixels: a linear blend between texels has a kink at every texel
// line, and a steep gradient - the cone's bright axis, a glow's rim - shows them as a stair
// of squares along its edge, in the light of a lamp or a headlight's beam. The eased blend
// is smooth to the second order and costs the same single fetch.
fn soft_picture(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_corona));
    let p = uv * dims - vec2<f32>(0.5);
    let i = floor(p);
    let f = p - i;
    let e = f * f * f * (f * (f * 6.0 - vec2<f32>(15.0)) + vec2<f32>(10.0));
    // (the level of detail from the true coordinates: the eased ones run slower and faster)
    return textureSampleGrad(t_corona, s_corona, (i + e + vec2<f32>(0.5)) / dims, dpdx(uv), dpdy(uv));
}

// The sprite's light: the round texture (in its own colours), and for a star light four thin rays across it.
fn corona_shape(in: CoronaOut) -> vec3<f32> {
    // (the sprite keeps the bitmap's own colours, modulated by the light's colour)
    // Read the right way up. `in.uv` is built the way the HUD's is (overlay.wgsl), where
    // v = 0 is the picture's *first*, topmost row; but a sprite's corners are placed by its
    // own `up`, which runs up the screen, so v = 1 comes to rest at its top. Taken as it
    // stands the picture is drawn upside down - invisible on a round glow or a star, plain
    // on an asymmetric effect, such as the light streak some buses' `[light_enh_2]`
    // `lights_abbl`/`lights_stand` bitmaps draw. The fan below keeps the unflipped `in.uv`:
    // it maps light_cone.bmp from its own `fan_uv` and only takes `r` and `phi` from here,
    // which its geometry needs as they stand.
    let t = soft_picture(vec2<f32>(in.uv.x, 1.0 - in.uv.y)).rgb;
    // the cone's fan: its apex has uv (0, 1) and a rim vertex (sin a, 1 - cos a)
    // (u = sin a, v = 1 - cos a), a being 0.05 inside
    // the inner cone and rising to 0.9 pi/2 + 0.05 at the outer edge: light_cone.bmp's
    // bright left column runs down the light's axis and fades towards the fan's edges.
    // (With sin and cos the other way round the streaks ran along both edges, a "V" of
    // light reaching out sideways and up from every lamp.)
    let r = length(in.uv);
    let phi = abs(atan2(in.uv.y, in.uv.x));
    let span = max(in.cone.y - in.cone.x, 0.0001);
    let a = 0.05 + 0.9 * 1.5708 * clamp((phi - in.cone.x) / span, 0.0, 1.0);
    let fan_uv = clamp(vec2<f32>(0.0, 1.0) + min(r, 1.0) * vec2<f32>(sin(a), -cos(a)), vec2<f32>(0.0), vec2<f32>(1.0));
    let tb = soft_picture(fan_uv).r;
    if (in.beam > 0.5) {
        let inside = select(0.0, 1.0, r <= 1.0 && phi <= in.cone.y);
        return vec3<f32>(tb * inside);
    }
    return t;
}

@fragment
fn fs_main(in: CoronaOut) -> @location(0) vec4<f32> {
    var shape = corona_shape(in);
    if (in.kind < 0.5 && in.beam < 0.5) {
        // A lamp's glare ends softly at its sprite's edge whatever its bitmap does there: a
        // glow picture whose border is not quite black (a model's own, some a few levels
        // off) showed the whole square of the sprite round a bright lamp - the classic
        // picture adds the sprite as it is, and a brightness of 2 doubles the border too.
        // (the Enhanced picture rounds the sprite off, see `fs_enhanced`)
        let edge = max(abs(in.uv.x - 0.5), abs(in.uv.y - 0.5)) * 2.0;
        shape = shape * (1.0 - smoothstep(0.88, 1.0, edge));
    }
    return vec4<f32>(in.color.rgb * shape * in.color.a, 1.0);
}

// Enhanced graphics: the corona in the pre-exposed high-range picture - the glare of a
// lamp is drawn relative to what the eye is adapted to, brighter than white so that the
// glow filter picks it up; precipitation streaks keep the vanilla level.
@fragment
fn fs_enhanced(in: CoronaOut) -> @location(0) vec4<f32> {
    if (in.wpos.w > 0.5) {
        let to_eye = normalize(camera.cam_pos.xyz - in.wpos.xyz);
        let l = precip_light(in.wpos.xyz, to_eye, in.wpos.w > 1.5);
        return vec4<f32>(in.color.rgb * corona_shape(in) * in.color.a * l * enh.exposure.x, 1.0);
    }
    // (a fog cone is lit fog, not glare: it keeps the scene's level too)
    let scale = select(enh.exposure.w, enh.exposure.y, in.kind > 0.5);
    // (a lamp's quad is `GLARE_K` times its sprite: the sprite's own coordinates are these
    // stretched out from the middle, and the rest of the quad holds the glare)
    let lamp = in.kind < 0.5 && in.beam < 0.5;
    var qin = in;
    if (lamp) {
        qin.uv = (in.uv - vec2<f32>(0.5)) * glare_scale() + vec2<f32>(0.5);
    }
    var shape = corona_shape(qin);
    // (taken here, in uniform control flow: how much of the sprite one pixel spans)
    let r = length(qin.uv - vec2<f32>(0.5)) * 2.0;
    let px = fwidth(r);
    var tail = 0.0;
    // (only with the glow on: off, a lamp is drawn exactly as it was before the option)
    if (lamp && glare_scale() > 1.0) {
        // The glare of a lens (and of an eye) round a bright source: a long tail falling as the
        // cube of the distance (the near term of the CIE's glare function, 10 / theta^3), from
        // the sprite's rim out to the quad's edge, where it is let go. Mist, fog and rain
        // have their own halo, which the bitmap carries; in clear air this is what is left.
        let wet0 = clamp(enh.fog.x * 400.0 + enh.weather.z * 0.7, 0.0, 1.0);
        let k = 0.8;
        tail = GLARE_TAIL * k * k * k / (r * r * r + k * k * k) * (1.0 - smoothstep(GLARE_K * 0.6, GLARE_K, r)) * (1.0 - wet0);
    }
    if (lamp) {
        // A lamp seen through clear air is a small, very bright core with a faint ring of
        // glare round it, not a lit disc as wide as its sprite (OMSI's glow bitmaps are
        // flat discs with soft rims): a narrow core bright enough for the glow pass to
        // spread it the way a lens does, a short falloff of glare, and a dim trace of the
        // bitmap (which keeps a star's rays and a streak's shape). Mist, fog and rain
        // scatter the light round the lamp, and there the full halo comes back.
        let wet = clamp(enh.fog.x * 400.0 + enh.weather.z * 0.7, 0.0, 1.0);
        // (the core never narrower than a pixel and a half: a far lamp stays a point of
        // light, as the sprite's distance floor keeps it on the screen)
        let lit = textureSampleLevel(t_corona, s_corona, vec2<f32>(0.5, 0.5), 0.0).rgb;
        let w = max(0.1, px * 1.5);
        let core = exp(-r * r / (w * w)) * 3.0 + exp(-r * 9.0) * 0.35;
        shape = mix(core * max(lit, shape) + shape * 0.22, shape, wet);
        // (round, inside the circle the sprite holds: a far lamp's sprite is a few pixels,
        // the core spans it, and brighter than white even in its corners it showed the
        // whole quad - a square blot of light for every distant headlamp)
        shape = shape * (1.0 - smoothstep(1.0 - px, 1.0, r));
        // (the tail is added after the core is scaled, and soft-limited just under white:
        // pre-exposed at night it clipped to a flat white disc as wide as the sprite - a
        // plane of light - where it should fall off smoothly into the dark; the glow filter
        // takes the core's HDR peak for the halo)
        let halo = vec3<f32>(tail) * max(lit, vec3<f32>(0.25)) * in.color.a * scale;
        let soft = vec3<f32>(0.9) * (vec3<f32>(1.0) - exp(-halo / vec3<f32>(0.9)));
        return vec4<f32>(in.color.rgb * (shape * in.color.a * scale + soft), 1.0);
    }
    return vec4<f32>(in.color.rgb * shape * in.color.a * scale, 1.0);
}

// Smoke ([smoke] particles): the smoke texture tinted with the particle's colour, lit by the
// scene's ambient and sun light, blended with its alpha (the particle's times the texture's),
// faded out into the ground under it (see `vs_main`).
fn smoke_color(in: CoronaOut) -> vec4<f32> {
    // upside up, as the sprite's own picture above
    let t = textureSample(t_corona, s_corona, vec2<f32>(in.uv.x, 1.0 - in.uv.y));
    let light = min(camera.ambient.rgb + camera.sun_color.rgb * 0.6, vec3<f32>(1.2));
    let over_ground = select(1.0, smoothstep(0.0, in.ground.y, in.ground.x), in.ground.y > 0.0);
    return vec4<f32>(in.color.rgb * t.rgb * light, clamp(t.a * in.color.a * over_ground, 0.0, 1.0));
}

@fragment
fn fs_smoke(in: CoronaOut) -> @location(0) vec4<f32> {
    return smoke_color(in);
}

@fragment
fn fs_smoke_enhanced(in: CoronaOut) -> @location(0) vec4<f32> {
    let c = smoke_color(in);
    return vec4<f32>(c.rgb * enh.exposure.y, c.a);
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

