#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils

@group(1) @binding(0)
var texture: texture_2d<f32>;

@group(1) @binding(1)
var our_sampler: sampler;

@group(1) @binding(2)
var<uniform> width: u32;

@group(1) @binding(3)
var<uniform> height: u32;

fn sampleTexture(x: f32, y: f32) -> f32 {
    return textureSample(texture, our_sampler, vec2<f32>(x, y)).r * 2. - 1.;
}

@fragment
fn fragment(
    @builtin(position) position: vec4<f32>,
    #import bevy_sprite::mesh2d_vertex_output
) -> @location(0) vec4<f32> {

    // Sample each color channel with an arbitrary shift
    var pixel_width = 1. / f32(width);
    var pixel_height = 1. / f32(height);
    var x = position.x * pixel_width;
    var y = position.y * pixel_height;
//    var sampled = (sampleTexture(x, y)
//                       + sampleTexture(x - pixel_width, y - pixel_height)
//                       + sampleTexture(x    , y - pixel_height)
//                       + sampleTexture(x + pixel_width, y - pixel_height)
//                       + sampleTexture(x - pixel_width, y + pixel_height)
//                       + sampleTexture(x   , y + pixel_height)
//                       + sampleTexture(x + pixel_width, y + pixel_height)
//                       + sampleTexture(x - pixel_width, y)
//                       + sampleTexture(x + pixel_width, y)) / 9.;
    var sampled = sampleTexture(x, y);
    var color = 0.;
    if (sampled > 0.35) {
        color = 1.;
    } else if (sampled > .15) {
        color = .5;
    }
    var output_color = vec4(color * .25, color * .05, color * .4, 1.);

    return output_color;
}