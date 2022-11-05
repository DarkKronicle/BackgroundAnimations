@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let randomNumber = randomFloat(u32(randomFloat(hash(u32(i32(num_workgroups.x) * location.x + location.y))) * 5123.0f) + u32(randomFloat(invocation_id.x) * 2340.0f) * invocation_id.y * num_workgroups.x + invocation_id.x + u32(location.y));
    let color = vec4<f32>(randomNumber);

    textureStore(texture, location, color);
}

fn get_value(location: vec2<i32>, offset_x: i32, offset_y: i32) -> f32 {
    let value: vec4<f32> = textureLoad(texture, location + vec2<i32>(offset_x, offset_y));
    return value.x * 2.f - 1.f;
}

fn calculate_convolution(location: vec2<i32>) -> f32 {
    return get_value(location, -1, -1) * .68f +
           get_value(location, -1,  0) * -.9f +
           get_value(location, -1,  1) * .68f +
           get_value(location,  0, -1) * -.9f +
           get_value(location,  0,  0) * -.66f +
           get_value(location,  0,  1) * -.9f +
           get_value(location,  1, -1) * .68f +
           get_value(location,  1,  0) * -.9f +
           get_value(location,  1,  1) * .68f;
}

fn activation(x: f32) -> f32 {
    return -1.0f / pow(2.0f, (0.6f*pow(x, 2.0f))) + 1.0f;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let conv = calculate_convolution(location);
    let act = activation(conv);

    let color = vec4<f32>(act / 2.f + .5);

    storageBarrier();

    textureStore(texture, location, color);
}