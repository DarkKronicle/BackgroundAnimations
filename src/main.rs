//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        RenderApp, RenderStage,
        view::RenderLayers,
    },
    reflect::TypeUuid,
    window::WindowDescriptor,
    sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle},
};
use std::borrow::Cow;
use bevy_framepace::{FramepaceSettings, Limiter};

const SIZE: (u32, u32) = (1840,  1840);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(WindowDescriptor {
            // uncomment for unthrottled FPS
            // present_mode: bevy::window::PresentMode::AutoNoVsync,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(GameOfLifeComputePlugin)
        .add_plugin(Material2dPlugin::<PostProcessingMaterial>::default())
        .add_plugin(bevy_framepace::FramepacePlugin)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut post_processing_materials: ResMut<Assets<PostProcessingMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut settings: ResMut<FramepaceSettings>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    settings.limiter = Limiter::from_framerate(30.);
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.spawn_bundle(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        texture: image.clone(),
        ..default()
    });
    commands.spawn_bundle(Camera2dBundle::default());

    let post_processing_pass_layer = RenderLayers::layer((RenderLayers::TOTAL_LAYERS - 1) as u8);
    let quad_handle = meshes.add(Mesh::from(shape::Quad::new(Vec2::new(
        SIZE.0 as f32,
        SIZE.1 as f32,
    ))));

    let material_handle = post_processing_materials.add(PostProcessingMaterial {
        size_x: SIZE.0,
        size_y: SIZE.1,
        source_image: image.clone(),
    });

    commands.spawn_bundle(MaterialMesh2dBundle {
            mesh: quad_handle.into(),
            material: material_handle,
            transform: Transform {
                translation: Vec3::new(0.0, 0.0, 1.5),
                ..default()
            },
            ..default()
        })
        .insert(post_processing_pass_layer);

    // The post-processing pass camera.
    commands.spawn_bundle(Camera2dBundle {
            camera: Camera {
                // renders after the first main camera which has default value: 0.
                priority: 1,
                ..default()
            },
            ..Camera2dBundle::default()
        })
        .insert(post_processing_pass_layer);

    commands.insert_resource(GameOfLifeImage(image));
}

pub struct GameOfLifeComputePlugin;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugin(ExtractResourcePlugin::<GameOfLifeImage>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<GameOfLifePipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("game_of_life", GameOfLifeNode::default());
        render_graph
            .add_node_edge(
                "game_of_life",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
    }
}

#[derive(Clone, Deref, ExtractResource)]
struct GameOfLifeImage(Handle<Image>);

struct GameOfLifeImageBindGroup(BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_image: Res<GameOfLifeImage>,
    render_device: Res<RenderDevice>,
) {
    let view = &gpu_images[&game_of_life_image.0];
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.texture_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&view.texture_view),
        }],
    });
    commands.insert_resource(GameOfLifeImageBindGroup(bind_group));
}

pub struct GameOfLifePipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/worms.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum GameOfLifeState {
    Loading,
    Init,
    Update,
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = GameOfLifeState::Init;
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = GameOfLifeState::Update;
                }
            }
            GameOfLifeState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<GameOfLifeImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GameOfLifePipeline>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}

/// Our custom post processing material
#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "bc2f08eb-a0fb-43f1-a908-54871ea597d5"]
struct PostProcessingMaterial {
    /// In this example, this image will be the result of the main camera.
    #[texture(0)]
    #[sampler(1)]
    source_image: Handle<Image>,
    #[uniform(2)]
    size_x: u32,
    #[uniform(3)]
    size_y: u32,
}

impl Material2d for PostProcessingMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/worms_fragment.wgsl".into()
    }
}