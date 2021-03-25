use anyhow::*;
use glow::*;
use image::GenericImageView;
use imgui::DrawVert;
use std::time::Duration;

fn main() -> Result<()> {
    unsafe { self::main_impl() }
}

const TITLE: &'static str = "SDL2 + glow";
const W: u32 = 1280;
const H: u32 = 720;

const VS_SRC: &str = include_str!("vs.glsl");
const FS_SRC: &str = include_str!("fs.glsl");

/// SDL window with OpenGL context
pub struct SdlHandles {
    pub sdl: sdl2::Sdl,
    pub vid: sdl2::VideoSubsystem,
    pub win: sdl2::video::Window,
    pub gl: sdl2::video::GLContext,
}

impl SdlHandles {
    fn new() -> Result<Self> {
        let sdl = sdl2::init().map_err(Error::msg)?;
        let vid = sdl.video().map_err(Error::msg)?;

        // GlCore33
        let attr = vid.gl_attr();
        attr.set_context_profile(sdl2::video::GLProfile::Core);
        attr.set_context_version(3, 3);

        let win = vid
            .window(TITLE, W, H)
            .position_centered()
            .opengl()
            // .resizable()
            .build()
            .map_err(Error::msg)?;

        let gl = win.gl_create_context().unwrap();

        Ok(Self { sdl, vid, win, gl })
    }

    pub fn swap_window(&self) {
        self.win.gl_swap_window();
    }
}

unsafe fn setup_vao(gl: &glow::Context, vao: glow::VertexArray) {
    gl.bind_vertex_array(Some(vao));

    let stride = std::mem::size_of::<imgui::DrawVert>() as i32;

    // pos: [f32: 2]
    let index = 0;
    gl.enable_vertex_attrib_array(index);
    gl.vertex_attrib_pointer_f32(
        // index: nth component of a vertex
        index,
        // size: number of `data_type`
        2,
        // data_type
        glow::FLOAT,
        // normalized
        false,
        // stride: size of vertex
        stride,
        // offset: byte offset of this component in a vertex
        0,
    );
    gl.enable_vertex_attrib_array(index);

    // uv: [f32: 2]
    let index = 1;
    gl.enable_vertex_attrib_array(index);
    gl.vertex_attrib_pointer_f32(
        // index
        index,
        // size
        2,
        // data_type
        glow::FLOAT,
        // normalized
        false,
        // stride
        stride,
        // offset
        2 * std::mem::size_of::<f32>() as i32,
    );

    // color: [u8: 2]
    let index = 2;
    gl.enable_vertex_attrib_array(index);
    gl.vertex_attrib_pointer_f32(
        index,
        // size
        4,
        // data_type
        glow::UNSIGNED_BYTE,
        // normalized
        true,
        // stride
        stride,
        // offset
        4 * std::mem::size_of::<f32>() as i32,
    );

    // TODO:
    // gl.set_vertex_array(None);
}

unsafe fn gen_shader_program(gl: &glow::Context, sources: &[(u32, &str)]) -> glow::Program {
    let program = gl.create_program().expect("Cannot create program");

    let mut shaders = Vec::with_capacity(sources.len());

    for (type_, src) in sources.iter() {
        let shader = gl.create_shader(*type_).expect("Cannot create shader");
        gl.shader_source(shader, &src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            panic!(gl.get_shader_info_log(shader));
        }
        gl.attach_shader(program, shader);
        shaders.push(shader);
    }

    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        panic!(gl.get_program_info_log(program));
    }

    for shader in shaders {
        gl.detach_shader(program, shader);
        gl.delete_shader(shader);
    }

    program
}

unsafe fn alloc_buffer(gl: &glow::Context, type_: u32, capacity: usize) -> Result<glow::Buffer> {
    let buf = gl.create_buffer().map_err(Error::msg)?;
    gl.bind_buffer(type_, Some(buf));
    // gl.buffer_data_size(type_, capacity as i32, glow::STREAM_DRAW);
    gl.buffer_data_size(type_, capacity as i32, glow::STATIC_DRAW);
    gl.bind_buffer(type_, None);
    Ok(buf)
}

unsafe fn gen_texture(gl: &glow::Context, pixels: &[u8], w: u32, h: u32) -> Result<glow::Texture> {
    let tex = gl.create_texture().map_err(Error::msg)?;

    gl.bind_texture(glow::TEXTURE_2D, Some(tex));

    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MAG_FILTER,
        glow::LINEAR as i32,
    );

    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,                 // level
        glow::RGBA as i32, // internal format
        w as i32,
        h as i32,
        0,          // border
        glow::RGBA, // format
        glow::UNSIGNED_BYTE,
        Some(pixels),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);

    gl.bind_texture(glow::TEXTURE_2D, None);

    Ok(tex)
}

pub struct Resources {
    pub vao: glow::VertexArray,
    pub program: glow::Program,
    pub vbuf: glow::Buffer,
    pub ibuf: glow::Buffer,
    pub tex: glow::Texture,
}

impl Resources {
    /// Allocates GPU resources without setting contents
    pub unsafe fn new(gl: &glow::Context) -> Result<Self> {
        let vao = gl
            .create_vertex_array()
            .expect("Cannot create vertex array");
        self::setup_vao(gl, vao);

        let sources = [
            (glow::VERTEX_SHADER, VS_SRC),
            (glow::FRAGMENT_SHADER, FS_SRC),
        ];
        let program = self::gen_shader_program(gl, &sources);

        let vbuf = self::alloc_buffer(gl, glow::ARRAY_BUFFER, 4 * 2048 * 20)?;
        let ibuf = self::alloc_buffer(gl, glow::ELEMENT_ARRAY_BUFFER, 6 * 2048 * 2)?;

        let img = include_bytes!("ika-chan.png");
        let img = image::load_from_memory(img).expect("Unable to load image");
        let tex = self::gen_texture(gl, img.as_bytes(), img.width(), img.height())?;

        Ok(Self {
            vao,
            program,
            vbuf,
            ibuf,
            tex,
        })
    }

    pub unsafe fn free(self, gl: &glow::Context) {
        gl.delete_program(self.program);
        gl.delete_vertex_array(self.vao);
        gl.delete_buffer(self.vbuf);
        gl.delete_buffer(self.ibuf);
        // FIXME: image ownership
        gl.delete_texture(self.tex);
    }

    pub unsafe fn set_uniforms(&self, gl: &glow::Context, mat: [f32; 16]) {
        let location = gl
            // we must not add '\0' here -- glow does it
            .get_uniform_location(self.program, "transform")
            .expect("Unable to locate transform uniform");
        gl.uniform_matrix_4_f32_slice(Some(&location), false, &mat);

        // FIXME:
        // let location = gl
        //     // we must not add '\0' here -- glow does it
        //     .get_uniform_location(self.program, "tex")
        //     .expect("Unable to locate texture uniform");
        // gl.uniform_1_i32(Some(&location), 0);
    }

    pub unsafe fn bind(&self, gl: &glow::Context) {
        self::setup_vao(gl, self.vao);
        // gl.bind_vertex_array(Some(self.vao));
        gl.use_program(Some(self.program));
        gl.bind_texture(glow::TEXTURE_2D, Some(self.tex));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbuf));
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ibuf));

        // TODO:
        // gl.enable(glow::BLEND);
        // TODO:
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::CULL_FACE);
        gl.disable(glow::STENCIL_TEST);
        gl.disable(glow::SCISSOR_TEST);
    }

    pub unsafe fn unbind(gl: &glow::Context) {
        gl.bind_vertex_array(None);
        gl.use_program(None);
        // gl.bind_texture(glow::TEXTURE_2D, None);
        // gl.bind_buffer(glow::ARRAY_BUFFER, None);
        // gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
    }

    pub unsafe fn draw(&self, gl: &glow::Context, base_elem: i32, n_elems: i32) {
        gl.draw_elements(
            // mode
            glow::TRIANGLES,
            // count
            n_elems as i32,
            // element_type: u16 index (unsighned short)
            glow::UNSIGNED_SHORT,
            // offset FIXME: in bytes?
            base_elem * std::mem::size_of::<imgui::DrawIdx>() as i32,
        );
    }
}

pub fn ortho_mat_gl(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [f32; 16] {
    [
        (2.0 / (right as f64 - left as f64)) as f32,
        0.0,
        0.0,
        0.0,
        // ---
        0.0,
        (2.0 / (top as f64 - bottom as f64)) as f32,
        0.0,
        0.0,
        // ---
        0.0,
        0.0,
        -(1.0 / (far as f64 - near as f64)) as f32,
        0.0,
        // ---
        -((right as f64 + left as f64) / (right as f64 - left as f64)) as f32,
        -((top as f64 + bottom as f64) / (top as f64 - bottom as f64)) as f32,
        (near as f64 / (near as f64 - far as f64)) as f32,
        1.0,
    ]
}

/// Setup allocated resource contents
///
/// NOTE: It has to be called while targetting VAO is binded..?
unsafe fn setup_resources(gl: &glow::Context, res: &mut Resources) -> Result<()> {
    let col = [255, 255, 255, 255];

    let verts = [
        // left-up
        DrawVert {
            // pos: [100.0, 100.0],
            pos: [0.0, 0.0],
            uv: [0.0, 0.0],
            col,
        },
        // right-up
        DrawVert {
            pos: [300.0, 100.0],
            uv: [1.0, 0.0],
            col,
        },
        // left-down
        DrawVert {
            pos: [100.0, 300.0],
            uv: [0.0, 1.0],
            col,
        },
        // left-down
        DrawVert {
            pos: [300.0, 300.0],
            uv: [1.0, 1.0],
            col,
        },
    ];

    let bytes = std::slice::from_raw_parts(
        verts.as_ptr() as *const _,
        verts.len() * std::mem::size_of::<DrawVert>(),
    );
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(res.vbuf));
    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);

    let indices: [imgui::DrawIdx; 6] = [0, 1, 2, 3, 1, 2];
    let bytes = std::slice::from_raw_parts(
        indices.as_ptr() as *const _,
        indices.len() * std::mem::size_of::<DrawVert>(),
    );
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(res.ibuf));
    gl.buffer_sub_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, 0, bytes);

    Ok(())
}

unsafe fn main_impl() -> Result<()> {
    // create SDL window and OpenGL context
    let handles = SdlHandles::new()?;
    // create glow context
    let gl =
        glow::Context::from_loader_function(|s| handles.vid.gl_get_proc_address(s) as *const _);

    let mut res = Resources::new(&gl)?;
    self::setup_resources(&gl, &mut res)?;

    gl.clear_color(0.1, 0.2, 0.3, 1.0);

    let mut pump = handles.sdl.event_pump().map_err(Error::msg)?;
    let dt = Duration::from_nanos(1_000_000_000 / 60);
    'running: loop {
        for event in pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        gl.clear(glow::COLOR_BUFFER_BIT);

        res.bind(&gl);
        let mat = self::ortho_mat_gl(
            0.0,      //
            W as f32, // right
            H as f32, // bottom
            0.0,      // top
            // FIXME: which is coorect
            1.0, // near
            0.0, // far
        );
        res.set_uniforms(&gl, mat);

        res.draw(&gl, 0, 6);
        Resources::unbind(&gl);

        handles.swap_window();
        std::thread::sleep(dt);
    }

    res.free(&gl);

    Ok(())
}
