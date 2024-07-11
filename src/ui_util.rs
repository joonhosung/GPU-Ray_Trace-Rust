use std::sync::Arc;
use egui::mutex::Mutex;
use eframe::{glow, egui_glow};
use glow::PixelUnpackData;
use crate::{RenderOut, ArcMux, BufferMux};

pub fn ui_on_render_out(render_out: Arc<RenderOut>, (region_width, region_height): (i32, i32)) -> Result<(), eframe::Error> {
    eframe::run_native(
        "Ray Tracing in Rust!",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([region_width as f32 + 10.0, region_height as f32 + 100.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx); // image support
            Box::new(MyApp::new(cc, region_width, region_height, render_out))
        })
    )
}

struct MyApp {
    gl_renderer: ArcMux<GLDrawer>,
    canv_width: i32, canv_height: i32,
    render_out: Arc<RenderOut>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                ui.heading("Demo!!!");
                self.gl_paint(ui, ctx);
            });
        });
    }
}

impl MyApp {
    fn new(cc: &eframe::CreationContext, canv_width: i32, canv_height: i32, render_out: Arc<RenderOut>) -> Self {
        let gl = cc.gl.as_ref().expect("cannot get gl context!");
        let gl_renderer = Arc::new(Mutex::new(GLDrawer::new(gl, canv_width, canv_height)));

        Self {
            gl_renderer,
            canv_width, canv_height,
            render_out,
        }
    }
    fn gl_paint(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (rect, _response) = ui.allocate_exact_size(egui::Vec2::new(self.canv_width as f32, self.canv_height as f32), egui::Sense::focusable_noninteractive());
        
        let gl_renderer = self.gl_renderer.clone();
        let _outer_rect = ctx.input(|i| i.viewport().outer_rect).unwrap();
        let new_ctx = ctx.clone();
        let buf_avail = self.render_out.buffer_avail.clone();
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let mut renderer = gl_renderer.lock();
                if let Some(buf) = buf_avail.lock().take() {
                    renderer.dump_content(buf);
                }
                renderer.paint(painter.gl(), rect, new_ctx.clone());
            })),
        };

        ui.painter().add(callback);
    }
}

struct GLDrawer {
    texture: glow::NativeTexture,
    framebuffer: glow::NativeFramebuffer,
    canv_width: i32, canv_height: i32,
    new_data: Option<BufferMux>,
}

impl GLDrawer {
    pub fn new(gl: &glow::Context, canv_width: i32, canv_height: i32) -> Self { 
        use glow::HasContext;

        let (texture, framebuffer) = unsafe {
            let framebuffer = gl.create_framebuffer().unwrap();
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, canv_width, canv_height);
            gl.bind_texture(glow::TEXTURE_2D, None);
            
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, 
                            glow::COLOR_ATTACHMENT0, 
                            glow::TEXTURE_2D,
                            Some(texture), 0
                            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);

            (texture, framebuffer)
        };
        Self { texture, framebuffer, canv_width, canv_height, new_data: None }
    }
    fn dump_content(&mut self, data: BufferMux) {
        self.new_data = Some(data);
    }
    pub fn paint(&mut self, gl: &glow::Context, rect: egui::Rect, ctx: egui::Context) {
        use glow::HasContext;

        unsafe {
            if let Some(new_data) = self.new_data.take() {
                let data = new_data.lock();
                gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D, 0,
                    0, 0, self.canv_width, self.canv_height,
                    glow::RGBA, glow::UNSIGNED_BYTE,
                    PixelUnpackData::Slice(&data),
                );
                gl.bind_texture(glow::TEXTURE_2D, None);
            }

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.framebuffer));
            gl.blit_framebuffer(
                0, 0, self.canv_width, self.canv_height,
                rect.min.x as i32, 40, rect.width() as i32 + rect.min.x as i32, rect.height() as i32 + 40,
                glow::COLOR_BUFFER_BIT, 
                glow::LINEAR
            );

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
        ctx.request_repaint();
    }
}