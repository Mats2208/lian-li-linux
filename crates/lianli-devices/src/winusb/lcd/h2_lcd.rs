use anyhow::Result;
use lianli_shared::screen::ScreenInfo;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use super::core::{SharedTransport, WinUsbLcdCore};
use super::trait_::{BoxedWinUsbLcd, WinUsbLcd};

pub struct H2WinUsbLcd {
    core: WinUsbLcdCore,
}

impl H2WinUsbLcd {
    pub(crate) fn new(
        device: rusb::Device<rusb::GlobalContext>,
        screen: ScreenInfo,
        name: &str,
    ) -> Result<Self> {
        let core = WinUsbLcdCore::open(device, screen, name, WRITE_TIMEOUT, READ_TIMEOUT)?;
        Ok(Self { core })
    }

    pub(crate) fn from_shared(
        transport: SharedTransport,
        screen: ScreenInfo,
        name: String,
    ) -> Self {
        Self {
            core: WinUsbLcdCore::from_shared(transport, screen, name, WRITE_TIMEOUT, READ_TIMEOUT),
        }
    }

    fn do_init(&mut self) -> Result<()> {
        self.core.init_logging();
        self.core.reset_failure_state();
        self.core.read_firmware();
        // FIX: this AIO never answers GetVer and set_frame_rate can fail
        // transiently. The `?` aborted do_init and left the shared control
        // channel unusable, taking fans and RGB down with it. Degrade instead.
        if let Err(e) = self.core.set_frame_rate(30) {
            tracing::warn!("set_frame_rate failed, continuing anyway: {e:#}");
        }
        let sync = self.core.builder_mut().sync_clock_header_winusb(2);
        self.core.send_command(sync, "SyncClock");
        let stop_clock = self.core.builder_mut().stop_clock_header_winusb();
        self.core.send_command(stop_clock, "StopClock");
        // FIX: clear_layers() paints a black 480x480 PNG over the panel. With
        // `lcds: []` the daemon never draws anything afterwards, so the screen
        // stays black and the firmware's own content is wiped. Media streaming
        // overwrites every frame anyway, so clearing here buys nothing.
        // self.core.clear_layers();
        self.core.initialized = true;
        Ok(())
    }
}

const WRITE_TIMEOUT: Duration = Duration::from_millis(200);
const READ_TIMEOUT: Duration = Duration::from_millis(2_000);

impl WinUsbLcd for H2WinUsbLcd {
    fn screen_info(&self) -> &ScreenInfo {
        self.core.screen()
    }
    fn firmware_str(&self) -> Option<&str> {
        self.core.firmware_str()
    }
    fn shared_transport(&self) -> SharedTransport {
        self.core.shared_transport()
    }
    fn transport_release(&self) {
        self.core.transport_release()
    }
    fn initialize(&mut self) -> Result<()> {
        if !self.core.initialized {
            self.do_init()?;
        }
        Ok(())
    }
    fn send_frame(&mut self, frame: &[u8]) -> Result<()> {
        if !self.core.initialized {
            self.do_init()?;
        }
        self.core.send_frame(frame)
    }
    fn send_frame_verified(&mut self, frame: &[u8]) -> Result<()> {
        if !self.core.initialized {
            self.do_init()?;
        }
        self.core.send_frame_verified(frame)
    }
    fn set_brightness_val(&mut self, brightness: u8) -> Result<()> {
        self.core.set_brightness_val(brightness)
    }
    fn switch_to_desktop_mode(&mut self) -> Result<()> {
        self.core.switch_to_desktop_mode()
    }
    fn stream_h264(
        &mut self,
        path: &Path,
        looping: bool,
        stop: &AtomicBool,
        fps: f32,
    ) -> Result<()> {
        if !self.core.initialized {
            self.do_init()?;
        }
        self.core.apply_stream_fps(fps)?;
        self.core
            .stream_h264(path, looping, stop, fps, self.core.screen().play_count, 0)
    }
    fn stream_h264_reader(
        &mut self,
        reader: &mut dyn std::io::Read,
        stop: &AtomicBool,
        fps: f32,
    ) -> Result<()> {
        if !self.core.initialized {
            self.do_init()?;
        }
        self.core.apply_stream_fps(fps)?;
        self.core
            .stream_h264_reader(reader, stop, self.core.screen().play_count, 0)
    }
}

pub(crate) fn boxed(
    device: rusb::Device<rusb::GlobalContext>,
    screen: ScreenInfo,
    name: &str,
) -> Result<BoxedWinUsbLcd> {
    Ok(Box::new(H2WinUsbLcd::new(device, screen, name)?))
}
