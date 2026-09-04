use super::renderers::{
    AsyncCustomH264Renderer, AsyncCustomRenderer, AsyncSensorH264Renderer, AsyncSensorRenderer,
    AsyncVideoPlayer,
};
use super::DaemonEvent;
use lianli_devices::crypto::PacketBuilder;
use lianli_devices::slv3_lcd::Slv3LcdDevice;
use lianli_devices::traits::LcdDevice;
use lianli_devices::winusb::lcd::WinUsbLcdDevice;
use lianli_devices::wireless::WirelessController;
use lianli_media::{MediaAsset, MediaAssetKind};
use lianli_shared::config::ConfigKey;
use lianli_shared::screen::ScreenInfo;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::process::ChildStdout;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::{debug, info, warn};

pub(super) type SharedHidLcd = Arc<Mutex<Box<dyn LcdDevice>>>;

// lock per access unit only, never for the whole stream
// no sane access unit comes close; malformed streams without a second
// boundary would otherwise grow `accum` without limit
const MAX_AU_BYTES: usize = 4 * 1024 * 1024;

/// Returns the join handle plus the worker's private halt flag so a
/// replacement stream can stop this one promptly.
fn spawn_hid_h264_stream(
    lcd: SharedHidLcd,
    mut reader: Box<dyn std::io::Read + Send>,
    stop: Arc<AtomicBool>,
    fps: f32,
) -> (JoinHandle<()>, Arc<AtomicBool>) {
    use lianli_devices::hydroshift_lcd::{find_au_split, pace_frame};
    use std::io::Read;
    use std::time::Instant;

    let halt = Arc::new(AtomicBool::new(false));
    let worker_halt = Arc::clone(&halt);
    let worker_stop = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let halted = || worker_stop.load(Ordering::Relaxed) || worker_halt.load(Ordering::Relaxed);
        let fps = {
            let mut guard = lcd.lock();
            guard.set_stream_fps(fps)
        };
        let frame_interval = Duration::from_secs_f32(1.0 / fps);
        let mut read_buf = vec![0u8; 64 * 1024];
        let mut accum: Vec<u8> = Vec::with_capacity(256 * 1024);
        let mut next_deadline = Instant::now() + frame_interval;
        // residual data is only flushed on a clean EOF, never after a stop
        // or error exit where it may be a partial access unit
        let mut clean_eof = false;
        loop {
            if halted() {
                break;
            }
            let n = match reader.read(&mut read_buf) {
                Ok(n) => n,
                Err(e) => {
                    warn!("HID h264 stream read error: {e:#}");
                    break;
                }
            };
            if n == 0 {
                clean_eof = true;
                break;
            }
            accum.extend_from_slice(&read_buf[..n]);
            if accum.len() > MAX_AU_BYTES {
                warn!(
                    "HID h264 stream: no AU boundary within {} bytes, aborting",
                    MAX_AU_BYTES
                );
                return;
            }
            while let Some(split) = find_au_split(&accum) {
                let au: Vec<u8> = accum.drain(..split).collect();
                if au.is_empty() {
                    continue;
                }
                let result = {
                    let mut guard = lcd.lock();
                    guard.send_h264_frame(&au)
                };
                if let Err(e) = result {
                    warn!("HID h264 stream send error: {e:#}");
                    return;
                }
                pace_frame(&mut next_deadline, frame_interval);
                if halted() {
                    return;
                }
            }
        }
        if clean_eof && !halted() && !accum.is_empty() {
            pace_frame(&mut next_deadline, frame_interval);
            if let Err(e) = lcd.lock().send_h264_frame(&accum) {
                debug!("HID h264 flush: {e:#}");
            }
        }
    });
    (handle, halt)
}

pub(super) enum LcdBackend {
    Slv3(Slv3LcdDevice),
    WinUsb(ThreadedWinUsbSender),
    HidLcd(SharedHidLcd),
}

/// The LCD mutex could not be taken within the bounded wait, meaning the
/// init worker is holding it across its long settle and firmware retries.
/// Frame sends treat it as a retry later rather than an error so the
/// streaming thread never tears down a target that is merely initializing.
#[derive(Debug)]
struct LcdBusy;

impl std::fmt::Display for LcdBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LCD busy (initializing)")
    }
}
impl std::error::Error for LcdBusy {}

/// How long a frame send waits for the LCD mutex before deferring.
const LCD_BUSY_WAIT: Duration = Duration::from_millis(100);

impl LcdBackend {
    fn send_frame(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
        frame: &[u8],
    ) -> anyhow::Result<()> {
        match self {
            Self::Slv3(d) => {
                if let Some(w) = wireless {
                    w.ensure_video_mode()?;
                }
                d.send_frame(builder, frame)
            }
            Self::WinUsb(d) => d.send_frame(frame),
            Self::HidLcd(d) => {
                let Some(mut guard) = d.try_lock_for(LCD_BUSY_WAIT) else {
                    return Err(LcdBusy.into());
                };
                guard.send_jpeg_frame(frame)
            }
        }
    }

    fn send_frame_verified(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
        frame: &[u8],
    ) -> anyhow::Result<()> {
        match self {
            Self::WinUsb(d) => d.send_frame_verified(frame),
            Self::HidLcd(d) => {
                let Some(mut guard) = d.try_lock_for(LCD_BUSY_WAIT) else {
                    return Err(LcdBusy.into());
                };
                guard.send_static_frame(frame)
            }
            _ => self.send_frame(wireless, builder, frame),
        }
    }

    pub(super) fn set_brightness(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
        brightness: u8,
    ) -> anyhow::Result<()> {
        match self {
            Self::Slv3(d) => {
                if let Some(w) = wireless {
                    w.ensure_video_mode()?;
                }
                d.set_brightness(builder, brightness)
            }
            Self::WinUsb(sender) => sender.set_brightness(brightness),
            Self::HidLcd(d) => d.lock().set_brightness(brightness),
        }
    }

    pub(super) fn start_h264_stream(
        &self,
        stdout: ChildStdout,
        stop: Arc<AtomicBool>,
        fps: f32,
    ) -> anyhow::Result<Option<HidStreamWorker>> {
        match self {
            Self::HidLcd(lcd) => {
                let (handle, halt) =
                    spawn_hid_h264_stream(Arc::clone(lcd), Box::new(stdout), stop, fps);
                Ok(Some(HidStreamWorker { handle, halt }))
            }
            Self::WinUsb(sender) => {
                sender.stream_h264_reader(stdout, fps)?;
                Ok(None)
            }
            _ => anyhow::bail!("h264 streaming not supported on this backend"),
        }
    }

    /// Restart-capable handle for render threads; takes the initial worker
    /// so the first restart can stop it.
    pub(super) fn stream_restarter(
        &self,
        initial: Option<HidStreamWorker>,
    ) -> Option<StreamRestarter> {
        match self {
            Self::HidLcd(lcd) => Some(StreamRestarter::HidLcd(
                Arc::clone(lcd),
                Mutex::new(initial),
            )),
            Self::WinUsb(sender) => Some(StreamRestarter::WinUsb(
                sender.tx.clone(),
                Arc::clone(&sender.h264_stop),
            )),
            Self::Slv3(_) => None,
        }
    }
}

pub(super) struct HidStreamWorker {
    handle: JoinHandle<()>,
    halt: Arc<AtomicBool>,
}

impl HidStreamWorker {
    /// The worker may be parked reading an encoder stdout that only EOFs
    /// once the caller replaces the encoder, so join is bounded.
    fn stop(&self, timeout: Duration) {
        self.halt.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + timeout;
        while !self.handle.is_finished() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(25));
        }
        if !self.handle.is_finished() {
            debug!("h264 stream worker did not stop within {timeout:?}, detaching");
        }
    }
}

/// Cloneable handle for restarting an h264 stream from inside a render thread.
pub(super) enum StreamRestarter {
    HidLcd(SharedHidLcd, Mutex<Option<HidStreamWorker>>),
    WinUsb(std::sync::mpsc::SyncSender<LcdThreadMsg>, Arc<AtomicBool>),
}

impl StreamRestarter {
    /// Start a new h264 stream reading from the given stdout. The old
    /// stream is halted and joined (bounded) first.
    pub(super) fn start_stream(
        &self,
        stdout: ChildStdout,
        stop: Arc<AtomicBool>,
        fps: f32,
    ) -> anyhow::Result<()> {
        match self {
            Self::HidLcd(lcd, current) => {
                let mut current = current.lock();
                if let Some(old) = current.take() {
                    old.stop(Duration::from_secs(1));
                }
                let (handle, halt) =
                    spawn_hid_h264_stream(Arc::clone(lcd), Box::new(stdout), stop, fps);
                *current = Some(HidStreamWorker { handle, halt });
                Ok(())
            }
            Self::WinUsb(tx, h264_stop) => {
                h264_stop.store(true, Ordering::Relaxed);
                tx.send(LcdThreadMsg::StreamH264Reader(stdout, fps))
                    .map_err(|_| anyhow::anyhow!("LCD sender thread exited"))
            }
        }
    }
}

pub(super) enum LcdThreadMsg {
    Frame(Vec<u8>),
    FrameVerified(Vec<u8>, std::sync::mpsc::SyncSender<anyhow::Result<()>>),
    StreamH264 {
        path: PathBuf,
        looping: bool,
        fps: f32,
    },
    StreamH264Reader(std::process::ChildStdout, f32),
    SwitchDesktop(std::sync::mpsc::SyncSender<anyhow::Result<()>>),
    SetBrightness(u8),
    Stop,
}

pub(super) struct ThreadedWinUsbSender {
    tx: std::sync::mpsc::SyncSender<LcdThreadMsg>,
    h264_stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ThreadedWinUsbSender {
    pub(super) fn new(mut device: WinUsbLcdDevice, index: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<LcdThreadMsg>(2);
        let h264_stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&h264_stop);
        let thread = thread::spawn(move || {
            for msg in rx {
                match msg {
                    LcdThreadMsg::Frame(data) => {
                        if let Err(e) = device.send_frame(&data) {
                            if lianli_transport::usb::shutting_down() {
                                debug!("LCD[{index}] frame send refused during shutdown: {e:#}");
                            } else {
                                warn!("LCD[{index}] sender thread frame error: {e}");
                            }
                        }
                    }
                    LcdThreadMsg::FrameVerified(data, reply) => {
                        let result = device.send_frame_verified(&data);
                        let _ = reply.send(result);
                    }
                    LcdThreadMsg::StreamH264 { path, looping, fps } => {
                        stop_clone.store(false, Ordering::Relaxed);
                        if let Err(e) = device.stream_h264(&path, looping, &stop_clone, fps) {
                            if lianli_transport::usb::shutting_down() {
                                debug!("LCD[{index}] h264 stream ended by shutdown: {e:#}");
                            } else {
                                warn!("LCD[{index}] h264 stream error: {e}");
                            }
                        }
                    }
                    LcdThreadMsg::StreamH264Reader(mut stdout, fps) => {
                        stop_clone.store(false, Ordering::Relaxed);
                        if let Err(e) = device.stream_h264_reader(&mut stdout, &stop_clone, fps) {
                            if lianli_transport::usb::shutting_down() {
                                debug!("LCD[{index}] h264 live stream ended by shutdown: {e:#}");
                            } else {
                                warn!("LCD[{index}] h264 live stream error: {e}");
                            }
                        }
                    }
                    LcdThreadMsg::SwitchDesktop(reply) => {
                        let result = device.switch_to_desktop_mode();
                        let _ = reply.send(result);
                        break;
                    }
                    LcdThreadMsg::SetBrightness(val) => {
                        if let Err(e) = device.set_brightness_val(val) {
                            warn!("LCD[{index}] set_brightness error: {e}");
                        }
                    }
                    LcdThreadMsg::Stop => break,
                }
            }
            device.transport_release();
        });
        Self {
            tx,
            h264_stop,
            thread: Some(thread),
        }
    }

    fn stream_h264(&self, path: PathBuf, looping: bool, fps: f32) -> anyhow::Result<()> {
        self.h264_stop.store(true, Ordering::Relaxed);
        self.tx
            .send(LcdThreadMsg::StreamH264 { path, looping, fps })
            .map_err(|_| anyhow::anyhow!("LCD sender thread exited"))?;
        Ok(())
    }

    fn stream_h264_reader(
        &self,
        stdout: std::process::ChildStdout,
        fps: f32,
    ) -> anyhow::Result<()> {
        self.h264_stop.store(true, Ordering::Relaxed);
        self.tx
            .send(LcdThreadMsg::StreamH264Reader(stdout, fps))
            .map_err(|_| anyhow::anyhow!("LCD sender thread exited"))?;
        Ok(())
    }

    fn set_brightness(&self, brightness: u8) -> anyhow::Result<()> {
        match self.tx.try_send(LcdThreadMsg::SetBrightness(brightness)) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                warn!("LCD sender busy, brightness command dropped");
                Ok(())
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("LCD sender thread exited")
            }
        }
    }

    fn send_frame(&self, frame: &[u8]) -> anyhow::Result<()> {
        self.h264_stop.store(true, Ordering::Relaxed);
        match self.tx.try_send(LcdThreadMsg::Frame(frame.to_vec())) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                debug!("LCD sender busy, dropping frame");
                Ok(())
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("LCD sender thread exited")
            }
        }
    }

    pub(super) fn switch_to_desktop_mode(&mut self) -> anyhow::Result<()> {
        self.h264_stop.store(true, Ordering::Relaxed);
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(LcdThreadMsg::SwitchDesktop(reply_tx))
            .map_err(|_| anyhow::anyhow!("LCD sender thread exited"))?;
        let result = reply_rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| anyhow::anyhow!("LCD sender thread timeout"))?;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        result
    }

    fn send_frame_verified(&self, frame: &[u8]) -> anyhow::Result<()> {
        self.h264_stop.store(true, Ordering::Relaxed);
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(LcdThreadMsg::FrameVerified(frame.to_vec(), reply_tx))
            .map_err(|_| anyhow::anyhow!("LCD sender thread exited"))?;
        reply_rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| anyhow::anyhow!("LCD sender thread timeout"))?
    }

    fn stop(&mut self) {
        self.h264_stop.store(true, Ordering::Relaxed);
        let _ = self.tx.send(LcdThreadMsg::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for ThreadedWinUsbSender {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) struct ActiveTarget {
    pub(super) index: usize,
    pub(super) key: ConfigKey,
    pub(super) device_identity: String,
    // `media` must drop before `lcd`: tearing down a live h264 pipeline closes
    // the encoder's stdin, ffmpeg flushes its trailer to stdout, and the WinUsb
    // thread (owned by `lcd`) needs to still be alive to drain it.
    media: Box<dyn FrameSource>,
    pub(super) lcd: LcdBackend,
    pub(super) asset: Arc<MediaAsset>,
    pub(super) screen: ScreenInfo,
    pub(super) custom_h264: bool,
    // This variable contains the last seen frame version. Each renderer holds a frame version counter which gets increased each time it actually writes into the frame. The first time it writes into the frame sets the frame version to 1
    // By using this mechanism we are able to detect whether we actually need to send the frame via USB bus to the LCD, and thus we can save quite a lot of time by not sending frames which are already displayed.
    pub(super) frame_counter: u64,
    pub(super) consecutive_errors: u32,
    recovery_stop: Arc<AtomicBool>,
    recovery_thread: Option<JoinHandle<()>>,
    /// Set once the init worker reports LcdInitComplete for this device.
    /// Before it, a false answer from supports_c_command only means the
    /// firmware is not known yet and must be retried later.
    init_complete: bool,
    /// Set when the device definitively does not support recovery, so the
    /// periodic retry stops probing it.
    recovery_unsupported: bool,
    /// Brightness that could not be applied because the init worker held
    /// the LCD. Applied when init completes.
    pending_brightness: Option<u8>,
}

fn spawn_recovery_thread(
    lcd: SharedHidLcd,
    stop: Arc<AtomicBool>,
    index: usize,
    device_id: String,
    tx: Option<Sender<DaemonEvent>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        use lianli_devices::traits::RecoveryAction;
        while !stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(2));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let Some(mut guard) = lcd.try_lock_for(Duration::from_secs(2)) else {
                continue;
            };
            match guard.check_and_recover_lcd(&stop) {
                Ok(RecoveryAction::Recovered) => {
                    if let Some(tx) = &tx {
                        if !stop.load(Ordering::Relaxed) {
                            tx.send(DaemonEvent::RecreateMedia {
                                target_index: index,
                                device_id: device_id.clone(),
                            })
                            .ok();
                        }
                    }
                }
                Ok(RecoveryAction::NoChange) => {}
                Err(e) => {
                    debug!("LCD[{index}] health check error: {e:#}");
                }
            }
        }
    })
}

impl ActiveTarget {
    pub(super) fn new(
        index: usize,
        key: ConfigKey,
        device_identity: String,
        lcd: LcdBackend,
        asset: Arc<MediaAsset>,
        screen: ScreenInfo,
        custom_h264: bool,
        tx: Option<Sender<DaemonEvent>>,
    ) -> Self {
        let media = make_frame_source(Arc::clone(&asset), tx.clone(), &lcd, &screen, custom_h264);
        let recovery_stop = Arc::new(AtomicBool::new(false));
        let recovery_thread = match &lcd {
            LcdBackend::HidLcd(d) => {
                // bounded: the init worker holds this mutex across the 10s
                // settle on some paths
                let supports = d
                    .try_lock_for(Duration::from_millis(200))
                    .is_some_and(|guard| guard.supports_c_command());
                if supports {
                    Some(spawn_recovery_thread(
                        Arc::clone(d),
                        Arc::clone(&recovery_stop),
                        index,
                        device_identity.clone(),
                        tx.clone(),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        };
        Self {
            index,
            key,
            device_identity,
            lcd,
            media,
            asset,
            screen,
            custom_h264,
            frame_counter: 0,
            consecutive_errors: 0,
            recovery_stop,
            recovery_thread,
            init_complete: false,
            recovery_unsupported: false,
            pending_brightness: None,
        }
    }

    /// Start the recovery thread if it is missing and the device now
    /// reports c-command support. Called from new, where firmware may
    /// already be known, after LcdInitComplete, and from the periodic
    /// device poll with a zero wait so a busy LCD is retried later
    /// instead of disabling recovery for the whole session.
    pub(super) fn maybe_start_recovery(&mut self, tx: Option<Sender<DaemonEvent>>, wait: Duration) {
        if self.recovery_thread.is_some() || self.recovery_unsupported {
            return;
        }
        let LcdBackend::HidLcd(d) = &self.lcd else {
            return;
        };
        let Some(guard) = d.try_lock_for(wait) else {
            if wait > Duration::ZERO {
                debug!(
                    "[devices] LCD[{}] busy, will retry starting recovery thread",
                    self.device_identity
                );
            }
            return;
        };
        let supports = guard.supports_c_command();
        // Only a device that actually answered its firmware query gives a
        // definitive no. When the read never succeeded the capability is
        // unknown, and a later successful read by the firmware tracker
        // must still be able to start recovery, so the retries continue.
        let firmware_known = guard.firmware_version_str().is_some();
        drop(guard);
        if !supports {
            // Before init completes this only means the firmware is not
            // known yet. After it, the answer is definitive.
            if self.init_complete && firmware_known {
                self.recovery_unsupported = true;
                debug!(
                    "[devices] LCD[{}] firmware does not support recovery, stopping retries",
                    self.device_identity
                );
            }
            return;
        }
        info!(
            "[devices] LCD[{}] starting recovery thread after init",
            self.device_identity
        );
        self.recovery_thread = Some(spawn_recovery_thread(
            Arc::clone(d),
            Arc::clone(&self.recovery_stop),
            self.index,
            self.device_identity.clone(),
            tx,
        ));
    }

    /// The init worker finished, so answers from the device are now
    /// definitive and deferred work can be applied.
    pub(super) fn mark_init_complete(&mut self) {
        self.init_complete = true;
    }

    /// Apply brightness now when the LCD is free, otherwise remember it
    /// for when init completes. The init worker holds the LCD mutex across
    /// its whole settle and firmware retry window, so an unbounded lock
    /// here would stall the main loop for that entire duration.
    pub(super) fn apply_brightness(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
        brightness: u8,
    ) {
        if let LcdBackend::HidLcd(d) = &self.lcd {
            match d.try_lock_for(Duration::from_millis(500)) {
                None => {
                    debug!(
                        "[devices] LCD[{}] initializing, brightness deferred",
                        self.index
                    );
                    self.pending_brightness = Some(brightness);
                }
                Some(guard) => {
                    if let Err(e) = guard.set_brightness(brightness) {
                        warn!(
                            "Failed to apply LCD brightness for LCD[{}]: {e:#}",
                            self.index
                        );
                    }
                }
            }
            return;
        }
        if let Err(e) = self.lcd.set_brightness(wireless, builder, brightness) {
            warn!(
                "Failed to apply LCD brightness for LCD[{}]: {e:#}",
                self.index
            );
        }
    }

    /// Apply a brightness that was deferred while the LCD was initializing.
    /// Called once init completed, so an unbounded lock is safe here.
    pub(super) fn flush_pending_brightness(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
    ) {
        let Some(brightness) = self.pending_brightness.take() else {
            return;
        };
        info!(
            "[devices] LCD[{}] applying deferred brightness {brightness}",
            self.index
        );
        if let Err(e) = self.lcd.set_brightness(wireless, builder, brightness) {
            warn!(
                "Failed to apply LCD brightness for LCD[{}]: {e:#}",
                self.index
            );
        }
    }

    pub(super) fn matches(&self, identity: &str, key: &ConfigKey) -> bool {
        self.device_identity == identity && key == &self.key
    }

    /// Replace the media asset without reopening the LCD transport.
    pub(super) fn swap_media(
        &mut self,
        asset: Arc<MediaAsset>,
        custom_h264: bool,
        tx: Option<Sender<DaemonEvent>>,
    ) {
        self.asset = Arc::clone(&asset);
        self.custom_h264 = custom_h264;
        self.media = make_frame_source(asset, tx, &self.lcd, &self.screen, custom_h264);
        self.frame_counter = 0;
        info!(
            "[devices] LCD[{}] media swapped (keeping transport)",
            self.index
        );
    }

    /// Apply a `custom_h264` toggle change without reloading media or
    /// reopening the transport. Rebuilds only the frame source so the live
    /// H.264 pipeline engages/disengages immediately on save.
    pub(super) fn update_custom_h264(
        &mut self,
        custom_h264: bool,
        tx: Option<Sender<DaemonEvent>>,
    ) {
        if self.custom_h264 == custom_h264 {
            return;
        }
        self.custom_h264 = custom_h264;
        self.media = make_frame_source(
            Arc::clone(&self.asset),
            tx,
            &self.lcd,
            &self.screen,
            custom_h264,
        );
        self.frame_counter = 0;
        info!(
            "[devices] LCD[{}] custom_h264 -> {custom_h264} (frame source rebuilt)",
            self.index
        );
    }

    pub(super) fn send_frame(
        &mut self,
        wireless: Option<&WirelessController>,
        builder: &mut PacketBuilder,
    ) -> Result<bool, SendError> {
        // H.264 / autonomous sources: kick off streaming on the first call,
        // then short-circuit (their threads push frames directly to the LCD).
        if self.media.is_autonomous() {
            self.media.start(&self.lcd).map_err(SendError::Other)?;
            return Ok(true);
        }

        let is_static = self.media.is_static();
        let frame = match self.media.next_frame() {
            Some(bytes) => bytes,
            None => return Ok(false),
        };

        let result = if is_static {
            self.lcd.send_frame_verified(wireless, builder, frame)
        } else {
            self.lcd.send_frame(wireless, builder, frame)
        };
        match result {
            Ok(()) => {}
            // The init worker holds the LCD for its whole settle window.
            // Report nothing sent so the version check retries next tick
            // instead of counting an error toward target recreation.
            Err(e) if e.downcast_ref::<LcdBusy>().is_some() => {
                debug!(
                    "[devices] LCD[{}] initializing, deferring frame send",
                    self.index
                );
                return Ok(false);
            }
            Err(err) => {
                let send_err = match err.downcast::<lianli_transport::TransportError>() {
                    Ok(usb) => SendError::Usb(usb),
                    Err(other) => SendError::Other(other),
                };
                return Err(send_err);
            }
        }

        self.frame_counter += 1;
        Ok(true)
    }

    pub(super) fn stop(&mut self) {
        self.recovery_stop.store(true, Ordering::Relaxed);
        // dropping media sets the renderer stop flags, which unblock stop()
        self.media = Box::new(NoopFrameSource);
        if let Some(t) = self.recovery_thread.take() {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !t.is_finished() && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(50));
            }
            if !t.is_finished() {
                warn!(
                    "LCD[{}] recovery thread did not stop in 5s — detaching it",
                    self.index
                );
            }
        }
    }
}

impl Drop for ActiveTarget {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A source of JPEG frames to push to an LCD, or an autonomous H.264
/// pipeline that streams directly to the device.
///
/// Every `MediaRuntime` variant is now one of these. The trait lets
/// `ActiveTarget::send_frame` dispatch without a 7-arm match.
trait FrameSource: Send {
    /// Called on the first `send_frame` after the source is attached. For
    /// H.264 file streaming, this kicks off the streaming thread.
    fn start(&mut self, _lcd: &LcdBackend) -> anyhow::Result<()> {
        Ok(())
    }

    /// Poll for the next JPEG frame. Returns `None` when no new frame has
    /// been rendered since the last call, or when the source is autonomous.
    fn next_frame(&mut self) -> Option<&[u8]> {
        None
    }

    /// `true` if the source produces a single unchanging frame (uses the
    /// verified-send path that tolerates a dropped USB write).
    fn is_static(&self) -> bool {
        false
    }

    /// `true` if the source pushes frames on its own thread and `send_frame`
    /// should skip the JPEG path entirely.
    fn is_autonomous(&self) -> bool {
        false
    }
}

struct NoopFrameSource;
impl FrameSource for NoopFrameSource {}

// ─── JPEG sources ──────────────────────────────────────────────────────

struct StaticSource {
    frame: Arc<Vec<u8>>,
}
impl FrameSource for StaticSource {
    fn next_frame(&mut self) -> Option<&[u8]> {
        Some(self.frame.as_slice())
    }
    fn is_static(&self) -> bool {
        true
    }
}

struct VideoSource {
    player: Arc<AsyncVideoPlayer>,
    frames: Arc<Vec<Vec<u8>>>,
    sent_index: usize,
}
impl FrameSource for VideoSource {
    fn next_frame(&mut self) -> Option<&[u8]> {
        let idx = self.player.get_frame_index();
        if idx <= self.sent_index || self.frames.is_empty() {
            return None;
        }
        let ret = Some(self.frames[idx % self.frames.len()].as_slice());
        self.sent_index = idx;
        ret
    }
}

struct SensorSource {
    renderer: Arc<AsyncSensorRenderer>,
    cached: Vec<u8>,
    sent_index: usize,
}
impl FrameSource for SensorSource {
    fn next_frame(&mut self) -> Option<&[u8]> {
        let idx = self.renderer.get_frame_index();
        if idx <= self.sent_index {
            return None;
        }
        self.cached = self.renderer.get_current_frame();
        self.sent_index = idx;
        Some(self.cached.as_slice())
    }
}

struct CustomSource {
    renderer: Arc<AsyncCustomRenderer>,
    cached: Vec<u8>,
    sent_index: usize,
}
impl FrameSource for CustomSource {
    fn next_frame(&mut self) -> Option<&[u8]> {
        let idx = self.renderer.get_frame_index();
        if idx <= self.sent_index {
            return None;
        }
        self.cached = self.renderer.get_current_frame();
        self.sent_index = idx;
        Some(self.cached.as_slice())
    }
}

// ─── H.264 autonomous sources ──────────────────────────────────────────

struct H264FileSource {
    path: PathBuf,
    looping: bool,
    fps: f32,
    started: bool,
    hid_thread: Option<JoinHandle<()>>,
    hid_stop: Arc<AtomicBool>,
    /// start() runs every streaming tick; back off after an open failure
    /// so a missing file retries periodically instead of once or per-tick.
    retry_after: Option<std::time::Instant>,
}

const FILE_OPEN_RETRY: Duration = Duration::from_secs(5);

impl H264FileSource {
    fn new(path: PathBuf, looping: bool, fps: f32) -> Self {
        Self {
            path,
            looping,
            fps,
            started: false,
            hid_thread: None,
            hid_stop: Arc::new(AtomicBool::new(false)),
            retry_after: None,
        }
    }
}

impl FrameSource for H264FileSource {
    fn start(&mut self, lcd: &LcdBackend) -> anyhow::Result<()> {
        if self.started {
            return Ok(());
        }
        if let Some(t) = self.retry_after {
            if std::time::Instant::now() < t {
                return Ok(());
            }
        }
        match lcd {
            LcdBackend::WinUsb(sender) => {
                sender.stream_h264(self.path.clone(), self.looping, self.fps)?;
            }
            LcdBackend::HidLcd(hid) => {
                // open before marking started so a missing file can retry
                let file = match std::fs::File::open(&self.path) {
                    Ok(f) => f,
                    Err(e) => {
                        warn!("HID h264 file open failed for {:?}: {e:#}", self.path);
                        self.retry_after = Some(std::time::Instant::now() + FILE_OPEN_RETRY);
                        return Ok(());
                    }
                };
                self.retry_after = None;
                let lcd = Arc::clone(hid);
                let (looping, fps) = (self.looping, self.fps);
                let stop = Arc::clone(&self.hid_stop);
                self.hid_thread = Some(thread::spawn(move || {
                    stream_h264_file_to_hid(lcd, file, looping, fps, stop);
                }));
            }
            _ => {}
        }
        self.started = true;
        Ok(())
    }
    fn is_autonomous(&self) -> bool {
        true
    }
}

impl Drop for H264FileSource {
    fn drop(&mut self) {
        self.hid_stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.hid_thread.take() {
            let _ = t.join();
        }
    }
}

struct CustomH264Source {
    #[allow(dead_code)]
    renderer: Arc<AsyncCustomH264Renderer>,
}
impl FrameSource for CustomH264Source {
    fn is_autonomous(&self) -> bool {
        true
    }
}

struct SensorH264Source {
    #[allow(dead_code)]
    renderer: Arc<AsyncSensorH264Renderer>,
}
impl FrameSource for SensorH264Source {
    fn is_autonomous(&self) -> bool {
        true
    }
}

/// Construct the appropriate `FrameSource` for a given media asset + LCD combo.
fn make_frame_source(
    asset: Arc<MediaAsset>,
    tx: Option<Sender<DaemonEvent>>,
    lcd: &LcdBackend,
    screen: &ScreenInfo,
    custom_h264: bool,
) -> Box<dyn FrameSource> {
    match &asset.kind {
        MediaAssetKind::Static { frame } => Box::new(StaticSource {
            frame: Arc::clone(frame),
        }),
        MediaAssetKind::Video { frames, .. } => {
            let player = Arc::new(AsyncVideoPlayer::new(tx, Arc::clone(&asset)));
            Box::new(VideoSource {
                player,
                frames: Arc::clone(frames),
                sent_index: 0,
            })
        }
        MediaAssetKind::Sensor {
            asset: sensor_asset,
        } => {
            if screen.h264 {
                match AsyncSensorH264Renderer::new(
                    Arc::clone(sensor_asset),
                    lcd,
                    screen,
                    asset.stream_fps,
                ) {
                    Ok(renderer) => {
                        info!("Sensor mode using live h264 pipeline");
                        return Box::new(SensorH264Source {
                            renderer: Arc::new(renderer),
                        });
                    }
                    Err(e) => {
                        warn!("Sensor h264 pipeline unavailable, falling back to JPEG: {e}");
                    }
                }
            }
            let renderer = Arc::new(AsyncSensorRenderer::new(
                tx,
                Arc::clone(sensor_asset),
                Arc::clone(&asset),
                screen.needs_keepalive,
            ));
            let cached = renderer.get_current_frame();
            Box::new(SensorSource {
                renderer,
                cached,
                sent_index: 0,
            })
        }
        MediaAssetKind::H264Stream {
            path, looping, fps, ..
        } => Box::new(H264FileSource::new(path.clone(), *looping, *fps)),
        MediaAssetKind::Custom {
            asset: custom_asset,
        } => {
            if custom_h264 && screen.h264 {
                match AsyncCustomH264Renderer::new(
                    Arc::clone(custom_asset),
                    lcd,
                    screen,
                    custom_asset.canvas_width(),
                    custom_asset.canvas_height(),
                    custom_asset.total_rotation_deg(),
                    asset.stream_fps,
                ) {
                    Ok(renderer) => {
                        info!("Custom mode using live h264 pipeline");
                        return Box::new(CustomH264Source {
                            renderer: Arc::new(renderer),
                        });
                    }
                    Err(e) => {
                        warn!("Custom h264 pipeline unavailable, falling back to JPEG: {e}");
                    }
                }
            }
            let renderer = Arc::new(AsyncCustomRenderer::new(
                tx,
                Arc::clone(custom_asset),
                Arc::clone(&asset),
                screen.needs_keepalive,
            ));
            let cached = renderer.get_current_frame();
            Box::new(CustomSource {
                renderer,
                cached,
                sent_index: 0,
            })
        }
    }
}

fn stream_h264_file_to_hid(
    lcd: SharedHidLcd,
    mut file: std::fs::File,
    looping: bool,
    fps: f32,
    stop: Arc<AtomicBool>,
) {
    use lianli_devices::hydroshift_lcd::{find_au_split, pace_frame};
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    let frame_interval = {
        let mut guard = lcd.lock();
        Duration::from_secs_f32(1.0 / guard.set_stream_fps(fps))
    };
    let mut read_buf = vec![0u8; 64 * 1024];
    let mut next_deadline = Instant::now() + frame_interval;
    // a single-AU file never yields a split boundary, its only frame rides
    // the EOF flush, allow it exactly one looping pass
    let mut first_pass = true;
    let mut saw_boundary = false;
    'outer: loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut accum: Vec<u8> = Vec::with_capacity(256 * 1024);
        let mut sent_any = false;
        loop {
            if stop.load(Ordering::Relaxed) {
                break 'outer;
            }
            let n = match file.read(&mut read_buf) {
                Ok(n) => n,
                Err(e) => {
                    warn!("HID h264 file read error: {e:#}");
                    break 'outer;
                }
            };
            if n == 0 {
                break;
            }
            accum.extend_from_slice(&read_buf[..n]);
            if accum.len() > MAX_AU_BYTES {
                warn!(
                    "HID h264 file: no AU boundary within {} bytes, aborting",
                    MAX_AU_BYTES
                );
                break 'outer;
            }
            while let Some(split) = find_au_split(&accum) {
                if stop.load(Ordering::Relaxed) {
                    break 'outer;
                }
                let au: Vec<u8> = accum.drain(..split).collect();
                if au.is_empty() {
                    continue;
                }
                let result = {
                    let mut guard = lcd.lock();
                    guard.send_h264_frame(&au)
                };
                if let Err(e) = result {
                    warn!("HID h264 stream send error: {e:#}");
                    break 'outer;
                }
                saw_boundary = true;
                sent_any = true;
                pace_frame(&mut next_deadline, frame_interval);
            }
        }
        // reached only via the EOF break (every other exit is break 'outer),
        // residual flush is paced like a regular AU
        if !stop.load(Ordering::Relaxed) && !accum.is_empty() {
            pace_frame(&mut next_deadline, frame_interval);
            match lcd.lock().send_h264_frame(&accum) {
                Ok(()) => sent_any = true,
                Err(e) => debug!("HID h264 flush: {e:#}"),
            }
        }
        if !looping || stop.load(Ordering::Relaxed) {
            break;
        }
        // only loop when real AU boundaries were found: a boundary-less
        // file's flush would otherwise re-send identical data forever
        if !sent_any || (first_pass && !saw_boundary) {
            warn!("HID h264 file produced no complete access units, stopping");
            break;
        }
        first_pass = false;
        if let Err(e) = file.seek(SeekFrom::Start(0)) {
            warn!("HID h264 file seek failed: {e:#}");
            break;
        }
    }
}

pub(super) enum SendError {
    Usb(lianli_transport::TransportError),
    Other(anyhow::Error),
}
