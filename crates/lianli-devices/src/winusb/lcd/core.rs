use crate::crypto::PacketBuilder;
use anyhow::{bail, Context, Result};
use lianli_shared::screen::ScreenInfo;
use lianli_transport::usb::{RusbBulk, EP_IN, EP_OUT};
use parking_lot::{Mutex, MutexGuard};
use rusb::{Device, GlobalContext};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// A control command (SyncPumpFan, PushRgbData, …) that the H2 AIO channel
/// handed to the LCD stream thread because the panel was busy ingesting
/// H.264. Sent verbatim at the next safe point; the reply is read and
/// discarded after `reply_wait`.
pub struct PendingCmd {
    pub label: &'static str,
    pub packet: Vec<u8>,
    pub reply_wait: Duration,
    pub queued_at: Instant,
    /// May this go out between chunks once the panel reports headroom? False
    /// for commands that hang the panel in play mode regardless of buffer
    /// level (PushRgbData — tested at levels 4 and 1, 2026-08-23); those wait
    /// for the stream to end.
    pub play_safe: bool,
}

/// USB bulk handle shared by the LCD stream and the HydroShift II control
/// channel (pump/fan/ring RGB), plus the coordination that keeps control
/// commands off the wire while the panel's ingest buffer is full.
///
/// Field evidence (usbmon, 2026-08-22/23): a SyncPumpFan or PushRgbData write
/// landing while the panel reports buffer level 3–4 mid-stream hangs the MCU
/// (bulk IN goes silent; sometimes EP0 dies too and only a power cycle helps).
/// So while `streaming` is set, control writers queue their packet here and the
/// stream thread — the only writer — flushes the queue once the panel reports
/// headroom.
pub struct LcdLink {
    bulk: Mutex<RusbBulk>,
    streaming: AtomicBool,
    pending: Mutex<Vec<PendingCmd>>,
}

impl LcdLink {
    pub fn new(bulk: RusbBulk) -> Self {
        Self {
            bulk: Mutex::new(bulk),
            streaming: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, RusbBulk> {
        self.bulk.lock()
    }

    /// True while an H.264 stream is feeding the panel.
    pub fn is_streaming(&self) -> bool {
        self.streaming.load(Ordering::Acquire)
    }

    pub(crate) fn set_streaming(&self, on: bool) {
        self.streaming.store(on, Ordering::Release);
    }

    /// Queue a control command for the stream thread. Latest wins per label:
    /// an older SyncPumpFan still waiting is replaced, not appended.
    pub fn defer(&self, cmd: PendingCmd) {
        let mut q = self.pending.lock();
        q.retain(|c| c.label != cmd.label);
        q.push(cmd);
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.lock().is_empty()
    }

    /// Take every queued command (stream over).
    pub(crate) fn take_pending(&self) -> Vec<PendingCmd> {
        std::mem::take(&mut *self.pending.lock())
    }

    /// Take only the commands that may be sent mid-stream.
    pub(crate) fn take_play_safe(&self) -> Vec<PendingCmd> {
        let mut q = self.pending.lock();
        let (safe, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut *q)
            .into_iter()
            .partition(|c| c.play_safe);
        *q = rest;
        safe
    }

    pub(crate) fn has_play_safe_pending(&self) -> bool {
        self.pending.lock().iter().any(|c| c.play_safe)
    }

    pub(crate) fn oldest_play_safe_age(&self) -> Option<Duration> {
        self.pending
            .lock()
            .iter()
            .filter(|c| c.play_safe)
            .map(|c| c.queued_at.elapsed())
            .max()
    }
}

pub type SharedTransport = Arc<LcdLink>;

/// Buffer level at or below which queued control commands go out right away.
const CONTROL_SAFE_LEVEL: u8 = 1;
/// If the panel never drains that far, accept this level once a command has
/// waited `CONTROL_RELAX_AFTER`.
const CONTROL_RELAXED_LEVEL: u8 = 2;
const CONTROL_RELAX_AFTER: Duration = Duration::from_secs(3);

const REOPEN_DELAY: Duration = Duration::from_millis(100);
const WAIT_BUFFER_POLL: Duration = Duration::from_millis(50);
const WAIT_BUFFER_NO_STOP_CAP: u32 = 600;

pub(crate) struct WinUsbLcdCore {
    transport: SharedTransport,
    builder: PacketBuilder,
    screen: ScreenInfo,
    write_timeout: Duration,
    read_timeout: Duration,
    name: String,
    raw_device: Option<Device<GlobalContext>>,
    pub(crate) initialized: bool,
    pub(crate) consecutive_failures: u32,
    pub(crate) h264_chunk_size: usize,
    pub(crate) device_gone: bool,
    pub(crate) firmware: Option<String>,
}

/// Read the serial the kernel cached at enumeration, matching on bus/device
/// number. Preferred over a live EP0 read: it cannot stall.
fn sysfs_serial(bus: u8, address: u8) -> Option<String> {
    let entries = std::fs::read_dir("/sys/bus/usb/devices").ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let rd = |f: &str| std::fs::read_to_string(path.join(f)).ok();
        let (Some(b), Some(d)) = (rd("busnum"), rd("devnum")) else {
            continue;
        };
        if b.trim().parse::<u8>().ok() != Some(bus) || d.trim().parse::<u8>().ok() != Some(address)
        {
            continue;
        }
        let serial = rd("serial")?;
        let serial = serial.trim();
        if !serial.is_empty() {
            tracing::debug!("using kernel-cached serial {serial}");
            return Some(serial.to_string());
        }
    }
    None
}

impl WinUsbLcdCore {
    pub(crate) fn open(
        device: Device<GlobalContext>,
        screen: ScreenInfo,
        name: &str,
        write_timeout: Duration,
        read_timeout: Duration,
    ) -> Result<Self> {
        let bus = device.bus_number();
        let address = device.address();
        let desc = device
            .device_descriptor()
            .context("reading device descriptor")?;
        // FIX: some units stop answering EP0 string-descriptor requests while
        // their bulk pipe keeps working, so the live read fails and the device
        // silently gets a positional id. That breaks config lookup (which keys
        // on the serial) and makes the daemon re-open an already-claimed
        // interface. The kernel cached the serial at enumeration time, so fall
        // back to sysfs before giving up on identity.
        // Ask the kernel first. It read the serial at enumeration, so this is a
        // cheap file read that always works. Going to the device instead costs
        // ~5s of EP0 timeouts on units that stop answering string-descriptor
        // requests, and that delay widens the window in which a second open
        // thread races this one and hits EBUSY on interface 0.
        let serial = sysfs_serial(bus, address)
            .or_else(|| {
                device
                    .open()
                    .and_then(|h| h.read_serial_number_string_ascii(&desc))
                    .ok()
            })
            .unwrap_or_else(|| format!("bus{bus}-addr{address}"));

        let mut transport = RusbBulk::open_device(device.clone()).context("opening WinUSB LCD")?;
        transport
            .detach_and_configure(name)
            .context("configuring WinUSB LCD")?;

        info!(
            "{name} opened: {}x{} at bus {bus} addr {address} serial {serial}",
            screen.width, screen.height
        );

        Ok(Self {
            transport: Arc::new(LcdLink::new(transport)),
            builder: PacketBuilder::new(),
            screen,
            write_timeout,
            read_timeout,
            name: name.to_string(),
            raw_device: Some(device),
            initialized: false,
            consecutive_failures: 0,
            h264_chunk_size: 202_752,
            device_gone: false,
            firmware: None,
        })
    }

    pub(crate) fn from_shared(
        transport: SharedTransport,
        screen: ScreenInfo,
        name: String,
        write_timeout: Duration,
        read_timeout: Duration,
    ) -> Self {
        Self {
            transport,
            builder: PacketBuilder::new(),
            screen,
            write_timeout,
            read_timeout,
            name,
            raw_device: None,
            initialized: false,
            consecutive_failures: 0,
            h264_chunk_size: 202_752,
            device_gone: false,
            firmware: None,
        }
    }

    pub(crate) fn screen(&self) -> &ScreenInfo {
        &self.screen
    }

    pub(crate) fn builder_mut(&mut self) -> &mut PacketBuilder {
        &mut self.builder
    }

    pub(crate) fn shared_transport(&self) -> SharedTransport {
        Arc::clone(&self.transport)
    }

    pub(crate) fn firmware_str(&self) -> Option<&str> {
        self.firmware.as_deref()
    }

    pub(crate) fn transport_release(&self) {}

    #[inline]
    fn tx_write_full(
        &self,
        data: &[u8],
    ) -> std::result::Result<(), lianli_transport::TransportError> {
        self.transport.lock().write_full(data, self.write_timeout)
    }

    #[inline]
    fn tx_read(
        &self,
        buf: &mut [u8],
    ) -> std::result::Result<usize, lianli_transport::TransportError> {
        self.transport.lock().read(buf, self.read_timeout)
    }

    #[inline]
    fn tx_read_flush(&self) {
        self.transport.lock().read_flush();
    }

    #[inline]
    fn tx_clear_halt(&self, ep: u8) -> std::result::Result<(), lianli_transport::TransportError> {
        self.transport.lock().clear_halt(ep)
    }

    fn note_write_success(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Vendor-faithful recovery: close the handle and reopen it from the raw
    /// device (ReInitDev), so a stalled endpoint is recovered within the
    /// session without a USB port reset (which would take down composite
    /// siblings like the LED MCU). Falls back to clear_halt when no raw device
    /// is available (shared-transport path).
    fn try_recover(&mut self) -> Result<()> {
        if lianli_transport::usb::shutting_down() {
            bail!("shutting down; skipping recovery");
        }
        if self.device_gone {
            bail!("device handle is stale; re-discovery required");
        }
        self.consecutive_failures += 1;

        if let Some(raw) = &self.raw_device {
            // FIX: release the interfaces the current handle holds *before*
            // reopening. The replacement only lands in self.transport once the
            // new open succeeds, so without this the old handle still owns
            // interface 0 and every claim_interface(0) returns EBUSY — the
            // recovery path could never succeed, it just retried 20 times
            // against a device already in an error state.
            self.transport.lock().release();
            std::thread::sleep(REOPEN_DELAY);
            match RusbBulk::open_device(raw.clone()) {
                Ok(mut t) => {
                    if t.detach_and_configure(&self.name).is_ok() {
                        *self.transport.lock() = t;
                        self.consecutive_failures = 0;
                        debug!("recovered via close+reopen");
                        return Ok(());
                    }
                }
                Err(e) => warn!("reopen failed: {e}"),
            }
        }

        let out_ok = self.tx_clear_halt(EP_OUT).is_ok();
        let _ = self.tx_clear_halt(EP_IN);
        if out_ok && self.consecutive_failures <= 5 {
            debug!("recovered EP_OUT stall via clear_halt");
            return Ok(());
        }

        self.device_gone = true;
        bail!("device unresponsive after recovery attempts; re-discovery required")
    }

    fn read_response(&mut self, context: &str) -> Option<[u8; 512]> {
        let mut buf = [0u8; 512];
        match self.tx_read(&mut buf) {
            Ok(n) if n > 0 => {
                debug!(
                    "Response for {context} ({n} bytes): {:02x?}",
                    &buf[..n.min(32)]
                );
                self.tx_read_flush();
                return Some(buf);
            }
            Ok(_) => debug!("No response for {context} (timeout)"),
            Err(e) => warn!("Read after {context} failed: {e}"),
        }
        self.tx_read_flush();
        None
    }

    pub(crate) fn send_command(&mut self, header: Vec<u8>, label: &str) {
        match self.tx_write_full(&header) {
            Ok(_) => self.note_write_success(),
            Err(e) => {
                warn!("{label} write failed: {e}");
                if let Err(rec_err) = self.try_recover() {
                    warn!("{label} recovery skipped: {rec_err}");
                    return;
                }
                if let Err(e2) = self.tx_write_full(&header) {
                    warn!("{label} write retry failed: {e2}");
                    return;
                }
                self.note_write_success();
            }
        }
        self.read_response(label);
    }

    pub(crate) fn read_firmware(&mut self) {
        let ver = self.builder.get_ver_header_winusb();
        match self.tx_write_full(&ver) {
            Ok(_) => self.note_write_success(),
            Err(e) => warn!("GetVer write failed: {e}"),
        }
        if let Some(resp) = self.read_response("GetVer") {
            let fw_bytes = &resp[8..40.min(resp.len())];
            let end = fw_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(fw_bytes.len());
            let fw_str = String::from_utf8_lossy(&fw_bytes[..end]).to_string();
            if !fw_str.is_empty() {
                info!("LCD firmware: {fw_str}");
                self.firmware = Some(fw_str);
            }
        }
    }

    pub(crate) fn query_h264_block(&mut self) {
        let h264_block = self.builder.get_h264_block_header_winusb();
        if self.tx_write_full(&h264_block).is_ok() {
            if let Some(resp) = self.read_response("GetH264Block") {
                if resp.len() >= 12 {
                    let size = u32::from_be_bytes([resp[8], resp[9], resp[10], resp[11]]) as usize;
                    if size > 0 {
                        self.h264_chunk_size = size;
                        debug!("H264 chunk size from device: {size}");
                    }
                }
            }
        }
    }

    pub(crate) fn clear_png_cmd(&mut self) {
        let h = self.builder.clear_png_header_winusb();
        self.send_command(h, "ClearPng");
    }

    pub(crate) fn stop_clock_resp(&mut self) -> Option<[u8; 512]> {
        let h = self.builder.stop_clock_header_winusb();
        match self.tx_write_full(&h) {
            Ok(_) => self.note_write_success(),
            Err(e) => {
                warn!("StopClock write failed: {e}");
                return None;
            }
        }
        self.read_response("StopClock")
    }

    pub(crate) fn clear_jpg_layer(&mut self) {
        use image::{ImageBuffer, Rgb};
        let jpg_img =
            ImageBuffer::from_pixel(self.screen.width, self.screen.height, Rgb([0u8, 0, 0]));
        let mut jpg_buf = Vec::new();
        {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut jpg_buf,
                self.screen.jpeg_quality,
            );
            if let Err(e) = encoder.encode_image(&jpg_img) {
                warn!("Failed to encode blank JPEG: {e}");
                return;
            }
        }
        let header = self.builder.jpeg_header_winusb(jpg_buf.len());
        let mut packet = vec![0u8; 512 + jpg_buf.len()];
        packet[..512].copy_from_slice(&header);
        packet[512..].copy_from_slice(&jpg_buf);
        if let Err(e) = self.tx_write_full(&packet) {
            warn!("ClearJpgLayer failed: {e}");
        } else {
            self.read_response("ClearJpgLayer");
        }
    }

    pub(crate) fn clear_layers(&mut self) {
        use image::{ImageBuffer, Rgb, Rgba};
        use std::io::Cursor;

        let w = self.screen.width;
        let h = self.screen.height;

        let png_img = ImageBuffer::from_pixel(w, h, Rgba([0u8, 0, 0, 0]));
        let mut png_buf = Vec::new();
        if png_img
            .write_to(&mut Cursor::new(&mut png_buf), image::ImageFormat::Png)
            .is_ok()
        {
            let header = self.builder.png_header_winusb(png_buf.len());
            let mut packet = vec![0u8; 512 + png_buf.len()];
            packet[..512].copy_from_slice(&header);
            packet[512..].copy_from_slice(&png_buf);
            if let Err(e) = self.tx_write_full(&packet) {
                warn!("ClearPngLayer failed: {e}");
            } else {
                self.read_response("ClearPngLayer");
            }
        }

        let jpg_img = ImageBuffer::from_pixel(w, h, Rgb([0u8, 0, 0]));
        let mut jpg_buf = Vec::new();
        {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut jpg_buf,
                self.screen.jpeg_quality,
            );
            if let Err(e) = encoder.encode_image(&jpg_img) {
                warn!("Failed to encode blank JPEG: {e}");
                return;
            }
        }
        let header = self.builder.jpeg_header_winusb(jpg_buf.len());
        let mut packet = vec![0u8; 512 + jpg_buf.len()];
        packet[..512].copy_from_slice(&header);
        packet[512..].copy_from_slice(&jpg_buf);
        if let Err(e) = self.tx_write_full(&packet) {
            warn!("ClearJpgLayer failed: {e}");
        } else {
            self.read_response("ClearJpgLayer");
        }
    }

    pub(crate) fn send_frame(&mut self, frame: &[u8]) -> Result<()> {
        if frame.len() > self.screen.max_payload {
            bail!(
                "frame payload {} exceeds LCD limit {}",
                frame.len(),
                self.screen.max_payload
            );
        }

        let header = if self.screen.png {
            self.builder.png_header_winusb(frame.len())
        } else {
            self.builder.jpeg_header_winusb(frame.len())
        };
        let total = 512 + frame.len();
        let mut packet = vec![0u8; total];
        packet[..512].copy_from_slice(&header);
        packet[512..total].copy_from_slice(frame);

        match self.tx_write_full(&packet) {
            Ok(_) => self.note_write_success(),
            Err(e) => {
                warn!("Frame write failed: {e}");
                self.try_recover()
                    .with_context(|| format!("recovering from frame write error: {e}"))?;
                self.tx_write_full(&packet)
                    .context("writing LCD frame after recovery")?;
                self.note_write_success();
            }
        }

        let resp = self.read_response("frame ack");
        if let Some(buf) = resp {
            if buf[8] > 3 {
                self.wait_buffer(2, None);
            }
        }
        Ok(())
    }

    pub(crate) fn send_frame_verified(&mut self, frame: &[u8]) -> Result<()> {
        for attempt in 0..3u32 {
            match self.send_frame(frame) {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 2 => {
                    warn!(
                        "Frame send failed (attempt {}): {e}, reinitializing",
                        attempt + 1
                    );
                    self.initialized = false;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    pub(crate) fn set_brightness_val(&mut self, brightness: u8) -> Result<()> {
        let header = self.builder.brightness_header_winusb(brightness);
        self.tx_write_full(&header).context("setting brightness")?;
        self.read_response("brightness");
        debug!("Set brightness to {}", brightness.min(100));
        Ok(())
    }

    pub(crate) fn set_frame_rate(&mut self, fps: u8) -> Result<()> {
        let header = self.builder.frame_rate_header_winusb(fps);
        self.tx_write_full(&header).context("setting frame rate")?;
        self.read_response("frame rate");
        debug!("Set frame rate to {fps}");
        Ok(())
    }

    pub(crate) fn apply_stream_fps(&mut self, fps: f32) -> Result<()> {
        let clamped = fps.round().clamp(1.0, self.screen.max_fps as f32) as u8;
        self.set_frame_rate(clamped)
    }

    pub(crate) fn switch_to_desktop_mode(&mut self) -> Result<()> {
        let stop = self.builder.stop_play_header_winusb();
        self.send_command(stop, "StopPlay");
        let switch_cmd = self.builder.switch_to_desktop_header_winusb();
        self.send_command(switch_cmd, "SwitchToDesktop");
        let reboot = self.builder.reboot_header_winusb();
        self.send_command(reboot, "Reboot");
        info!("Sent SwitchToDesktop + Reboot — device will reboot into desktop mode");
        self.initialized = false;
        Ok(())
    }

    fn query_buffer_level(&mut self) -> Option<u8> {
        let header = self.builder.query_buffer_level_header_winusb();
        self.tx_write_full(&header).ok()?;
        let mut buf = [0u8; 512];
        match self.tx_read(&mut buf) {
            Ok(n) if n > 0 => {
                self.tx_read_flush();
                Some(buf[8])
            }
            _ => {
                self.tx_read_flush();
                None
            }
        }
    }

    /// Vendor-faithful: poll QueryBlock every 50ms until the buffer drains to
    /// `threshold` or less. When a `stop` flag is supplied (H264 streaming) it
    /// is honoured as the cancellation token; otherwise a safety cap prevents
    /// an indefinite hang on a wedged device.
    ///
    /// Returns the last buffer level read, if any.
    pub(crate) fn wait_buffer(&mut self, threshold: u8, stop: Option<&AtomicBool>) -> Option<u8> {
        let mut iter = 0u32;
        let mut last = None;
        loop {
            if let Some(s) = stop {
                if s.load(Ordering::Relaxed) {
                    return last;
                }
            } else if iter >= WAIT_BUFFER_NO_STOP_CAP {
                debug!("Buffer wait capped after {} polls", WAIT_BUFFER_NO_STOP_CAP);
                return last;
            }
            iter += 1;
            match self.query_buffer_level() {
                Some(level) if level <= threshold => return Some(level),
                Some(level) => {
                    last = Some(level);
                    std::thread::sleep(WAIT_BUFFER_POLL)
                }
                None => {
                    debug!("Buffer wait aborted (no response)");
                    return last;
                }
            }
        }
    }

    /// Send play-safe control commands queued by the H2 AIO channel while we
    /// stream. Only called from the stream thread, which is the sole writer
    /// while `streaming` is set, so each reply here belongs to the command
    /// just sent.
    fn flush_pending_control(&mut self, level: Option<u8>) {
        if !self.transport.has_play_safe_pending() {
            return;
        }
        let safe = match level {
            Some(l) if l <= CONTROL_SAFE_LEVEL => true,
            Some(l) if l <= CONTROL_RELAXED_LEVEL => self
                .transport
                .oldest_play_safe_age()
                .is_some_and(|age| age >= CONTROL_RELAX_AFTER),
            _ => false,
        };
        if !safe {
            return;
        }
        for cmd in self.transport.take_play_safe() {
            debug!(
                "Sending deferred {} ({} bytes, waited {} ms, level {:?})",
                cmd.label,
                cmd.packet.len(),
                cmd.queued_at.elapsed().as_millis(),
                level
            );
            if let Err(e) = self.send_deferred(&cmd) {
                warn!("Deferred {} write failed: {e}", cmd.label);
            }
        }
    }

    /// Write one deferred control packet and discard its reply, holding the
    /// transport across both halves. Taking the lock twice would let another
    /// exchange in between, and the reply read here would consume an answer
    /// belonging to that command — the failure GetH2Params was made atomic for.
    fn send_deferred(
        &self,
        cmd: &PendingCmd,
    ) -> std::result::Result<(), lianli_transport::TransportError> {
        let transport = self.transport.lock();
        transport.write_full(&cmd.packet, self.write_timeout)?;
        let mut buf = [0u8; 512];
        let _ = transport.read(&mut buf, cmd.reply_wait);
        Ok(())
    }

    fn send_h264_chunk(
        &mut self,
        data: &[u8],
        is_last: bool,
        play_count: u8,
        play_tick: u32,
        stop: &AtomicBool,
    ) -> Result<()> {
        let header =
            self.builder
                .start_play_header_winusb(data.len(), is_last, play_count, play_tick);
        let mut packet = vec![0u8; 512 + data.len()];
        packet[..512].copy_from_slice(&header);
        packet[512..512 + data.len()].copy_from_slice(data);

        match self.tx_write_full(&packet) {
            Ok(_) => self.note_write_success(),
            Err(e) => {
                // The write is refused on purpose once shutdown starts, so
                // unwinding quietly is the expected outcome, not a fault
                // that recovery should chase.
                if lianli_transport::usb::shutting_down() {
                    debug!("H264 chunk write refused during shutdown: {e}");
                    bail!("shutting down");
                }
                warn!("H264 chunk write failed: {e}");
                self.try_recover()
                    .with_context(|| format!("recovering from h264 write error: {e}"))?;
                self.tx_write_full(&packet)
                    .context("h264 chunk write after recovery")?;
                self.note_write_success();
            }
        }

        let resp = self.read_response("h264 chunk");
        let mut level = resp.map(|buf| buf[8]);
        if let Some(buf) = resp {
            if buf[8] > 3 {
                level = self.wait_buffer(2, Some(stop)).or(level);
            }
        }
        self.flush_pending_control(level);
        Ok(())
    }

    /// Mark the start of an H.264 stream: control writers defer to us.
    /// The flag is flipped while holding the bulk mutex, and control
    /// writers recheck it under the same mutex, so a writer that observed
    /// not streaming either finishes its write before any stream chunk or
    /// sees the flag flip and queues instead. It can never land a control
    /// packet mid stream.
    fn stream_begin(&self) {
        let _bulk = self.transport.lock();
        self.transport.set_streaming(true);
    }

    /// Mark the end of a stream. If it ended cleanly the panel is idle now, so
    /// anything still queued goes out directly (bypassing the level check);
    /// after an error the queue is dropped rather than hammering a device that
    /// just stopped answering.
    fn stream_end(&mut self, clean: bool) {
        {
            let _bulk = self.transport.lock();
            self.transport.set_streaming(false);
        }
        let pending = self.transport.take_pending();
        if !clean {
            return;
        }
        for cmd in pending {
            debug!("Sending deferred {} after stream end", cmd.label);
            if let Err(e) = self.send_deferred(&cmd) {
                warn!("Deferred {} write failed: {e}", cmd.label);
            }
        }
    }

    pub(crate) fn stream_h264(
        &mut self,
        path: &std::path::Path,
        looping: bool,
        stop: &AtomicBool,
        fps: f32,
        play_count: u8,
        play_tick: u32,
    ) -> Result<()> {
        let mut file = std::fs::File::open(path).context("opening h264 file")?;
        let mut file_buf = vec![0u8; self.h264_chunk_size];
        let interval = chunk_interval(fps);
        let mut next_deadline = Instant::now() + interval;

        self.stream_begin();
        let result = self.stream_h264_inner(
            &mut file,
            &mut file_buf,
            looping,
            stop,
            interval,
            &mut next_deadline,
            play_count,
            play_tick,
        );
        self.stream_end(result.is_ok());
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn stream_h264_inner(
        &mut self,
        file: &mut std::fs::File,
        file_buf: &mut [u8],
        looping: bool,
        stop: &AtomicBool,
        interval: Duration,
        next_deadline: &mut Instant,
        play_count: u8,
        play_tick: u32,
    ) -> Result<()> {
        use std::io::{Read, Seek};
        loop {
            let n = file.read(file_buf).context("reading h264 chunk")?;
            if n == 0 {
                if looping && !stop.load(Ordering::Relaxed) {
                    file.seek(std::io::SeekFrom::Start(0))?;
                    continue;
                }
                break;
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let is_last = {
                let pos = file.stream_position()?;
                let len = file.metadata()?.len();
                pos >= len
            };
            self.send_h264_chunk(&file_buf[..n], is_last, play_count, play_tick, stop)?;
            sleep_until(next_deadline, interval);
        }

        self.tx_read_flush();
        self.initialized = false;
        Ok(())
    }

    pub(crate) fn stream_h264_reader(
        &mut self,
        reader: &mut dyn std::io::Read,
        stop: &AtomicBool,
        play_count: u8,
        play_tick: u32,
    ) -> Result<()> {
        let mut buf = vec![0u8; self.h264_chunk_size];
        self.stream_begin();
        let result = (|| -> Result<()> {
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let n = reader
                    .read(&mut buf)
                    .context("WinUSB LCD: read h264 stream")?;
                if n == 0 {
                    break;
                }
                self.send_h264_chunk(&buf[..n], false, play_count, play_tick, stop)?;
            }
            Ok(())
        })();
        self.stream_end(result.is_ok());
        self.tx_read_flush();
        self.initialized = false;
        result
    }

    pub(crate) fn init_logging(&self) {
        info!(
            "Initializing LCD ({}x{}, quality {})",
            self.screen.width, self.screen.height, self.screen.jpeg_quality
        );
    }

    pub(crate) fn reset_failure_state(&mut self) {
        self.device_gone = false;
        self.consecutive_failures = 0;
        self.tx_read_flush();
    }
}

fn chunk_interval(fps: f32) -> Duration {
    let target = Duration::from_secs_f32(1.0 / fps.max(1.0));
    target.max(Duration::from_millis(30))
}

fn sleep_until(next_deadline: &mut Instant, interval: Duration) {
    let now = Instant::now();
    if now < *next_deadline {
        std::thread::sleep(*next_deadline - now);
    }
    *next_deadline += interval;
    let now = Instant::now();
    if *next_deadline < now {
        *next_deadline = now + interval;
    }
}
