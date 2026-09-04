use super::DetectedDevice;
use anyhow::Result;
use lianli_shared::config::HidBackend;
use lianli_shared::device_id::{DeviceFamily, UsbId, KNOWN_DEVICES};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use tracing::{debug, warn};

/// Enumerate all Lian Li USB devices on the system, sorted by (bus, address).

/// Devices whose live EP0 serial probe already failed once. The enumeration
/// runs every poll, so without this cache a device that has no kernel cached
/// serial and does not answer string descriptor requests would be probed
/// once a second, which is exactly the pattern that wedges some controllers.
/// A replugged healthy device gets its serial cached by the kernel and is
/// served from sysfs, so a stale entry here is never consulted for it.
type Ep0SerialKey = (u16, u16, u8, u8);
type Ep0SerialFailures = Mutex<HashSet<Ep0SerialKey>>;

static EP0_SERIAL_FAILED: OnceLock<Ep0SerialFailures> = OnceLock::new();

fn ep0_serial_failed() -> &'static Ep0SerialFailures {
    EP0_SERIAL_FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Serial the kernel cached at enumeration, matched on bus/device number.
/// Cheap file read, and unlike an EP0 request it cannot stall or upset a
/// device that does not implement string descriptors.
fn sysfs_serial(bus: u8, address: u8) -> Option<String> {
    for entry in std::fs::read_dir("/sys/bus/usb/devices").ok()?.flatten() {
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
            return Some(serial.to_string());
        }
    }
    None
}

pub fn enumerate_devices() -> Result<Vec<DetectedDevice>> {
    let usb_devices = rusb::devices()?;
    let mut found = Vec::new();

    for device in usb_devices.iter() {
        let desc = match device.device_descriptor() {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "USB device at bus {} addr {}: descriptor read failed ({e}), skipping",
                    device.bus_number(),
                    device.address()
                );
                continue;
            }
        };

        let vid = desc.vendor_id();
        let pid = desc.product_id();
        let id = UsbId::new(vid, pid);

        if let Some(entry) = KNOWN_DEVICES.iter().find(|e| e.id == id) {
            let bus = device.bus_number();
            let address = device.address();

            // FIX: ask the kernel, which cached the serial at enumeration, and
            // only fall back to the device itself. This runs once a second for
            // every known device, and the previous version opened each one and
            // issued an EP0 string-descriptor request every time.
            //
            // Some controllers stop answering string descriptors while their
            // bulk pipe still works — the HydroShift II LCD Circle here returned
            // nothing on 17 consecutive attempts. Asking anyway, once a second,
            // wedges it: after roughly 47s it stops accepting writes altogether,
            // reads keep succeeding, and the fans quietly decay to minimum with
            // nothing logged. Driving the same device from a script that never
            // enumerates ran 180s without a single failure.
            let serial = sysfs_serial(bus, address).or_else(|| {
                let key = (vid, pid, bus, address);
                let mut failed = ep0_serial_failed()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if failed.contains(&key) {
                    return None;
                }
                let got = device
                    .open()
                    .ok()
                    .and_then(|h| h.read_serial_number_string_ascii(&desc).ok());
                if got.is_none() {
                    failed.insert(key);
                    debug!(
                        "{:04x}:{:04x} at bus {} addr {} answered neither sysfs nor EP0 serial, \
                         caching the negative result",
                        vid, pid, bus, address
                    );
                }
                got
            });

            debug!(
                "Found {} ({:04x}:{:04x}) at bus {} addr {} serial={}",
                entry.name,
                vid,
                pid,
                bus,
                address,
                serial.as_deref().unwrap_or("none")
            );

            found.push(DetectedDevice {
                device,
                family: entry.family,
                name: entry.name,
                vid,
                pid,
                bus,
                address,
                serial,
                hid_usage_page: entry.hid_usage_page,
            });
        }
    }

    found.sort_by_key(|d| (d.bus, d.address));
    Ok(found)
}

/// Probe TL LCD identities via rusb. The TL LCD firmware reports the same
/// iSerial across daisy-chained devices, so they cannot be told apart by
/// serial alone — we open each one and read its `(port, index)` identity
/// record to disambiguate.
pub fn probe_tl_lcd_port_indices(
    devices: &[DetectedDevice],
    backend: HidBackend,
) -> HashMap<String, (u8, u8)> {
    let mut out = HashMap::new();
    for det in devices.iter().filter(|d| d.family == DeviceFamily::TlLcd) {
        let port_numbers = det.device.port_numbers().unwrap_or_default();
        let shared = match super::backends::open_shared_hid(
            &det.device,
            det.hid_usage_page,
            det.vid,
            det.pid,
            det.bus,
            &port_numbers,
            backend,
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("TL LCD open failed for {}: {e:#}", det.device_id());
                continue;
            }
        };
        let tl = crate::tl_lcd::TlLcdDevice::new(shared);
        match tl.read_identity_raw() {
            Ok(ident) => {
                out.insert(det.device_id(), (ident.port, ident.index));
            }
            Err(e) => warn!("TL LCD identity read failed for {}: {e:#}", det.device_id()),
        }
    }
    out
}

/// Find the rusb `Device` matching a VID/PID pair.
pub(super) fn find_usb_device(vid: u16, pid: u16) -> Option<rusb::Device<GlobalContext>> {
    rusb::devices().ok()?.iter().find(|d| {
        d.device_descriptor()
            .map(|desc| desc.vendor_id() == vid && desc.product_id() == pid)
            .unwrap_or(false)
    })
}

/// Re-export so callers can refer to the rusb context type without depending
/// on rusb directly.
pub use rusb::GlobalContext;
