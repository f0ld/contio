//! contio desktop app. A thin shell around the Svelte UI, which talks to
//! `contiod` over its management API (spec §9.2); the daemon is not embedded.
//!
//! In the AppImage, `contiod` and `contio` ship next to this binary. Without
//! the background service, the app starts the daemon if nothing answers on its
//! port and stops it again on exit (only a daemon it started). With the
//! service (Settings → Background service), systemd runs the daemon and the app
//! only controls it. `daemon …` / `cli …` pass through to the bundled binaries:
//!
//! ```text
//! contio.AppImage              # GUI
//! contio.AppImage daemon       # contiod in the foreground, no GUI
//! contio.AppImage cli ps       # the contio CLI
//! ```

mod service;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Manager, RunEvent};

/// Where the GUI expects the daemon (also in the CSP of tauri.conf.json).
const DAEMON_ADDR: &str = "127.0.0.1:8080";
/// The daemon gives open requests 10 s, then stops its backends.
const STOP_TIMEOUT: Duration = Duration::from_secs(25);

/// Variables the AppImage runtime sets for the GUI's bundled libraries. The
/// daemon and the inference servers it launches must not inherit them.
const APPIMAGE_ENV: [&str; 10] = [
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_DATA_PREFIX",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_MODULE_DIR",
    "GSETTINGS_SCHEMA_DIR",
    "PYTHONHOME",
];

/// A program shipped next to this executable (AppImage sidecars).
fn sibling(name: &str) -> Option<PathBuf> {
    let p = std::env::current_exe().ok()?.parent()?.join(name);
    p.is_file().then_some(p)
}

fn clean(cmd: &mut Command) -> &mut Command {
    for v in APPIMAGE_ENV {
        cmd.env_remove(v);
    }
    cmd
}

/// `daemon …` / `cli …`: replace this process with the sidecar. Returns
/// normally (GUI) for any other arguments.
pub fn dispatch() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let program = match args.first().map(String::as_str) {
        Some("daemon") => "contiod",
        Some("cli") => "contio",
        _ => return,
    };
    let Some(path) = sibling(program) else {
        eprintln!("{program} isn't bundled with this build (it is in the AppImage)");
        std::process::exit(1);
    };
    use std::os::unix::process::CommandExt;
    let err = clean(&mut Command::new(path)).args(&args[1..]).exec();
    eprintln!("starting {program}: {err}");
    std::process::exit(1);
}

fn daemon_running() -> bool {
    let addr: SocketAddr = DAEMON_ADDR.parse().expect("static address");
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn wait_until(stopped: bool, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if daemon_running() != stopped {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

/// The daemon this app started, if any.
struct OwnDaemon(Mutex<Option<Child>>);

impl OwnDaemon {
    fn alive(&self) -> bool {
        self.0.lock().expect("daemon lock").as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)))
    }
}

fn spawn_daemon() -> Result<Child, String> {
    let path = sibling("contiod").ok_or("contiod isn't bundled with this build; start it yourself")?;
    let log_dir = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .ok_or("no home directory")?
        .join("contio");
    let _ = std::fs::create_dir_all(&log_dir);
    let log = std::fs::File::create(log_dir.join("contiod.log")).map_err(|e| e.to_string())?;
    clean(&mut Command::new(path))
        .stdin(Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .spawn()
        .map_err(|e| format!("starting contiod: {e}"))
}

/// SIGTERM (the daemon stops its backends), SIGKILL after a timeout.
fn stop_child(mut child: Child) {
    let _ = Command::new("kill").arg("-TERM").arg(child.id().to_string()).status();
    let deadline = Instant::now() + STOP_TIMEOUT;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Ask a daemon started elsewhere to stop (`POST /api/shutdown`).
fn request_shutdown() -> Result<(), String> {
    let addr: SocketAddr = DAEMON_ADDR.parse().expect("static address");
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(1)).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    s.write_all(b"POST /api/shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    // "HTTP/1.1 202"
    let mut head = [0u8; 12];
    s.read_exact(&mut head).map_err(|e| e.to_string())?;
    if head.ends_with(b"202") {
        Ok(())
    } else {
        Err(format!("the daemon refused to stop ({})", String::from_utf8_lossy(&head)))
    }
}

#[derive(Serialize)]
struct DaemonInfo {
    running: bool,
    /// "app", "service", "external" or "none"
    owner: &'static str,
    /// This build ships contiod (the AppImage).
    bundled: bool,
    service_available: bool,
    service_installed: bool,
    service_version: Option<String>,
}

fn info(own: &OwnDaemon) -> DaemonInfo {
    let running = daemon_running();
    let service_installed = service::installed();
    let owner = if !running {
        "none"
    } else if own.alive() {
        "app"
    } else if service_installed && service::active() {
        "service"
    } else {
        "external"
    };
    DaemonInfo {
        running,
        owner,
        bundled: sibling("contiod").is_some(),
        service_available: service::available(),
        service_installed,
        service_version: service::installed_version(),
    }
}

/// Blocking work off the UI thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn daemon_info(app: tauri::AppHandle) -> Result<DaemonInfo, String> {
    blocking(move || info(&app.state::<OwnDaemon>())).await
}

#[tauri::command]
async fn daemon_start(app: tauri::AppHandle) -> Result<DaemonInfo, String> {
    blocking(move || {
        let own = app.state::<OwnDaemon>();
        if !daemon_running() {
            if service::installed() {
                service::start()?;
            } else {
                *own.0.lock().expect("daemon lock") = Some(spawn_daemon()?);
            }
            if !wait_until(false, Duration::from_secs(20)) {
                return Err("contiod didn't come up; see ~/.local/state/contio/contiod.log".into());
            }
        }
        Ok(info(&own))
    })
    .await?
}

#[tauri::command]
async fn daemon_stop(app: tauri::AppHandle) -> Result<DaemonInfo, String> {
    blocking(move || {
        let own = app.state::<OwnDaemon>();
        let child = own.0.lock().expect("daemon lock").take();
        match (child, info(&own).owner) {
            (Some(c), _) => stop_child(c),
            (None, "service") => service::stop()?,
            (None, "external") => {
                request_shutdown()?;
                wait_until(true, STOP_TIMEOUT);
            }
            _ => {}
        }
        Ok(info(&own))
    })
    .await?
}

/// Turn the background service on (installs and starts it, replacing a daemon
/// this app started) or off (stops and removes it).
#[tauri::command]
async fn service_set(app: tauri::AppHandle, enabled: bool) -> Result<DaemonInfo, String> {
    blocking(move || {
        let own = app.state::<OwnDaemon>();
        if enabled {
            let contiod = sibling("contiod").ok_or("the background service needs the AppImage build")?;
            if let Some(c) = own.0.lock().expect("daemon lock").take() {
                stop_child(c); // the service takes over the port
            }
            service::enable(&contiod)?;
            wait_until(false, Duration::from_secs(20));
        } else {
            service::disable()?;
        }
        Ok(info(&own))
    })
    .await?
}

/// On launch: use the service if it's set up (refreshing its binaries after
/// an AppImage update), else start a daemon if none is running.
fn launch_daemon(own: &OwnDaemon) {
    if service::installed() {
        if let Some(bundled) = sibling("contiod") {
            match service::refresh(&bundled) {
                Ok(true) => eprintln!("background service updated to this version"),
                Ok(false) => {}
                Err(e) => eprintln!("updating the background service: {e}"),
            }
        }
        if !service::active()
            && let Err(e) = service::start()
        {
            eprintln!("{e}");
        }
        return;
    }
    if daemon_running() || sibling("contiod").is_none() {
        return; // already up, or a dev build (start contiod yourself)
    }
    match spawn_daemon() {
        Ok(c) => *own.0.lock().expect("daemon lock") = Some(c),
        Err(e) => eprintln!("{e}"),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(OwnDaemon(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![daemon_info, daemon_start, daemon_stop, service_set])
        .setup(|app| {
            let handle = app.handle().clone();
            // systemctl and a version check can take a moment: not on the UI thread
            std::thread::spawn(move || launch_daemon(&handle.state::<OwnDaemon>()));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event
                && let Some(child) = app.state::<OwnDaemon>().0.lock().expect("daemon lock").take()
            {
                stop_child(child);
            }
        });
}
