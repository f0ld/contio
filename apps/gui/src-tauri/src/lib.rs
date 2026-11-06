//! contio desktop app. A thin shell around the Svelte UI, which talks to
//! `contiod` over its management API (spec §9.2); the daemon is not embedded.
//!
//! In the AppImage, `contiod` and `contio` ship next to this binary. The app
//! starts the daemon if nothing answers on its port and stops it again on exit
//! (only a daemon it started), and passes `daemon …` / `cli …` through:
//!
//! ```text
//! contio.AppImage              # GUI (starts contiod if needed)
//! contio.AppImage daemon       # contiod in the foreground, no GUI
//! contio.AppImage cli ps       # the contio CLI
//! ```

use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

/// Where the GUI expects the daemon (also in the CSP of tauri.conf.json).
const DAEMON_ADDR: &str = "127.0.0.1:8080";
/// The daemon gives open requests 10 s, then stops its backends.
const STOP_TIMEOUT: Duration = Duration::from_secs(20);

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

/// The daemon this app started, if any.
struct OwnDaemon(Mutex<Option<Child>>);

fn start_daemon() -> Option<Child> {
    if daemon_running() {
        return None;
    }
    let path = sibling("contiod")?; // dev builds: start contiod yourself
    let log_dir = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?
        .join("contio");
    let _ = std::fs::create_dir_all(&log_dir);
    let log = std::fs::File::create(log_dir.join("contiod.log")).ok()?;
    let child = clean(&mut Command::new(path))
        .stdin(Stdio::null())
        .stdout(log.try_clone().ok()?)
        .stderr(log)
        .spawn();
    match child {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("starting contiod: {e}");
            None
        }
    }
}

/// SIGTERM (the daemon stops its backends), SIGKILL after a timeout.
fn stop_daemon(mut child: Child) {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            app.manage(OwnDaemon(Mutex::new(start_daemon())));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event
                && let Some(child) = app.state::<OwnDaemon>().0.lock().expect("daemon lock").take()
            {
                stop_daemon(child);
            }
        });
}
