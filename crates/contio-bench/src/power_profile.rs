//! Switching the platform power profile for a run (spec §8.1): it caps the
//! SoC's power, and M0 measured +40% prefill between power-saver and balanced.
//!
//! Talks to `net.hadess.PowerProfiles` (power-profiles-daemon, or tuned-ppd on
//! Fedora), as GNOME's power menu does. performance and power-saver are
//! *held*: the daemon drops the hold when our connection closes, so the
//! user's profile comes back even if contiod crashes. balanced can't be held;
//! it is set and restored afterwards.

use zbus::{Connection, proxy};

#[proxy(
    interface = "net.hadess.PowerProfiles",
    default_service = "net.hadess.PowerProfiles",
    default_path = "/net/hadess/PowerProfiles"
)]
trait PowerProfiles {
    fn hold_profile(&self, profile: &str, reason: &str, application_id: &str) -> zbus::Result<u32>;
    fn release_profile(&self, cookie: u32) -> zbus::Result<()>;
    #[zbus(property)]
    fn active_profile(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn set_active_profile(&self, profile: &str) -> zbus::Result<()>;
}

pub const PROFILES: [&str; 3] = ["power-saver", "balanced", "performance"];

enum Restore {
    Nothing,
    Release(u32),
    Set(String),
}

/// Keeps the requested profile until [`ProfileGuard::restore`] (or drop).
pub struct ProfileGuard {
    conn: Connection,
    restore: Restore,
    /// What was active before.
    pub previous: String,
}

pub async fn current() -> Result<String, String> {
    let conn = Connection::system().await.map_err(err)?;
    PowerProfilesProxy::new(&conn).await.map_err(err)?.active_profile().await.map_err(err)
}

/// Switch to `profile` for the duration of a benchmark.
pub async fn apply(profile: &str) -> Result<ProfileGuard, String> {
    if !PROFILES.contains(&profile) {
        return Err(format!("unknown power profile `{profile}` (power-saver, balanced, performance)"));
    }
    let conn = Connection::system().await.map_err(err)?;
    let proxy = PowerProfilesProxy::new(&conn).await.map_err(err)?;
    let previous = proxy.active_profile().await.map_err(err)?;
    let restore = if previous == profile {
        Restore::Nothing
    } else if profile == "balanced" {
        proxy.set_active_profile(profile).await.map_err(err)?;
        Restore::Set(previous.clone())
    } else {
        Restore::Release(proxy.hold_profile(profile, "benchmark", "contio").await.map_err(err)?)
    };
    Ok(ProfileGuard { conn, restore, previous })
}

impl ProfileGuard {
    pub async fn restore(self) -> Result<(), String> {
        let proxy = PowerProfilesProxy::new(&self.conn).await.map_err(err)?;
        match &self.restore {
            Restore::Nothing => Ok(()),
            Restore::Release(cookie) => proxy.release_profile(*cookie).await.map_err(err),
            Restore::Set(p) => proxy.set_active_profile(p).await.map_err(err),
        }
    }
}

fn err(e: zbus::Error) -> String {
    format!("power profile: {e}")
}
