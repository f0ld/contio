//! contio desktop app. A thin shell around the Svelte UI, which talks to
//! `contiod` over its management API (spec §9.2); the daemon is not embedded.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
