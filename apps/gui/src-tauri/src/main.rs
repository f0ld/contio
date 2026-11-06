// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // `daemon …` / `cli …` run the bundled contiod / contio instead of the GUI
    contio_gui_lib::dispatch();
    contio_gui_lib::run()
}
