# contio

Runs local LLMs on an AMD Ryzen AI laptop (CPU, Radeon iGPU and XDNA2 NPU) behind one
OpenAI- and Anthropic-compatible API, with a desktop app to pick, install, try and
benchmark models. Design and status: [`docs/SPEC.md`](docs/SPEC.md).

## Install (AppImage)

Download `contio_<version>_amd64.AppImage` from the
[releases](https://github.com/f0ld/contio/releases), then:

```bash
chmod +x contio_*_amd64.AppImage
./contio_*_amd64.AppImage             # desktop app; starts the daemon if it isn't running
./contio_*_amd64.AppImage daemon      # only the daemon (API on http://127.0.0.1:8080)
./contio_*_amd64.AppImage cli ps      # the command-line client
```

The app starts the daemon if none is running and stops it when you close the window.
To keep the API up without the window and start it at login, turn on **Settings →
Background service → Run in the background**: it installs `contiod` as a systemd user
service (binaries in `~/.local/share/contio/bin`, refreshed when you start a newer
AppImage). Start/Stop are there too, and in the header when the daemon is offline.

Inference backends (llama.cpp, whisper.cpp, FastFlowLM) are set up from the app the first time
a model needs them; NPU support needs the XRT runtime (`docs/spikes/m0-npu.md`).

## Build from source

```bash
cargo build --release                 # contiod (daemon) and contio (CLI) in target/release/
cd apps/gui && npm install && npm run tauri dev   # desktop app against a running contiod
```

Optional config: `~/.config/contio/config.toml` (`docs/config.example.toml`).

## Releases

```bash
scripts/release.sh 0.4.0              # sets the version everywhere, commits, tags v0.4.0
git push origin main v0.4.0           # GitHub Actions builds the AppImage and publishes the release
```

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
Unless you state otherwise, any contribution you submit for inclusion in contio is dual licensed as above, without additional terms.

Benchmark data from GSM8K, HumanEval, MMLU-Pro and LibriSpeech keeps its own license; see [THIRD_PARTY.md](THIRD_PARTY.md).
