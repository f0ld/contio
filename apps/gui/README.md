# contio GUI

Tauri 2 + SvelteKit (SPA mode) + TypeScript. A thin client of `contiod`'s
management API; it does not embed the daemon (spec §9.2).

```bash
cargo run -p contiod          # terminal 1: daemon on 127.0.0.1:8080
cd apps/gui && npm install
npm run tauri dev             # terminal 2: app with hot reload
npm run check                 # svelte-check / TypeScript
```

Set `VITE_CONTIOD_URL` to point at a daemon on a different address (also update
the CSP `connect-src` in `src-tauri/tauri.conf.json`).
