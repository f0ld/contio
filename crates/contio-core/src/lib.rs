//! Shared domain types for contio.
//!
//! Everything here is plain data that crosses crate and process boundaries
//! (daemon ↔ GUI/CLI), so all types are `serde`-serializable and carry no logic
//! beyond small conveniences.

pub mod hw;
