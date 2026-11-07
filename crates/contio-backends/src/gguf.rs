//! Just enough of the GGUF header to answer questions about a model file
//! without loading it (e.g. whether it carries MTP layers).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// Whether the model has multi-token-prediction layers (`<arch>.nextn_predict_layers` > 0),
/// usable for speculative decoding with `--spec-type draft-mtp`. Cached per file.
pub fn has_mtp(path: &Path) -> bool {
    static CACHE: LazyLock<Mutex<HashMap<PathBuf, (u64, bool)>>> = LazyLock::new(Default::default);
    let Ok(len) = std::fs::metadata(path).map(|m| m.len()) else { return false };
    if let Some(&(cached_len, v)) = CACHE.lock().expect("gguf cache poisoned").get(path)
        && cached_len == len
    {
        return v;
    }
    let v = read_u64_key(path, |k| k.ends_with(".nextn_predict_layers")).is_some_and(|n| n > 0);
    CACHE.lock().expect("gguf cache poisoned").insert(path.to_owned(), (len, v));
    v
}

/// The first integer metadata value whose key matches.
fn read_u64_key(path: &Path, wanted: impl Fn(&str) -> bool) -> Option<u64> {
    let mut r = BufReader::new(File::open(path).ok()?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" || read_u32(&mut r)? < 2 {
        return None;
    }
    let _tensors = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        if wanted(&key) {
            return read_int(&mut r, ty);
        }
        skip_value(&mut r, ty)?;
    }
    None
}

fn read_u32(r: &mut impl Read) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> Option<String> {
    let len = read_u64(r)?;
    if len > 1 << 20 {
        return None; // not a key
    }
    let mut b = vec![0u8; len as usize];
    r.read_exact(&mut b).ok()?;
    String::from_utf8(b).ok()
}

/// GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
/// 8 string, 9 array, 10 u64, 11 i64, 12 f64.
fn scalar_size(ty: u32) -> Option<u64> {
    Some(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn read_int(r: &mut impl Read, ty: u32) -> Option<u64> {
    let size = scalar_size(ty)? as usize;
    if matches!(ty, 6 | 12) {
        return None; // floats
    }
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..size]).ok()?;
    Some(u64::from_le_bytes(b))
}

fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> Option<()> {
    match ty {
        8 => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
        }
        9 => {
            let elem = read_u32(r)?;
            let n = read_u64(r)?;
            if elem == 8 {
                for _ in 0..n {
                    skip_value(r, 8)?;
                }
            } else {
                let bytes = n.checked_mul(scalar_size(elem)?)?;
                r.seek(SeekFrom::Current(i64::try_from(bytes).ok()?)).ok()?;
            }
        }
        t => {
            r.seek(SeekFrom::Current(scalar_size(t)? as i64)).ok()?;
        }
    }
    Some(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Write;

    use super::*;

    /// A header-only GGUF with string, array and integer values.
    pub fn write_gguf(path: &Path, nextn: Option<u32>) {
        let mut f = File::create(path).unwrap();
        let s = |f: &mut File, v: &str| {
            f.write_all(&(v.len() as u64).to_le_bytes()).unwrap();
            f.write_all(v.as_bytes()).unwrap();
        };
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&(3u64 + u64::from(nextn.is_some())).to_le_bytes()).unwrap();
        s(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        s(&mut f, "qwen35moe");
        s(&mut f, "tokenizer.ggml.tokens");
        f.write_all(&9u32.to_le_bytes()).unwrap();
        f.write_all(&8u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        s(&mut f, "a");
        s(&mut f, "bc");
        s(&mut f, "qwen35moe.block_count");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&41u32.to_le_bytes()).unwrap();
        if let Some(n) = nextn {
            s(&mut f, "qwen35moe.nextn_predict_layers");
            f.write_all(&4u32.to_le_bytes()).unwrap();
            f.write_all(&n.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn detects_mtp_layers() {
        let d = tempfile::tempdir().unwrap();
        let (with, without, zero) = (d.path().join("a.gguf"), d.path().join("b.gguf"), d.path().join("c.gguf"));
        write_gguf(&with, Some(1));
        write_gguf(&without, None);
        write_gguf(&zero, Some(0));
        assert!(has_mtp(&with));
        assert!(!has_mtp(&without) && !has_mtp(&zero));
        assert!(!has_mtp(&d.path().join("missing.gguf")));
        std::fs::write(d.path().join("junk.gguf"), b"not a gguf").unwrap();
        assert!(!has_mtp(&d.path().join("junk.gguf")));
    }
}

#[cfg(test)]
mod real_files {
    /// `CONTIO_GGUF_MTP=<file with MTP> CONTIO_GGUF_PLAIN=<file without> cargo test -- --ignored real_files`
    #[test]
    #[ignore]
    fn real_files() {
        let with = std::env::var("CONTIO_GGUF_MTP").unwrap();
        let without = std::env::var("CONTIO_GGUF_PLAIN").unwrap();
        let t = std::time::Instant::now();
        assert!(super::has_mtp(with.as_ref()));
        assert!(!super::has_mtp(without.as_ref()));
        eprintln!("both headers read in {:?}", t.elapsed());
    }
}
