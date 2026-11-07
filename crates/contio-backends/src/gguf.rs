//! Just enough of the GGUF header to answer questions about a model file
//! without loading it (e.g. whether it carries MTP layers).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// Results per file, with the file length they were computed for.
type FileCache<T> = HashMap<PathBuf, (u64, T)>;

/// Whether the model has multi-token-prediction layers (`<arch>.nextn_predict_layers` > 0),
/// usable for speculative decoding with `--spec-type draft-mtp`. Cached per file.
pub fn has_mtp(path: &Path) -> bool {
    static CACHE: LazyLock<Mutex<FileCache<bool>>> = LazyLock::new(Default::default);
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

/// Vocabulary size (`tokenizer.ggml.tokens`), to check a draft model matches.
pub fn vocab_size(path: &Path) -> Option<u64> {
    let (mut r, h) = open_header(path)?;
    for _ in 0..h.kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        if key == "tokenizer.ggml.tokens" && ty == 9 {
            let _elem = read_u32(&mut r)?;
            return read_u64(&mut r);
        }
        skip_value(&mut r, ty)?;
    }
    None
}

/// Bytes of MoE expert weights per layer (`blk.N.ffn_*_exps.*`), from the
/// tensor table: what `--n-cpu-moe N` keeps in system memory, layer by layer.
/// Empty for dense models. Cached per file.
pub fn expert_bytes_per_layer(path: &Path) -> Vec<u64> {
    static CACHE: LazyLock<Mutex<FileCache<Vec<u64>>>> = LazyLock::new(Default::default);
    let Ok(len) = std::fs::metadata(path).map(|m| m.len()) else { return Vec::new() };
    if let Some((cached_len, v)) = CACHE.lock().expect("gguf cache poisoned").get(path)
        && *cached_len == len
    {
        return v.clone();
    }
    let mut layers: Vec<u64> = Vec::new();
    for (name, bytes) in tensor_bytes(path).unwrap_or_default() {
        let Some(rest) = name.strip_prefix("blk.") else { continue };
        let Some((layer, tail)) = rest.split_once('.') else { continue };
        if !(tail.starts_with("ffn_") && tail.contains("_exps")) {
            continue;
        }
        let Ok(i) = layer.parse::<usize>() else { continue };
        if layers.len() <= i {
            layers.resize(i + 1, 0);
        }
        layers[i] += bytes;
    }
    CACHE.lock().expect("gguf cache poisoned").insert(path.to_owned(), (len, layers.clone()));
    layers
}

struct Header {
    tensors: u64,
    kv_count: u64,
}

fn open_header(path: &Path) -> Option<(BufReader<File>, Header)> {
    let mut r = BufReader::new(File::open(path).ok()?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" || read_u32(&mut r)? < 2 {
        return None;
    }
    let tensors = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    Some((r, Header { tensors, kv_count }))
}

/// The first integer metadata value whose key matches.
fn read_u64_key(path: &Path, wanted: impl Fn(&str) -> bool) -> Option<u64> {
    let (mut r, h) = open_header(path)?;
    for _ in 0..h.kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        if wanted(&key) {
            return read_int(&mut r, ty);
        }
        skip_value(&mut r, ty)?;
    }
    None
}

/// (name, bytes) of every tensor. Sizes come from the offsets: tensors are laid
/// out in order in the data section, so no table of quantization formats is needed.
/// Only single-file models (the shards of a split model each carry their own table).
fn tensor_bytes(path: &Path) -> Option<Vec<(String, u64)>> {
    let file_len = std::fs::metadata(path).ok()?.len();
    let (mut r, h) = open_header(path)?;
    let mut alignment = 32u64;
    for _ in 0..h.kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        if key == "general.alignment" {
            alignment = read_int(&mut r, ty)?.max(1);
        } else {
            skip_value(&mut r, ty)?;
        }
    }
    let mut infos: Vec<(String, u64)> = Vec::with_capacity(usize::try_from(h.tensors).ok()?.min(1 << 16));
    for _ in 0..h.tensors {
        let name = read_string(&mut r)?;
        let dims = read_u32(&mut r)?;
        r.seek(SeekFrom::Current(i64::from(dims) * 8 + 4)).ok()?; // dims, type
        infos.push((name, read_u64(&mut r)?));
    }
    let header_end = r.stream_position().ok()?;
    let data_start = header_end.div_ceil(alignment) * alignment;
    let data_len = file_len.checked_sub(data_start)?;
    let mut order: Vec<usize> = (0..infos.len()).collect();
    order.sort_by_key(|&i| infos[i].1);
    let mut sizes = vec![0u64; infos.len()];
    for (k, &i) in order.iter().enumerate() {
        let end = order.get(k + 1).map_or(data_len, |&j| infos[j].1);
        sizes[i] = end.saturating_sub(infos[i].1);
    }
    Some(infos.into_iter().zip(sizes).map(|((n, _), s)| (n, s)).collect())
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

    /// A GGUF with a tensor table: two MoE layers and one dense tensor.
    fn write_moe_gguf(path: &Path) {
        let mut f = File::create(path).unwrap();
        let s = |f: &mut File, v: &str| {
            f.write_all(&(v.len() as u64).to_le_bytes()).unwrap();
            f.write_all(v.as_bytes()).unwrap();
        };
        let tensors: [(&str, u64); 4] =
            [("token_embd.weight", 64), ("blk.0.ffn_up_exps.weight", 320), ("blk.0.attn_q.weight", 96), ("blk.1.ffn_down_exps.weight", 160)];
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        s(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        s(&mut f, "test");
        let mut offset = 0u64;
        for (name, bytes) in tensors {
            s(&mut f, name);
            f.write_all(&1u32.to_le_bytes()).unwrap(); // one dimension
            f.write_all(&bytes.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap(); // type
            f.write_all(&offset.to_le_bytes()).unwrap();
            offset += bytes; // all multiples of 32: no padding
        }
        let pos = f.stream_position().unwrap();
        let start = pos.div_ceil(32) * 32;
        f.write_all(&vec![0u8; (start - pos + offset) as usize]).unwrap();
    }

    #[test]
    fn expert_bytes_from_the_tensor_table() {
        let d = tempfile::tempdir().unwrap();
        let moe = d.path().join("moe.gguf");
        write_moe_gguf(&moe);
        assert_eq!(expert_bytes_per_layer(&moe), vec![320, 160]);
        let dense = d.path().join("dense.gguf");
        write_gguf(&dense, None);
        assert!(expert_bytes_per_layer(&dense).is_empty());
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
        assert_eq!(vocab_size(&with), Some(2), "two tokens in the fixture");
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
        let experts = super::expert_bytes_per_layer(without.as_ref());
        let total: u64 = experts.iter().sum();
        let file = std::fs::metadata(&without).unwrap().len();
        eprintln!("{} MoE layers, {:.1} GiB of {:.1} GiB in experts", experts.len(), total as f64 / (1u64 << 30) as f64, file as f64 / (1u64 << 30) as f64);
        assert!(experts.len() == 40 && total > file * 8 / 10 && total < file, "35B-A3B: experts are most of the file");
    }
}
