use std::io::{Cursor, Read};
use std::path::Path;

use eyre::{OptionExt, Result, WrapErr, bail, ensure, eyre};
use rayon::prelude::*;
use repugnant_pickle::{SequenceType, Value, evaluate, parse_ops};
use safetensors::tensor::Dtype as St;
use zip::{CompressionMethod, ZipArchive};

use crate::weights::{Data, Entry, map};

const REBUILD: &str = "_rebuild_tensor_v2";

pub fn torch(path: &Path, from: &str, to: &str) -> Result<Vec<(String, Entry)>> {
    let mmap = map(path)?;
    let mut zip = ZipArchive::new(Cursor::new(&mmap[..]))?;
    let pkl = zip
        .file_names()
        .filter_map(Result::ok)
        .find(|n| n.ends_with("/data.pkl"))
        .map(std::borrow::Cow::into_owned)
        .ok_or_eyre("no data.pkl")?;
    let root = pkl.trim_end_matches("data.pkl").to_owned();
    let mut buf = Vec::new();
    zip.by_name(&pkl)?.read_to_end(&mut buf)?;
    let (_, ops) = parse_ops::<()>(&buf).map_err(|e| eyre!("{}: {e:?}", path.display()))?;
    let (values, _) = evaluate(&ops, true).map_err(|e| eyre!("{}: {e}", path.display()))?;
    let found = values.iter().fold(Vec::new(), |mut acc, v| {
        tensors(v, "", &mut acc);
        acc
    });
    found
        .into_iter()
        .filter_map(|(name, t)| Some((format!("{to}{}", name.strip_prefix(from)?), t)))
        .map(|(name, t)| {
            let (storage, offset, shape, stride) = rebuild(t).ok_or_else(|| eyre!("{name}: unreadable tensor"))?;
            let file = zip.by_name(&format!("{root}data/{storage}"))?;
            ensure!(file.compression() == CompressionMethod::Stored, "{name}: compressed");
            let contiguous = shape.iter().rev().scan(1, |n, &d| {
                let s = *n;
                *n *= d;
                Some(s)
            });
            ensure!(stride.iter().rev().copied().eq(contiguous), "{name}: not contiguous");
            let start = usize::try_from(file.data_start().ok_or_eyre("no data start")?)? + 4 * offset;
            let len = 4 * shape.iter().product::<usize>();
            Ok((name, Entry { shape, dtype: St::F32, data: Data::Map(mmap.clone(), start..start + len) }))
        })
        .collect()
}

fn tensors<'v>(v: &'v Value<'v>, prefix: &str, out: &mut Vec<(String, &'v [Value<'v>])>) {
    match v {
        Value::Global(g, args) if matches!(g.as_ref(), Value::Raw(op) if format!("{op:?}").contains(REBUILD)) => {
            out.push((prefix.trim_end_matches('.').to_owned(), args));
        }
        Value::Seq(SequenceType::Tuple | SequenceType::Dict, kv) if matches!(kv.as_slice(), [Value::String(_), _]) => {
            if let [Value::String(k), v] = kv.as_slice() {
                tensors(v, &format!("{prefix}{k}."), out);
            }
        }
        Value::Seq(_, items) | Value::Global(_, items) => items.iter().for_each(|i| tensors(i, prefix, out)),
        Value::Build(a, b) => [a, b].into_iter().for_each(|i| tensors(i, prefix, out)),
        _ => {}
    }
}

fn rebuild(args: &[Value]) -> Option<(String, usize, Vec<usize>, Vec<usize>)> {
    let ints = |v: &Value| match v {
        Value::Seq(SequenceType::Tuple, s) => s
            .iter()
            .map(|x| if let Value::Int(n) = x { usize::try_from(*n).ok() } else { None })
            .collect(),
        _ => None,
    };
    let [Value::Seq(SequenceType::Tuple, a)] = args else { return None };
    let [Value::PersId(pid), Value::Int(offset), shape, stride, ..] = a.as_slice() else { return None };
    let Value::Seq(SequenceType::Tuple, pid) = pid.as_ref() else { return None };
    let [_, Value::Raw(kind), Value::String(storage), ..] = pid.as_slice() else { return None };
    format!("{kind:?}")
        .contains("FloatStorage")
        .then(|| Some(((*storage).to_owned(), usize::try_from(*offset).ok()?, ints(shape)?, ints(stride)?)))
        .flatten()
}

pub fn rvm(path: &Path) -> Result<Vec<(String, Entry)>> {
    let mmap = map(path)?;
    let mut zip = ZipArchive::new(Cursor::new(&mmap[..]))?;
    let entries = (0..zip.len())
        .filter_map(|i| {
            let file = zip.by_index(i).ok()?;
            let name = rvm_name(file.name().ok()?.strip_suffix(".npy")?)?;
            let start = usize::try_from(file.data_start()?).ok()?;
            let len = usize::try_from(file.size()).ok()?;
            (file.compression() == CompressionMethod::Stored).then_some((name, start..start + len))
        })
        .collect::<Vec<_>>();
    let mut out = entries
        .into_par_iter()
        .map(|(name, range)| {
            let (shape, bytes) = npy(&mmap[range]).wrap_err_with(|| name.clone())?;
            let v = bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            let (shape, v) = jax_to_torch(&name, &shape, v)?;
            Ok((name, shape, v))
        })
        .collect::<Result<Vec<_>>>()?;
    let d = out.iter().find(|(n, ..)| n == "rnn_core.state_layer_norm.weight").map_or(0, |(_, s, _)| s[0]);
    out.push(("rnn_core.state_layer_norm.bias".into(), vec![d], vec![0.0; d]));
    out.push(("base_posenc".into(), vec![16, 16, d], posenc(16 * 16, d)));
    Ok(out
        .into_iter()
        .map(|(name, shape, v)| {
            let data = Data::Owned(bytemuck::cast_slice(&v).to_vec());
            (format!("rvm.{name}"), Entry { shape, dtype: St::F32, data })
        })
        .collect())
}

fn npy(b: &[u8]) -> Result<(Vec<usize>, &[u8])> {
    ensure!(b.len() > 12 && b.starts_with(b"\x93NUMPY"), "not an npy array");
    let (len, at) = if b[6] == 1 {
        (usize::from(u16::from_le_bytes([b[8], b[9]])), 10)
    } else {
        (usize::try_from(u32::from_le_bytes([b[8], b[9], b[10], b[11]]))?, 12)
    };
    let header = std::str::from_utf8(b.get(at..at + len).ok_or_eyre("short npy header")?)?;
    ensure!(header.contains("'<f4'") && header.contains("False"), "want C-order little-endian f32: {header}");
    let dims = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_eyre("no shape")?;
    let shape = dims.split(',').map(str::trim).filter(|d| !d.is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    Ok((shape, &b[at + len..]))
}

fn rvm_name(key: &str) -> Option<String> {
    if key == "cls_token" {
        return Some(key.into());
    }
    let top = key.split('/').next()?;
    if !["tokenizer", "encoder", "rnn_core"].contains(&top) {
        return None;
    }
    let (path, leaf) = key.rsplit_once('/')?;
    let leaf = match leaf {
        "kernel" | "scale" => "weight",
        "bias" => "bias",
        _ => return None,
    };
    let path = path
        .replace('/', ".")
        .replace("tokenizer.patch_embedding.Conv_0", "patch_embed")
        .replace("encoder.LayerNorm_0", "encoder_final_norm")
        .replace("encoder.layers_", "encoder_layers.")
        .replace("xa_blocks_", "xa_blocks.");
    Some(format!("{path}.{leaf}"))
}

fn jax_to_torch(name: &str, shape: &[usize], v: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
    let leaf = name.rsplit_once('.').map_or("", |(_, l)| l);
    let kernel = leaf == "weight" && shape.len() > 1;
    Ok(match (kernel, shape) {
        (true, &[1, kh, kw, c, d]) => {
            let t = (0..d * c * kh * kw)
                .map(|i| {
                    let (o, ci, y, x) = (i / (c * kh * kw), i / (kh * kw) % c, i / kw % kh, i % kw);
                    v[((y * kw + x) * c + ci) * d + o]
                })
                .collect();
            (vec![d, c, kh, kw], t)
        }
        (true, &[h, dh, o]) if name.contains(".out.") => transpose(h * dh, o, &v),
        (true, &[i, h, dh]) => transpose(i, h * dh, &v),
        (true, &[r, c]) => transpose(r, c, &v),
        (true, s) => bail!("{name}: unexpected kernel shape {s:?}"),
        (false, &[h, dh]) if leaf == "bias" => (vec![h * dh], v),
        (false, s) => (s.to_vec(), v),
    })
}

fn transpose(r: usize, c: usize, v: &[f32]) -> (Vec<usize>, Vec<f32>) {
    (vec![c, r], (0..r * c).map(|i| v[i % r * c + i / r]).collect())
}

fn posenc(n: usize, d: usize) -> Vec<f32> {
    (0..n * d)
        .map(|i| {
            let (p, j) = (i / d, i % d);
            let half = (j / 2) as f32;
            let angle = p as f32 / 10000f32.powf(2.0 * half / d as f32);
            if j % 2 == 0 { angle.sin() } else { angle.cos() }
        })
        .collect()
}
