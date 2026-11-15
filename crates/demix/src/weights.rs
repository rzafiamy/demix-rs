//! Weight store shared by the checkpoint and GGUF loaders, and the layers
//! built from it: a linear layer that runs dense or quantized (`QMatMul`),
//! and BS-RoFormer's RMSNorm.

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{D, DType, Device, Module, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Weights {
    map: HashMap<String, Arc<QTensor>>,
    pub device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, Arc<QTensor>>, device: Device) -> Self {
        Self { map, device }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<&Arc<QTensor>> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor '{name}'"))
    }

    /// Dense tensor in `dtype` on the model device.
    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let q = self.raw(name)?;
        let t = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
        Ok(t.to_device(&self.device)?)
    }

    pub fn linear(&self, prefix: &str, bias: bool, dtype: DType) -> Result<Linear> {
        let name = format!("{prefix}.weight");
        let q = self.raw(&name)?;
        let bias = if bias {
            Some(self.get(&format!("{prefix}.bias"), dtype)?)
        } else {
            None
        };
        let inner = match q.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                let t = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
                LinearInner::Dense(t.to_device(&self.device)?)
            }
            _ => {
                let q = if q.device().same_device(&self.device) {
                    q.clone()
                } else {
                    // GGUF tensors are read straight to the device; this is the
                    // checkpoint path (never quantized), kept for completeness.
                    let t = q.dequantize(&Device::Cpu)?;
                    Arc::new(QTensor::quantize(&t.to_device(&self.device)?, q.dtype())?)
                };
                LinearInner::Quant(QMatMul::from_arc(q)?)
            }
        };
        Ok(Linear { inner, bias })
    }
}

#[derive(Debug, Clone)]
enum LinearInner {
    /// `[out, in]` weight in the layer's compute dtype.
    Dense(Tensor),
    /// Block-quantized weight; takes and returns F32 activations.
    Quant(QMatMul),
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
    bias: Option<Tensor>,
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Flatten to 2-D: candle 0.11 matmul against a stride-0
        // (broadcast_left) batch returns wrong values.
        let dims = xs.dims().to_vec();
        let k = dims[dims.len() - 1];
        let ys = match &self.inner {
            LinearInner::Dense(w) => {
                let x2 = xs.to_dtype(w.dtype())?.reshape(((), k))?;
                x2.matmul(&w.t()?)?
            }
            LinearInner::Quant(q) => {
                let x2 = xs.to_dtype(DType::F32)?.reshape(((), k))?;
                q.forward(&x2)?.to_dtype(xs.dtype())?
            }
        };
        let ys = match &self.bias {
            Some(b) => ys.broadcast_add(&b.to_dtype(ys.dtype())?)?,
            None => ys,
        };
        let mut out = dims;
        *out.last_mut().unwrap() = ys.dim(1)?;
        ys.reshape(out)
    }
}

/// `F.normalize(x, dim=-1) * sqrt(dim) * gamma`, computed in F32, output in
/// the input dtype.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    gamma: Tensor,
    scale: f64,
}

impl RmsNorm {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let gamma = w.get(&format!("{prefix}.gamma"), DType::F32)?;
        let dim = gamma.dim(0)?;
        Ok(Self {
            gamma,
            scale: (dim as f64).sqrt(),
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = xs.dtype();
        let x = xs.to_dtype(DType::F32)?;
        let norm = x
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12f32, f32::MAX)?;
        let y = x.broadcast_div(&norm)?.affine(self.scale, 0.0)?;
        y.broadcast_mul(&self.gamma)?.to_dtype(dtype)
    }
}
