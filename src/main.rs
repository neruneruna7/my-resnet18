mod data;
mod resnet18;
mod train;

use burn::{
    backend::{
        Autodiff,
        wgpu::{Metal, WgpuDevice},
    },
    optim::{AdamConfig, decay::WeightDecayConfig},
};

use crate::{resnet18::ResNet18Config, train::MnistTrainingConfig};

pub static ARTIFACT_DIR: &str = "./tmp/burn-resnet18-mnist";

fn main() {
    let device = WgpuDevice::default();
    let config = MnistTrainingConfig::new(
        ResNet18Config::new(10, 1, 64),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );
    train::run::<Autodiff<Metal>>(ARTIFACT_DIR, config, device);
}
