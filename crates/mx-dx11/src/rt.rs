//! Ray tracing: not on this tier (acceleration structures and ray queries need bindless).

use std::ops::{Index, IndexMut};

use mx_types as t;

use crate::{Device, Error, Result};

#[derive(Clone)]
pub struct Blas(());

pub struct Tlas {
    instances: Vec<Option<t::TlasInstance<Blas>>>,
}

impl Tlas {
    pub fn max_instances(&self) -> u32 {
        self.instances.len() as u32
    }

    /// (never called: this backend makes no acceleration structures)
    pub fn index(&self) -> u32 {
        unreachable!("no acceleration structures on this backend")
    }
}

impl Index<usize> for Tlas {
    type Output = Option<t::TlasInstance<Blas>>;
    fn index(&self, i: usize) -> &Self::Output {
        &self.instances[i]
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.instances[i]
    }
}

impl Device {
    pub fn create_blas(&self, _desc: &t::CreateBlasDescriptor<'_>, _sizes: &[t::BlasTriangleGeometrySizeDescriptor]) -> Result<Blas> {
        Err(Error::Unsupported("ray queries (acceleration structures need Direct3D 12)".into()))
    }
    pub fn create_tlas(&self, _desc: &t::CreateTlasDescriptor<'_>) -> Result<Tlas> {
        Err(Error::Unsupported("ray queries (acceleration structures need Direct3D 12)".into()))
    }
}
