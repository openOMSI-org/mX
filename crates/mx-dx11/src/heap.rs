//! The heap on the CPU: what each index stands for, with the deferred reuse of the other
//! backends. Draws bind from it (see `command::bind_stage`).

use windows::Win32::Graphics::Direct3D11::{ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11UnorderedAccessView};

use crate::resource::Buffer;
use crate::{Error, Result};

/// A buffer view: the range shaders read (or write) through it.
#[derive(Clone)]
pub(crate) struct BufferSlot {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
}

struct Slots<T> {
    items: Vec<Option<T>>,
    capacity: u32,
    free: Vec<u32>,
    deferred: Vec<(u64, u32)>,
    in_use: usize,
}

impl<T: Clone> Slots<T> {
    fn new(capacity: u32) -> Self {
        Slots {
            items: Vec::new(),
            capacity,
            free: Vec::new(),
            deferred: Vec::new(),
            in_use: 0,
        }
    }

    fn add(&mut self, item: T, what: &'static str) -> Result<u32> {
        let slot = match self.free.pop() {
            Some(s) => s,
            None if self.items.len() as u32 == self.capacity => return Err(Error::HeapFull(what)),
            None => {
                self.items.push(None);
                self.items.len() as u32 - 1
            }
        };
        self.items[slot as usize] = Some(item);
        self.in_use += 1;
        Ok(slot)
    }

    fn remove(&mut self, slot: u32, after: u64) {
        self.items[slot as usize] = None;
        self.in_use -= 1;
        self.deferred.push((after, slot));
    }

    fn get(&self, slot: u32) -> Option<&T> {
        self.items.get(slot as usize).and_then(|x| x.as_ref())
    }

    fn reclaim(&mut self, completed: u64) {
        let free = &mut self.free;
        self.deferred.retain(|&(after, slot)| {
            let done = after <= completed;
            if done {
                free.push(slot);
            }
            !done
        });
    }
}

/// A texture view: its shader resource view (`TEXTURE_BINDING`), and its unordered access view
/// when it is a storage texture of the heap.
#[derive(Clone)]
pub(crate) struct TextureSlot {
    pub srv: Option<ID3D11ShaderResourceView>,
    pub uav: Option<ID3D11UnorderedAccessView>,
}

pub(crate) struct Heap {
    textures: Slots<TextureSlot>,
    samplers: Slots<ID3D11SamplerState>,
    buffers: Slots<BufferSlot>,
}

// SAFETY: Direct3D 11 views and states are free-threaded objects.
unsafe impl Send for Heap {}

impl Heap {
    pub fn new(textures: u32, samplers: u32, buffers: u32) -> Heap {
        Heap {
            textures: Slots::new(textures),
            samplers: Slots::new(samplers),
            buffers: Slots::new(buffers),
        }
    }

    pub fn add_texture(&mut self, t: TextureSlot) -> Result<u32> {
        self.textures.add(t, "textures")
    }
    pub fn remove_texture(&mut self, slot: u32, after: u64) {
        self.textures.remove(slot, after);
    }
    pub fn texture(&self, slot: u32) -> Option<&TextureSlot> {
        self.textures.get(slot)
    }

    pub fn add_sampler(&mut self, s: ID3D11SamplerState) -> Result<u32> {
        self.samplers.add(s, "samplers")
    }
    pub fn remove_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.remove(slot, after);
    }
    pub fn sampler(&self, slot: u32) -> Option<&ID3D11SamplerState> {
        self.samplers.get(slot)
    }

    pub fn add_buffer(&mut self, b: BufferSlot) -> Result<u32> {
        self.buffers.add(b, "buffers")
    }
    pub fn remove_buffer(&mut self, slot: u32, after: u64) {
        self.buffers.remove(slot, after);
    }
    pub fn buffer(&self, slot: u32) -> Option<&BufferSlot> {
        self.buffers.get(slot)
    }

    pub fn reclaim(&mut self, completed: u64) {
        self.textures.reclaim(completed);
        self.samplers.reclaim(completed);
        self.buffers.reclaim(completed);
    }

    pub fn textures_in_use(&self) -> usize {
        self.textures.in_use
    }
    pub fn buffers_in_use(&self) -> usize {
        self.buffers.in_use
    }
}
