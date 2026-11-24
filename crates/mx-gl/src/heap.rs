//! The heap on the CPU: what each index stands for, with the deferred reuse of the other
//! backends. Draws bind from it (see `command::bind_slots`).

use crate::{Error, Result};

/// A texture view as GL binds it: the texture, its target, the mips it shows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureSlot {
    pub raw: glow::Texture,
    pub target: u32,
    pub base_mip: u32,
    pub mip_count: u32,
    /// For depth-stencil textures: which aspect is sampled (`DEPTH_STENCIL_TEXTURE_MODE`).
    pub depth_stencil_mode: Option<u32>,
    /// A storage texture's image binding: its internal format and layer (GL 4.2 / ES 3.1
    /// image units), for the heap's storage-texture views.
    pub image: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BufferSlot {
    pub raw: glow::Buffer,
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

impl<T: Copy> Slots<T> {
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

    fn get(&self, slot: u32) -> Option<T> {
        self.items.get(slot as usize).copied().flatten()
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

pub(crate) struct Heap {
    textures: Slots<TextureSlot>,
    samplers: Slots<glow::Sampler>,
    buffers: Slots<BufferSlot>,
}

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
    pub fn texture(&self, slot: u32) -> Option<TextureSlot> {
        self.textures.get(slot)
    }

    pub fn add_sampler(&mut self, s: glow::Sampler) -> Result<u32> {
        self.samplers.add(s, "samplers")
    }
    pub fn remove_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.remove(slot, after);
    }
    pub fn sampler(&self, slot: u32) -> Option<glow::Sampler> {
        self.samplers.get(slot)
    }

    pub fn add_buffer(&mut self, b: BufferSlot) -> Result<u32> {
        self.buffers.add(b, "buffers")
    }
    pub fn remove_buffer(&mut self, slot: u32, after: u64) {
        self.buffers.remove(slot, after);
    }
    pub fn buffer(&self, slot: u32) -> Option<BufferSlot> {
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
