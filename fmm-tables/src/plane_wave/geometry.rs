//! The geometry of plane-wave M2L with merge-and-shift (CONVENTIONS §3.14, "Directions"
//! and "Merge and shift"): the six directions, the 216 source slots of a parent, and the
//! accumulators of each direction.
//!
//! Coordinates are box indices on the level of the targets (the children). A parent P
//! has the children 2P + b, b ∈ {0, 1}³, with child index o = 4 b_x + 2 b_y + b_z
//! (CONVENTIONS §3.12, "Child octants"). The V-list sources of its children are among the
//! children of its 27 colleagues, the 6 × 6 × 6 boxes 2P − 2 + (a_x, a_y, a_z),
//! a ∈ {0, …, 5}³: the *slots* of the parent, numbered 36 a_x + 6 a_y + a_z. A V pair
//! from slot a to child b has the offset d = target − source = b + 2 − a, so a slot and
//! a child index fix the pair, and everything here is independent of the parent, the
//! level and the tree.

/// The slots of a parent: the children of its 27 colleagues.
pub const SLOTS: usize = 216;

/// The children of a parent.
pub const CHILDREN: usize = 8;

/// The coordinates a ∈ {0, …, 5}³ of `slot`.
pub fn slot_coordinates(slot: usize) -> [i64; 3] {
    [slot / 36, (slot / 6) % 6, slot % 6].map(|x| x as i64)
}

/// The bits b ∈ {0, 1}³ of the child index `child` (CONVENTIONS §3.12).
pub fn child_bits(child: usize) -> [i64; 3] {
    [(child >> 2) & 1, (child >> 1) & 1, child & 1].map(|x| x as i64)
}

/// The slot of the source of the V pair with offset `d` (target − source) to the child
/// `child`: a = b + 2 − d, or `None` if that lies outside the 6 × 6 × 6 slots.
pub fn slot_of(child: usize, d: [i64; 3]) -> Option<usize> {
    let b = child_bits(child);
    let a: [i64; 3] = core::array::from_fn(|i| b[i] + 2 - d[i]);
    a.iter()
        .all(|x| (0..6).contains(x))
        .then(|| (36 * a[0] + 6 * a[1] + a[2]) as usize)
}

/// The six directions: the axis along which the target lies at least two boxes from the
/// source after the frame's rotation, and its sign (CONVENTIONS §3.14, "Directions").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Along +z: the target above the source.
    Up,
    /// Along −z.
    Down,
    /// Along +y.
    North,
    /// Along −y.
    South,
    /// Along +x.
    East,
    /// Along −x.
    West,
}

impl Direction {
    /// Every direction, in index order.
    pub const ALL: [Direction; 6] = [
        Direction::Up,
        Direction::Down,
        Direction::North,
        Direction::South,
        Direction::East,
        Direction::West,
    ];

    /// The direction of the V-list offset `d` (target − source) by its first axis of z,
    /// y, x with |dᵢ| ≥ 2, or `None` if there is none.
    pub fn of(d: [i64; 3]) -> Option<Self> {
        match d {
            [_, _, z] if z >= 2 => Some(Self::Up),
            [_, _, z] if z <= -2 => Some(Self::Down),
            [_, y, _] if y >= 2 => Some(Self::North),
            [_, y, _] if y <= -2 => Some(Self::South),
            [x, _, _] if x >= 2 => Some(Self::East),
            [x, _, _] if x <= -2 => Some(Self::West),
            _ => None,
        }
    }

    /// The direction of the V pair from `slot` to the child `child` under merge and shift,
    /// as FMM3D's lists assign it: if the slot lies at least two boxes from every child
    /// of the parent along z, then y, then x (its coordinate there is 0 or 5), that axis,
    /// so that one shift serves all eight children; otherwise [`Direction::of`] the
    /// offset. `None` if the two boxes are adjacent (no V pair). Either way the offset
    /// along the axis is 2 or 3 boxes, the region of the quadrature.
    pub fn for_slot(slot: usize, child: usize) -> Option<Self> {
        let a = slot_coordinates(slot);
        let b = child_bits(child);
        let d: [i64; 3] = core::array::from_fn(|i| b[i] + 2 - a[i]);
        if d.iter().all(|x| x.abs() <= 1) {
            return None;
        }
        for (axis, (up, down)) in [
            (2, (Self::Up, Self::Down)),
            (1, (Self::North, Self::South)),
            (0, (Self::East, Self::West)),
        ] {
            match a[axis] {
                0 => return Some(up),
                5 => return Some(down),
                _ => {}
            }
        }
        Self::of(d)
    }

    /// The index in [`Direction::ALL`].
    pub fn index(self) -> usize {
        self as usize
    }

    /// The frame of the direction: 0 for ±z, 1 for ±y, 2 for ±x.
    pub fn frame(self) -> usize {
        self.index() / 2
    }

    /// Whether the direction is the positive one of its frame (the target above the
    /// source after the frame's rotation): the up waves of the frame.
    pub fn is_up(self) -> bool {
        self.index().is_multiple_of(2)
    }

    /// The vector `d` in the frame of the direction, (lateral, lateral, along): the
    /// components of Q d for the frame's rotation Q ([`FRAMES`](super::FRAMES)): d for
    /// ±z, (x, −z, y) for ±y, (−z, y, x) for ±x.
    pub fn rotate<T: Copy + core::ops::Neg<Output = T>>(self, d: [T; 3]) -> [T; 3] {
        match self.frame() {
            0 => d,
            1 => [d[0], -d[2], d[1]],
            _ => [-d[2], d[1], d[0]],
        }
    }
}

/// One accumulator of a direction: the slots whose waves it sums, and the children it
/// serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    /// The children the accumulator serves, one bit per child index.
    pub mask: u8,
    /// Its slots, ascending: the slots whose pairs of this direction reach exactly the
    /// children of `mask`.
    pub slots: Vec<u8>,
}

/// The accumulators of a parent, the same for every parent (CONVENTIONS §3.14, "Merge and
/// shift"): for each direction, the slots grouped by the set of children they reach in
/// that direction ([`Direction::for_slot`]), one group per distinct set, the groups by
/// ascending mask.
///
/// The masks are geometric: they count all eight children, whether or not a tree holds
/// them, so an accumulator sums the same slots on every rank and in every tree; a slot
/// that a tree does not hold contributes nothing, and a child that a rank does not
/// compute receives nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeGeometry {
    /// Per slot and child, the direction index of the pair, or [`MergeGeometry::NONE`].
    directions: Vec<[u8; CHILDREN]>,
    /// Per direction, its accumulators by ascending mask.
    accumulators: [Vec<Accumulator>; 6],
}

impl MergeGeometry {
    /// No direction: the slot and the child are adjacent.
    pub const NONE: u8 = u8::MAX;

    /// The geometry, computed from [`Direction::for_slot`].
    pub fn new() -> Self {
        let directions: Vec<[u8; CHILDREN]> = (0..SLOTS)
            .map(|slot| {
                core::array::from_fn(|child| {
                    Direction::for_slot(slot, child).map_or(Self::NONE, |d| d.index() as u8)
                })
            })
            .collect();
        let accumulators = core::array::from_fn(|dir| {
            let mut masks: Vec<(u8, u8)> = (0..SLOTS)
                .filter_map(|slot| {
                    let mask = (0..CHILDREN)
                        .filter(|&c| directions[slot][c] == dir as u8)
                        .fold(0u8, |mask, c| mask | (1 << c));
                    (mask != 0).then_some((mask, slot as u8))
                })
                .collect();
            masks.sort_unstable();
            let mut accumulators: Vec<Accumulator> = Vec::new();
            for (mask, slot) in masks {
                match accumulators.last_mut() {
                    Some(last) if last.mask == mask => last.slots.push(slot),
                    _ => accumulators.push(Accumulator {
                        mask,
                        slots: vec![slot],
                    }),
                }
            }
            accumulators
        });
        Self {
            directions,
            accumulators,
        }
    }

    /// The direction of the pair from `slot` to `child`, as [`Direction::for_slot`].
    pub fn direction(&self, slot: usize, child: usize) -> Option<Direction> {
        let index = self.directions[slot][child];
        (index != Self::NONE).then(|| Direction::ALL[index as usize])
    }

    /// The accumulators of `direction`, by ascending mask.
    pub fn accumulators(&self, direction: Direction) -> &[Accumulator] {
        &self.accumulators[direction.index()]
    }

    /// The largest number of accumulators of a direction.
    pub fn max_accumulators(&self) -> usize {
        self.accumulators.iter().map(Vec::len).max().unwrap_or(0)
    }
}

impl Default for MergeGeometry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every V pair of a child has exactly one slot and one direction, and its offset
    /// along the direction is 2 or 3; the 189 pairs of an interior child.
    #[test]
    fn every_pair_has_one_direction() {
        let geometry = MergeGeometry::new();
        for child in 0..CHILDREN {
            let mut pairs = 0;
            for slot in 0..SLOTS {
                let a = slot_coordinates(slot);
                let b = child_bits(child);
                let d: [i64; 3] = core::array::from_fn(|i| b[i] + 2 - a[i]);
                assert_eq!(slot_of(child, d), Some(slot));
                let v_pair = d.iter().any(|x| x.abs() >= 2);
                let direction = geometry.direction(slot, child);
                assert_eq!(direction.is_some(), v_pair, "slot {slot}, child {child}");
                if let Some(direction) = direction {
                    pairs += 1;
                    let along = direction.rotate(d)[2];
                    let expected = if direction.is_up() { 2..=3 } else { -3..=-2 };
                    assert!(expected.contains(&along), "{d:?} along {direction:?}");
                    let accumulators = geometry.accumulators(direction);
                    let serving: Vec<&Accumulator> = accumulators
                        .iter()
                        .filter(|acc| acc.slots.contains(&(slot as u8)))
                        .collect();
                    assert_eq!(serving.len(), 1);
                    assert!(serving[0].mask & (1 << child) != 0);
                }
            }
            assert_eq!(pairs, 189, "child {child}");
        }
    }

    /// An interior parent: 248 (slot, direction) entries, 152 of them serving all eight
    /// children (Phase 6 T10's count, `spikes/plane-wave/src/geometry.rs`).
    #[test]
    fn an_interior_parent_merges_248_entries() {
        let geometry = MergeGeometry::new();
        let entries: usize = Direction::ALL
            .iter()
            .flat_map(|&d| geometry.accumulators(d))
            .map(|acc| acc.slots.len())
            .sum();
        assert_eq!(entries, 248);
        let all: usize = Direction::ALL
            .iter()
            .flat_map(|&d| geometry.accumulators(d))
            .filter(|acc| acc.mask == 0xff)
            .map(|acc| acc.slots.len())
            .sum();
        assert_eq!(all, 152);
        for direction in Direction::ALL {
            let masks: Vec<u8> = geometry
                .accumulators(direction)
                .iter()
                .map(|acc| acc.mask)
                .collect();
            assert!(masks.windows(2).all(|w| w[0] < w[1]), "{direction:?}");
        }
    }

    #[test]
    fn rotations_take_each_direction_to_its_axis() {
        for direction in Direction::ALL {
            let unit: [i64; 3] = match direction {
                Direction::Up => [0, 0, 1],
                Direction::Down => [0, 0, -1],
                Direction::North => [0, 1, 0],
                Direction::South => [0, -1, 0],
                Direction::East => [1, 0, 0],
                Direction::West => [-1, 0, 0],
            };
            let along = if direction.is_up() { 1 } else { -1 };
            assert_eq!(direction.rotate(unit), [0, 0, along], "{direction:?}");
            assert_eq!(Direction::of(unit.map(|x| 2 * x)), Some(direction));
        }
    }
}
