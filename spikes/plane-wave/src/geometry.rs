//! A uniform octree level: boxes in Morton order, V lists (CONVENTIONS §3.12), the six
//! directions of plane-wave M2L, and the merge-and-shift groups of each parent.
//!
//! Coordinates are in box widths of the level: box (i, j, k) has its centre at
//! (i, j, k) + ½. A V-list pair has the offset d = index(target) − index(source)
//! (§3.12), with max |dᵢ| ≥ 2 and |dᵢ| ≤ 3.

/// The six directions: the axis along which the target lies at least two boxes from
/// the source, and its sign. [`Direction::for_pair`] gives every V pair exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// d_z ≥ 2: the target above the source.
    Up,
    /// d_z ≤ −2.
    Down,
    /// |d_z| ≤ 1, d_y ≥ 2.
    North,
    /// |d_z| ≤ 1, d_y ≤ −2.
    South,
    /// |d_z|, |d_y| ≤ 1, d_x ≥ 2.
    East,
    /// |d_z|, |d_y| ≤ 1, d_x ≤ −2.
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

    /// The direction of the V-list offset `d` (target − source), or `None` if no
    /// component has |dᵢ| ≥ 2.
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

    /// The direction of the V pair from the source at `s` to the child `c` of the
    /// parent at `parent` (box coordinates; `c` one of [`children`]`(parent)`), as
    /// FMM3D's lists assign it: if the source lies at least two boxes from every child
    /// of the parent along z, then y, then x, that axis (so one shift serves all eight
    /// children); otherwise the first axis of z, y, x along which it lies two boxes from
    /// `c` ([`Direction::of`]). Either way |dᵢ| ∈ {2, 3} along the axis, the region of
    /// the quadrature.
    ///
    /// # Panics
    ///
    /// If the pair is not a V pair.
    pub fn for_pair(parent: [i64; 3], s: [i64; 3], c: [i64; 3]) -> Self {
        let below = |axis: usize| s[axis] <= 2 * parent[axis] - 2;
        let above = |axis: usize| s[axis] >= 2 * parent[axis] + 3;
        for (axis, (up, down)) in [
            (2, (Self::Up, Self::Down)),
            (1, (Self::North, Self::South)),
            (0, (Self::East, Self::West)),
        ] {
            // The target above the source: the source below every child.
            if below(axis) {
                return up;
            }
            if above(axis) {
                return down;
            }
        }
        Self::of([c[0] - s[0], c[1] - s[1], c[2] - s[2]]).expect("a V pair")
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
    /// source after the frame's rotation).
    pub fn is_up(self) -> bool {
        self.index().is_multiple_of(2)
    }

    /// The offset `d` in the frame of the direction: (lateral x, lateral y, along),
    /// the components of Q d for the frame's rotation Q (`planewave::FRAMES`):
    /// identity for ±z, (x, −z, y) for ±y, (−z, y, x) for ±x.
    pub fn rotate<T: Copy + core::ops::Neg<Output = T>>(self, d: [T; 3]) -> [T; 3] {
        match self.frame() {
            0 => d,
            1 => [d[0], -d[2], d[1]],
            _ => [-d[2], d[1], d[0]],
        }
    }
}

/// Interleaves the bits of (i, j, k) into a Morton index, i the most significant of
/// each triple.
pub fn morton(i: usize, j: usize, k: usize) -> usize {
    let mut out = 0;
    for bit in 0..21 {
        out |= ((i >> bit) & 1) << (3 * bit + 2);
        out |= ((j >> bit) & 1) << (3 * bit + 1);
        out |= ((k >> bit) & 1) << (3 * bit);
    }
    out
}

/// The coordinates of a Morton index.
pub fn demorton(m: usize) -> [usize; 3] {
    let mut c = [0; 3];
    for bit in 0..21 {
        c[0] |= ((m >> (3 * bit + 2)) & 1) << bit;
        c[1] |= ((m >> (3 * bit + 1)) & 1) << bit;
        c[2] |= ((m >> (3 * bit)) & 1) << bit;
    }
    c
}

/// A uniform level of 2^level boxes per side, numbered in Morton order.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// The level; 2^level boxes per side.
    pub level: u32,
}

/// One V-list pair.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    /// The target box.
    pub target: usize,
    /// The source box.
    pub source: usize,
    /// The offset index of CONVENTIONS §3.12.
    pub offset: usize,
    /// The offset d = target − source.
    pub d: [i64; 3],
}

/// The table index of a V-list offset (CONVENTIONS §3.12, "V-list offsets").
pub fn offset_index(d: [i64; 3]) -> usize {
    nd_fmm_tables::geometry::m2l_offset_index(d).expect("a V-list offset")
}

impl Level {
    /// Boxes per side.
    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// Boxes on the level.
    pub fn boxes(&self) -> usize {
        self.side().pow(3)
    }

    /// The coordinates of box `b`.
    pub fn coords(&self, b: usize) -> [i64; 3] {
        demorton(b).map(|c| c as i64)
    }

    /// The box at `c`, if inside the level.
    pub fn index(&self, c: [i64; 3]) -> Option<usize> {
        let n = self.side() as i64;
        c.iter()
            .all(|&x| (0..n).contains(&x))
            .then(|| morton(c[0] as usize, c[1] as usize, c[2] as usize))
    }

    /// Every V-list pair, by target in Morton order, each target's sources in Morton
    /// order.
    pub fn pairs(&self) -> Vec<Pair> {
        let mut pairs = Vec::new();
        for t in 0..self.boxes() {
            let ct = self.coords(t);
            let pt = ct.map(|x| x.div_euclid(2));
            let mut row = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let colleague = [pt[0] + dx, pt[1] + dy, pt[2] + dz];
                        for o in 0..8 {
                            let cs = [
                                2 * colleague[0] + (o >> 2) as i64,
                                2 * colleague[1] + ((o >> 1) & 1) as i64,
                                2 * colleague[2] + (o & 1) as i64,
                            ];
                            let Some(s) = self.index(cs) else { continue };
                            let d = [ct[0] - cs[0], ct[1] - cs[1], ct[2] - cs[2]];
                            if d.iter().any(|x| x.abs() >= 2) {
                                row.push(Pair {
                                    target: t,
                                    source: s,
                                    offset: offset_index(d),
                                    d,
                                });
                            }
                        }
                    }
                }
            }
            row.sort_by_key(|pair| pair.source);
            pairs.extend(row);
        }
        pairs
    }
}

/// The merge-and-shift work of one parent: for each direction the sources that reach
/// at least one of its children in that direction, each with the set of children it
/// reaches (a bit mask), and the distinct masks (each one accumulator).
#[derive(Clone, Debug, Default)]
pub struct ParentGroups {
    /// Per direction: (source box, child mask), sources in Morton order.
    pub sources: [Vec<(usize, u8)>; 6],
}

impl ParentGroups {
    /// The groups of parent `parent` (a box of the level above `level`).
    pub fn of(level: &Level, parent: [i64; 3]) -> Self {
        let mut groups = Self::default();
        let mut masks = [[0u8; 216]; 6];
        let mut boxes = [usize::MAX; 216];
        for (c, child) in children(parent).iter().enumerate() {
            for (slot, cs) in candidates(parent).iter().enumerate() {
                let Some(s) = level.index(*cs) else { continue };
                let d = [child[0] - cs[0], child[1] - cs[1], child[2] - cs[2]];
                if d.iter().any(|x| x.abs() >= 2) {
                    let dir = Direction::for_pair(parent, *cs, *child);
                    masks[dir.index()][slot] |= 1 << c;
                    boxes[slot] = s;
                }
            }
        }
        for (sources, masks) in groups.sources.iter_mut().zip(&masks) {
            let mut list: Vec<(usize, u8)> = (0..216)
                .filter(|&slot| masks[slot] != 0)
                .map(|slot| (boxes[slot], masks[slot]))
                .collect();
            list.sort_by_key(|&(s, _)| s);
            *sources = list;
        }
        groups
    }

    /// The distinct child masks of direction `dir`, ascending.
    pub fn masks(&self, dir: usize) -> Vec<u8> {
        let mut masks: Vec<u8> = self.sources[dir].iter().map(|&(_, m)| m).collect();
        masks.sort_unstable();
        masks.dedup();
        masks
    }
}

/// The eight children of a parent at `parent`, in child-index order (the bits of
/// (i, j, k) as in [`morton`]).
pub fn children(parent: [i64; 3]) -> [[i64; 3]; 8] {
    core::array::from_fn(|o| {
        [
            2 * parent[0] + (o >> 2) as i64,
            2 * parent[1] + ((o >> 1) & 1) as i64,
            2 * parent[2] + (o & 1) as i64,
        ]
    })
}

/// The 216 children of the 27 colleagues of `parent` (some outside the level).
fn candidates(parent: [i64; 3]) -> [[i64; 3]; 216] {
    core::array::from_fn(|slot| {
        let (a, b, c) = (slot / 36, (slot / 6) % 6, slot % 6);
        [
            2 * parent[0] - 2 + a as i64,
            2 * parent[1] - 2 + b as i64,
            2 * parent[2] - 2 + c as i64,
        ]
    })
}

/// The counts of a level that the model needs, per box of the level.
#[derive(Clone, Copy, Debug)]
pub struct Counts {
    /// Boxes on the level.
    pub boxes: usize,
    /// V-list pairs.
    pub pairs: usize,
    /// Source shifts of merge-and-shift: Σ over parents and directions of the sources.
    pub shifts: usize,
    /// Accumulators: Σ over parents and directions of the distinct masks.
    pub accumulators: usize,
    /// Accumulator additions into children: Σ over accumulators of their children.
    pub child_adds: usize,
    /// Child shifts: Σ over children of the directions with an accumulator.
    pub child_shifts: usize,
}

impl Counts {
    /// Counts `level` (at least level 2).
    pub fn of(level: &Level) -> Self {
        let pairs = level.pairs().len();
        let half = (level.side() / 2) as i64;
        let (mut shifts, mut accumulators, mut child_adds, mut child_shifts) = (0, 0, 0, 0);
        for i in 0..half {
            for j in 0..half {
                for k in 0..half {
                    let groups = ParentGroups::of(level, [i, j, k]);
                    for dir in 0..6 {
                        shifts += groups.sources[dir].len();
                        let masks = groups.masks(dir);
                        accumulators += masks.len();
                        child_adds += masks.iter().map(|m| m.count_ones() as usize).sum::<usize>();
                        let union = masks.iter().fold(0u8, |a, &m| a | m);
                        child_shifts += union.count_ones() as usize;
                    }
                }
            }
        }
        Self {
            boxes: level.boxes(),
            pairs,
            shifts,
            accumulators,
            child_adds,
            child_shifts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_round_trips() {
        for m in [0, 1, 7, 8, 511, 4095, 123_456] {
            let [i, j, k] = demorton(m);
            assert_eq!(morton(i, j, k), m);
        }
    }

    #[test]
    fn interior_box_has_189_pairs_and_t1_level_4_has_584_136() {
        let level = Level { level: 4 };
        let pairs = level.pairs();
        assert_eq!(pairs.len(), 584_136);
        let interior = level.index([7, 7, 7]).unwrap();
        assert_eq!(pairs.iter().filter(|p| p.target == interior).count(), 189);
    }

    /// Every V pair of a child reaches it through exactly one (source, direction) entry
    /// of its parent's groups.
    #[test]
    fn groups_cover_every_pair_once() {
        let level = Level { level: 3 };
        let pairs = level.pairs();
        let mut covered = 0;
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let groups = ParentGroups::of(&level, [i, j, k]);
                    for (dir, list) in groups.sources.iter().enumerate() {
                        for &(s, mask) in list {
                            for (c, child) in children([i, j, k]).iter().enumerate() {
                                if mask & (1 << c) == 0 {
                                    continue;
                                }
                                let t = level.index(*child).unwrap();
                                let pair = pairs
                                    .iter()
                                    .find(|p| p.target == t && p.source == s)
                                    .expect("a V pair");
                                let cs = level.coords(s);
                                let dir_of_pair = Direction::for_pair([i, j, k], cs, *child);
                                assert_eq!(dir_of_pair.index(), dir);
                                let along = dir_of_pair.rotate(pair.d)[2].abs();
                                assert!(along == 2 || along == 3, "{:?}", pair.d);
                                covered += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(covered, pairs.len());
    }

    #[test]
    /// An interior parent: 208 sources reach its children; 152 lie two boxes from all
    /// eight along one axis (one shift each), the 56 others take 96 entries.
    fn interior_parent_merges_248_entries() {
        let level = Level { level: 4 };
        let groups = ParentGroups::of(&level, [3, 3, 3]);
        let total: usize = groups.sources.iter().map(Vec::len).sum();
        assert_eq!(total, 248);
        let all: usize = groups
            .sources
            .iter()
            .map(|list| list.iter().filter(|&&(_, m)| m == 0xff).count())
            .sum();
        assert_eq!(all, 152);
    }
}
