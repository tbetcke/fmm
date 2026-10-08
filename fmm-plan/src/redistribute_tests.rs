//! Pure local tests of the redistribution: bucketing, in-place permutations, count
//! arithmetic, and the grouping, order and round trip of `Sender` and `Receiver` on
//! hand-made routing data, with the all-to-all-v simulated. No MPI.
use super::{Cycles, Receiver, Sender, bucket, max_per_item, value_counts};

/// A small deterministic generator (SplitMix64), so that the tests need no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A random permutation of `0..n`, as slots.
    fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut slots: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            slots.swap(i, self.below(i + 1));
        }
        slots
    }
}

#[test]
fn buckets_count_and_keep_the_element_order() {
    let (counts, slots) = bucket(&[2, 0, 2, 1, 0], 4);
    assert_eq!(counts, [2, 1, 2, 0]);
    // Bucket starts 0, 2, 3, 5; each bucket in element order.
    assert_eq!(slots, [3, 0, 4, 2, 1]);

    let (counts, slots) = bucket(&[], 3);
    assert_eq!(counts, [0, 0, 0]);
    assert!(slots.is_empty());
}

#[test]
fn cycles_move_chunks_as_the_permutation_says() {
    let mut rng = Rng(7);
    for n in 0..40 {
        let slots = rng.permutation(n);
        let cycles = Cycles::new(&slots);
        let fixed = (0..n).filter(|&a| slots[a] as usize == a).count();
        assert_eq!(
            cycles.elements.len(),
            n - fixed,
            "fixed points are left out"
        );
        for size in [1, 3] {
            let values: Vec<usize> = (0..n * size).collect();
            let mut forward = vec![0; n * size];
            for a in 0..n {
                let s = slots[a] as usize;
                forward[s * size..(s + 1) * size]
                    .copy_from_slice(&values[a * size..(a + 1) * size]);
            }
            let mut applied = values.clone();
            cycles.apply(&mut applied, size);
            assert_eq!(applied, forward, "n = {n}, size = {size}");

            let mut inverse = vec![0; n * size];
            for a in 0..n {
                let s = slots[a] as usize;
                inverse[a * size..(a + 1) * size]
                    .copy_from_slice(&values[s * size..(s + 1) * size]);
            }
            let mut unapplied = values.clone();
            cycles.apply_inverse(&mut unapplied, size);
            assert_eq!(unapplied, inverse, "n = {n}, size = {size}");

            cycles.apply_inverse(&mut applied, size);
            assert_eq!(applied, values, "the inverse undoes the permutation");
        }
    }
    let identity: Vec<u32> = (0..10).collect();
    assert_eq!(Cycles::new(&identity), Cycles::new(&[]), "no cycles");
}

#[test]
fn value_counts_scale_items_and_prefix_sum_them() {
    let mut values = [0i32; 3];
    let mut displacements = [0i32; 3];
    value_counts(&[2, 0, 3], 3, &mut values, &mut displacements);
    assert_eq!(values, [6, 0, 9]);
    assert_eq!(displacements, [0, 6, 6]);
    value_counts(&[2, 0, 3], 0, &mut values, &mut displacements);
    assert_eq!(values, [0, 0, 0]);
    assert_eq!(displacements, [0, 0, 0]);
}

#[test]
fn max_per_item_keeps_every_count_in_i32() {
    let limit = i32::MAX as usize;
    assert_eq!(max_per_item(0, 0), limit);
    assert_eq!(max_per_item(10, 3), limit / 10);
    assert_eq!(max_per_item(3, 10), limit / 10);
    assert_eq!(max_per_item(limit, 1), 1);
    assert_eq!(max_per_item(limit + 1, 0), 0);
    for (nsent, nreceived) in [(1, 1), (7, 1000), (123_456, 99), (limit / 3, 2)] {
        let k = max_per_item(nsent, nreceived);
        assert!(nsent.max(nreceived) * k <= limit);
        assert!(nsent.max(nreceived) * (k + 1) > limit);
    }
}

/// One rank's items of a simulated redistribution: (destination rank, leaf there).
type Items = Vec<(u32, u32)>;

/// Simulate an all-to-all-v: rank r sends `sends[r]` (chunks of `size` values, grouped
/// by destination with `counts[r][d]` chunks for rank d), and rank d receives the chunks
/// of every origin in rank order.
fn all_to_all_v<T: Copy>(sends: &[Vec<T>], counts: &[Vec<usize>], size: usize) -> Vec<Vec<T>> {
    let nranks = sends.len();
    let mut received = vec![Vec::new(); nranks];
    for (send, counts) in sends.iter().zip(counts) {
        let mut start = 0;
        for (d, &count) in counts.iter().enumerate() {
            received[d].extend_from_slice(&send[start..start + count * size]);
            start += count * size;
        }
        assert_eq!(start, send.len());
    }
    received
}

/// Route every rank's `items` as `Redistribution::new` does, with the exchange of
/// (position, leaf) simulated, and check the grouping by leaf, the order within a leaf,
/// the counts, the origins, and that `forward` then `backward` of payloads that name
/// their origin is the identity.
fn check_simulated(items: &[Items], nleaves: &[usize]) {
    let nranks = items.len();
    let senders: Vec<Sender> = items
        .iter()
        .map(|items| {
            let destinations: Vec<u32> = items.iter().map(|&(d, _)| d).collect();
            Sender::new(&destinations, nranks)
        })
        .collect();
    let send_counts: Vec<Vec<usize>> = senders.iter().map(|s| s.rank_counts.clone()).collect();

    // The routing data: position and leaf of every item, packed by destination.
    let routed: Vec<Vec<u32>> = items
        .iter()
        .zip(&senders)
        .map(|(items, sender)| {
            let mut values: Vec<u32> = items
                .iter()
                .enumerate()
                .flat_map(|(position, &(_, leaf))| [position as u32, leaf])
                .collect();
            sender.pack(&mut values, 2);
            values
        })
        .collect();
    let arrived = all_to_all_v(&routed, &send_counts, 2);
    let receivers: Vec<Receiver> = (0..nranks)
        .map(|d| {
            let counts = (0..nranks).map(|r| send_counts[r][d]).collect();
            let positions: Vec<u32> = arrived[d].chunks(2).map(|c| c[0]).collect();
            let leaves: Vec<u32> = arrived[d].chunks(2).map(|c| c[1]).collect();
            Receiver::new(counts, &positions, &leaves, nleaves[d])
        })
        .collect();

    // The expected received order: by leaf, then by (origin rank, position).
    for (d, receiver) in receivers.iter().enumerate() {
        let mut expected: Vec<(u32, u32, u32)> = items
            .iter()
            .enumerate()
            .flat_map(|(r, items)| {
                items
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(dest, _))| dest as usize == d)
                    .map(move |(p, &(_, leaf))| (leaf, r as u32, p as u32))
            })
            .collect();
        expected.sort_unstable();
        let origins: Vec<(u32, u32)> = expected.iter().map(|&(_, r, p)| (r, p)).collect();
        assert_eq!(receiver.origins, origins, "rank {d}: origins");
        let mut counts = vec![0; nleaves[d]];
        for &(leaf, _, _) in &expected {
            counts[leaf as usize] += 1;
        }
        assert_eq!(receiver.leaf_counts, counts, "rank {d}: counts per leaf");
        assert_eq!(receiver.len(), expected.len());
    }

    // forward: three values per item naming its origin.
    let payload = |r: usize, p: usize| [r as u64, p as u64, (r * 1000 + p) as u64];
    let sends: Vec<Vec<u64>> = items
        .iter()
        .zip(&senders)
        .enumerate()
        .map(|(r, (items, sender))| {
            let mut values: Vec<u64> = (0..items.len()).flat_map(|p| payload(r, p)).collect();
            sender.pack(&mut values, 3);
            values
        })
        .collect();
    let mut received = all_to_all_v(&sends, &send_counts, 3);
    for (d, (receiver, received)) in receivers.iter().zip(&mut received).enumerate() {
        receiver.unpack(received, 3);
        let expected: Vec<u64> = receiver
            .origins
            .iter()
            .flat_map(|&(r, p)| payload(r as usize, p as usize))
            .collect();
        assert_eq!(*received, expected, "rank {d}: forwarded payload");
    }

    // backward: the received payloads back to their origins, in item order.
    let receive_counts: Vec<Vec<usize>> = receivers.iter().map(|r| r.rank_counts.clone()).collect();
    for (receiver, received) in receivers.iter().zip(&mut received) {
        receiver.pack(received, 3);
    }
    let mut returned = all_to_all_v(&received, &receive_counts, 3);
    for (r, (sender, returned)) in senders.iter().zip(&mut returned).enumerate() {
        sender.unpack(returned, 3);
        let expected: Vec<u64> = (0..items[r].len()).flat_map(|p| payload(r, p)).collect();
        assert_eq!(*returned, expected, "rank {r}: forward then backward");
        assert_eq!(sender.len(), items[r].len());
    }
}

#[test]
fn simulated_ranks_group_by_leaf_in_origin_order_and_round_trip() {
    // Hand-made: three ranks, items out of order, duplicates of a leaf on several
    // ranks, an empty rank, and a leaf that receives nothing.
    let items = vec![
        vec![(1, 0), (0, 2), (1, 1), (1, 0), (2, 3), (0, 0)],
        vec![],
        vec![(0, 2), (1, 0), (0, 2), (2, 0), (2, 3), (0, 1)],
    ];
    check_simulated(&items, &[3, 2, 4]);

    // One rank: the received order is the caller's order, stably grouped by leaf.
    check_simulated(&[vec![(0, 1), (0, 0), (0, 1), (0, 0)]], &[2]);
    check_simulated(&[vec![]], &[0]);

    // Random routing data on 1 to 5 ranks.
    let mut rng = Rng(11);
    for nranks in 1..=5 {
        let nleaves: Vec<usize> = (0..nranks).map(|_| rng.below(6)).collect();
        let owners: Vec<u32> = (0..nranks as u32)
            .filter(|&d| nleaves[d as usize] > 0)
            .collect();
        let items: Vec<Items> = (0..nranks)
            .map(|_| {
                let n = if owners.is_empty() { 0 } else { rng.below(30) };
                (0..n)
                    .map(|_| {
                        let d = owners[rng.below(owners.len())];
                        (d, rng.below(nleaves[d as usize]) as u32)
                    })
                    .collect()
            })
            .collect();
        check_simulated(&items, &nleaves);
    }
}
