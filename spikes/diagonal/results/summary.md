# CH0.6 (S6) results

Written by `examples/summary.rs` from 3 rows files: 682 units (kind, k a, p), 185504 (unit, class, variant) points. Errors are the worst over the three placements of max(potential, gradient), relative ℓ2 over the targets. u = 2⁻⁵³ (f64), 2⁻²⁴ (f32).

## 1. Rounding against the bounds

measured / (u 10^term) (the proposal's E with C_T = 1) and measured / (u max|T|), each times the damping factor e^{Im(k)(|t| + 2a)} of §7 (1 for real k), quantiles 0, ½, 0.9, 1, over the points whose bound exceeds the floor (≥ 10⁻¹³ in f64, 10⁻⁵ in f32), and the largest measured error among the floor points:

| kind | variant | points | E: min / median / 90 % / max | max|T|: min / median / 90 % / max | floor points: max measured |
| --- | --- | --- | --- | --- | --- |
| complex0.1 | f64:2P-2 | 7216 | 1.3e-17 / 2.8e-4 / 8.0e-3 / 1.6e-1 | 6.1e-15 / 2.1e-2 / 2.5e-1 / 7.2e0 | 6.2e-15 |
| complex0.1 | f64:3(P-1)/2 | 7017 | 2.0e-16 / 3.9e-2 / 5.2e6 / 4.9e10 | 7.3e-14 / 4.2e0 / 3.5e8 / 7.3e11 | 2.6e-3 |
| complex0.1 | f64:P-1 | 6344 | 4.1e-6 / 1.8e6 / 2.6e10 / 2.7e12 | 1.1e-3 / 1.1e8 / 9.6e11 / 6.1e13 | 4.2e-1 |
| complex0.1 | f64i:2P-2 | 7216 | 1.2e-17 / 3.1e-4 / 1.1e-2 / 3.5e-1 | 5.9e-15 / 2.5e-2 / 3.3e-1 / 1.2e1 | 8.6e-15 |
| complex0.1 | f64b:2P-2 | 7216 | 4.5e-16 / 1.1e-2 / 3.6e-1 / 2.7e0 | 2.4e-13 / 7.5e-1 / 1.1e1 / 2.5e2 | 1.6e-13 |
| complex0.1 | f32:2P-2 | 6711 | 1.0e-17 / 1.5e-4 / 7.9e-3 / 1.9e-1 | 4.9e-15 / 1.5e-2 / 2.5e-1 / 7.9e0 | 7.0e-7 |
| complex0.1 | f32:3(P-1)/2 | 7060 | 9.3e-18 / 9.3e-4 / 4.9e-2 / 9.1e1 | 3.4e-15 / 4.2e-2 / 1.9e0 / 1.4e3 | 5.2e-4 |
| complex0.1 | f32:P-1 | 6997 | 7.6e-15 / 2.1e-2 / 1.7e2 / 1.4e4 | 2.0e-12 / 9.2e-1 / 4.8e3 / 1.8e5 | 1.5e-1 |
| complex1 | f64:2P-2 | 3298 | 1.8e-16 / 4.7e-7 / 4.5e-5 / 5.7e-4 | 1.8e-14 / 1.6e-5 / 9.5e-4 / 4.9e-2 | 0.0e0 |
| complex1 | f64:3(P-1)/2 | 3298 | 2.8e-16 / 5.8e-6 / 8.8e3 / 5.4e9 | 2.7e-14 / 1.3e-4 / 1.5e5 / 4.6e10 | 0.0e0 |
| complex1 | f64:P-1 | 3298 | 4.0e-16 / 2.5e-2 / 9.1e7 / 4.1e11 | 2.8e-14 / 4.7e-1 / 9.4e8 / 3.0e12 | 0.0e0 |
| complex1 | f64i:2P-2 | 3298 | 2.4e-16 / 6.1e-7 / 5.9e-5 / 6.3e-4 | 2.4e-14 / 2.0e-5 / 1.3e-3 / 5.5e-2 | 0.0e0 |
| complex1 | f64b:2P-2 | 3298 | 9.3e-15 / 1.1e-5 / 1.4e-3 / 1.1e-2 | 6.0e-13 / 3.7e-4 / 3.0e-2 / 1.4e0 | 0.0e0 |
| complex1 | f32:2P-2 | 2474 | 5.1e-16 / 5.2e-7 / 5.1e-5 / 5.5e-4 | 3.1e-14 / 1.8e-5 / 1.0e-3 / 3.1e-2 | 0.0e0 |
| complex1 | f32:3(P-1)/2 | 2946 | 2.0e-16 / 7.6e-7 / 1.9e-4 / 1.0e1 | 1.7e-14 / 2.1e-5 / 3.5e-3 / 8.6e1 | 0.0e0 |
| complex1 | f32:P-1 | 3281 | 3.0e-16 / 4.1e-6 / 1.7e-1 / 7.7e2 | 2.3e-14 / 7.8e-5 / 1.8e0 / 5.6e3 | 0.0e0 |
| real | f64:2P-2 | 7184 | 3.5e-4 / 1.1e-2 / 2.0e-2 / 3.1e-1 | 1.4e-2 / 1.1e0 / 8.3e0 / 2.8e1 | 2.0e-14 |
| real | f64:3(P-1)/2 | 4647 | 2.2e-3 / 3.3e2 / 1.6e12 / 3.6e12 | 2.0e-2 / 1.7e4 / 6.7e14 / 1.6e15 | 3.5e-1 |
| real | f64:P-1 | 1958 | 3.3e-3 / 4.6e1 / 6.1e12 / 7.3e12 | 2.4e-2 / 1.0e3 / 1.9e15 / 2.7e15 | 1.0e0 |
| real | f64i:2P-2 | 7184 | 3.5e-4 / 1.2e-2 / 2.7e-2 / 4.5e-1 | 1.6e-2 / 1.3e0 / 8.5e0 / 3.6e1 | 1.8e-14 |
| real | f64b:2P-2 | 7184 | 2.3e-2 / 5.5e-1 / 1.4e0 / 7.2e0 | 1.7e-1 / 7.2e1 / 5.8e2 / 5.1e3 | 3.3e-13 |
| real | f32:2P-2 | 11429 | 3.2e-4 / 1.5e-2 / 3.7e-2 / 3.5e-1 | 4.3e-2 / 3.5e0 / 7.4e0 / 2.6e1 | 2.6e-6 |
| real | f32:3(P-1)/2 | 10269 | 2.0e-3 / 5.8e0 / 3.3e3 / 1.3e4 | 3.6e-2 / 5.9e2 / 1.0e6 / 3.0e6 | 2.2e-2 |
| real | f32:P-1 | 7349 | 3.5e-3 / 1.5e4 / 3.5e4 / 7.3e4 | 3.1e-2 / 2.6e6 / 3.9e6 / 5.4e6 | 1.0e0 |

## 2. C_T and θ (f64, L = 2P − 2, T rounded from double-double)

Above the floor the ratio measured / (u 10^term e^{Im(k)(|t| + 2a)}) lies in [1.3e-17, 3.1e-1] (median 2.6e-3); the largest at real k a 2.828 p 7 class [0, 0, 2]; the largest per kind: complex0.1 1.65e-1, complex1 5.65e-4, real 3.05e-1. With C_T = 1.53e-1 (half the largest ratio) the bound E = C_T u 10^term e^{Im(k)(|t| + 2a)} never under-predicts by more than 2×, and θ = 1/2 makes E ≤ θ ε imply a measured rounding error ≤ ε. Below the floor (u 10^term e^{…} < 10⁻¹³) the rounding error is at most 176 u, so the bound with the floor is E = u (C_T 10^term e^{…} + 176).


## 3. complex0.1, formula: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | 4,5 / 4 (16 %) [13] | 22 classes / 17 classes (72 %) [16] | 34 classes / 33 classes (100 %) [19] | 34 classes / 34 classes (100 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [29] | 34 classes / 34 classes (100 %) [32] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | none / none (0 %) [14] | 4,5,6,8 / 4,5,6 (35 %) [17] | 27 classes / 23 classes (92 %) [20] | 34 classes / 33 classes (100 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | none / none (0 %) [15] | 4,5 / 4 (16 %) [18] | 4,5,6,8,9,10 / 4,5,6,8,9 (56 %) [21] | 29 classes / 25 classes (93 %) [24] | 34 classes / 34 classes (100 %) [28] | 34 classes / 34 classes (100 %) [31] | 34 classes / 34 classes (100 %) [34] | 34 classes / 34 classes (100 %) [37] |
| 8.000 | none / none (0 %) [18] | 4 / none (3 %) [21] | 4,5,6,8 / 4,5,6 (35 %) [24] | 22 classes / 4,5,6,8,9,10 (72 %) [27] | 31 classes / 25 classes (96 %) [30] | 34 classes / 33 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] | 34 classes / 34 classes (100 %) [39] |
| 11.314 | none / none (0 %) [22] | 4 / none (3 %) [25] | 4,5,6 / 4 (29 %) [27] | 4,5,6,8,9 / 4,5,6 (49 %) [30] | 22 classes / 4,5,6,8,9,10 (72 %) [33] | 31 classes / 25 classes (96 %) [36] | 34 classes / 31 classes (100 %) [39] | 34 classes / 34 classes (100 %) [42] |
| 16.000 | none / none (0 %) [28] | 4 / none (3 %) [31] | 4,5 / 4 (16 %) [33] | 4,5,6,8 / 4,5,6 (35 %) [36] | 18 classes / 4,5,6,8 (62 %) [39] | 25 classes / 4,5,6,8,9,10 (85 %) [41] | 31 classes / 23 classes (96 %) [44] | 34 classes / 29 classes (100 %) [47] |
| 22.627 | none / none (0 %) [36] | 4 / none (3 %) [39] | 4,5 / 4 (16 %) [41] | 4,5,6 / 4,5 (29 %) [44] | 4,5,6,8,9 / 4,5,6 (49 %) [47] | 19 classes / 4,5,6,8 (66 %) [49] | 29 classes / 4,5,6,8,9,10 (93 %) [52] | 34 classes / 23 classes (100 %) [55] |
| 32.000 | none / none (0 %) [47] | 4 / none (3 %) [50] | 4,5 / 4 (16 %) [53] | 4,5,6 / 4,5 (29 %) [56] | 4,5,6,8,9,10 / 4,5,6 (56 %) [59] | 22 classes / 4,5,6 (72 %) [61] | 33 classes / 4,5,6,8,9 (99 %) [64] | 34 classes / 18 classes (100 %) [67] |
| 45.255 | 4 / none (3 %) [62] | 4,5 / none (16 %) [65] | 4,5,6 / 4 (29 %) [69] | 4,5,6,8,9 / 4,5 (49 %) [72] | 25 classes / 4,5,6 (85 %) [75] | 34 classes / 4,5,6 (100 %) [78] | 34 classes / 4,5,6,8,9 (100 %) [81] | 34 classes / 4,5,6,8,9,10 (100 %) [83] |
| 64.000 | 4,5 / none (16 %) [83] | 4,5,6 / none (29 %) [87] | 18 classes / 4 (62 %) [90] | 33 classes / 4,5 (99 %) [94] | 34 classes / 4,5,6 (100 %) [97] | 34 classes / 4,5,6 (100 %) [100] | 34 classes / 4,5,6,8,9 (100 %) [103] | 34 classes / 17 classes (100 %) [106] |
| 90.510 | – | 34 classes / 4 (100 %) [116] | 34 classes / 4,5 (100 %) [120] | 34 classes / 4,5 (100 %) [124] | 34 classes / 4,5,6,9 (100 %) [128] | 34 classes / 4,5,6,8,9 (100 %) [131] | 34 classes / 18 classes (100 %) [135] | 34 classes / 28 classes (100 %) [138] |
| 128.000 | – | 34 classes / 4,5 (100 %) [157] | 34 classes / 4,5,6,9 (100 %) [161] | – | 34 classes / 15 classes (100 %) [170] | 34 classes / 25 classes (100 %) [174] | 34 classes / 33 classes (100 %) [177] | 34 classes / 34 classes (100 %) [181] |
| 181.019 | – | – | 34 classes / 18 classes (100 %) [218] | 34 classes / 26 classes (100 %) [223] | – | – | – | 34 classes / 34 classes (100 %) [241] |

## 3. complex0.1, S7: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | none / none (0 %) [12] | 31 classes / 26 classes (96 %) [18] | 34 classes / 34 classes (100 %) [23] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [35] | 34 classes / 34 classes (100 %) [41] | 34 classes / 34 classes (100 %) [48] | 34 classes / 34 classes (100 %) [55] |
| 4.000 | none / none (0 %) [12] | 4,5,6,8 / 4,5,6 (35 %) [17] | 33 classes / 30 classes (99 %) [22] | 34 classes / 34 classes (100 %) [27] | 34 classes / 34 classes (100 %) [34] | 34 classes / 34 classes (100 %) [40] | 34 classes / 34 classes (100 %) [46] | 34 classes / 34 classes (100 %) [54] |
| 5.657 | none / none (0 %) [13] | 4,5 / 4 (16 %) [18] | 25 classes / 18 classes (85 %) [23] | 34 classes / 34 classes (100 %) [29] | 34 classes / 34 classes (100 %) [35] | 34 classes / 34 classes (100 %) [41] | 34 classes / 34 classes (100 %) [48] | 34 classes / 34 classes (100 %) [54] |
| 8.000 | none / none (0 %) [16] | none / none (0 %) [19] | 4,5,6,8 / 4,5,6 (35 %) [24] | 29 classes / 25 classes (93 %) [30] | 34 classes / 33 classes (100 %) [36] | 34 classes / 34 classes (100 %) [42] | 34 classes / 34 classes (100 %) [50] | 34 classes / 34 classes (100 %) [55] |
| 11.314 | none / none (0 %) [19] | none / none (0 %) [22] | 4,5 / 4 (16 %) [26] | 4,5,6,8,9,10 / 4,5,6,8 (56 %) [31] | 29 classes / 25 classes (93 %) [37] | 34 classes / 33 classes (100 %) [43] | 34 classes / 34 classes (100 %) [50] | 34 classes / 34 classes (100 %) [56] |
| 16.000 | none / none (0 %) [24] | none / none (0 %) [28] | 4 / none (3 %) [30] | 4,5,6 / 4,5 (29 %) [34] | 19 classes / 4,5,6,8,9 (66 %) [41] | 31 classes / 25 classes (96 %) [48] | 34 classes / 31 classes (100 %) [52] | 34 classes / 34 classes (100 %) [59] |
| 22.627 | none / none (0 %) [33] | none / none (0 %) [35] | 4 / none (3 %) [37] | 4,5 / 4 (16 %) [40] | 4,5,6 / 4,5 (29 %) [43] | 22 classes / 4,5,6,8,9 (72 %) [51] | 33 classes / 25 classes (99 %) [58] | 34 classes / 30 classes (100 %) [62] |
| 32.000 | none / none (0 %) [44] | none / none (0 %) [46] | 4 / none (3 %) [48] | 4,5 / 4 (16 %) [51] | 4,5,6 / 4,5 (29 %) [54] | 19 classes / 4,5,6 (66 %) [59] | 33 classes / 4,5,6,8,9 (99 %) [65] | 34 classes / 19 classes (100 %) [70] |
| 45.255 | 4 / none (3 %) [61] | 4 / none (3 %) [63] | 4,5 / none (16 %) [65] | 4,5,6 / 4 (29 %) [67] | 4,5,6,8,9,10 / 4,5 (56 %) [69] | 31 classes / 4,5 (96 %) [71] | 34 classes / 4,5,6 (100 %) [76] | 34 classes / 4,5,6,8,9,10 (100 %) [81] |
| 64.000 | – | 4,5,6 / none (29 %) [86] | 18 classes / 4 (62 %) [90] | 31 classes / 4,5 (96 %) [92] | 34 classes / 4,5 (100 %) [94] | 34 classes / 4,5,6 (100 %) [96] | 34 classes / 4,5,6 (100 %) [97] | 34 classes / 4,5,6,8,9 (100 %) [99] |
| 90.510 | – | 34 classes / none (100 %) [109] | 34 classes / 4 (100 %) [113] | 34 classes / 4,5 (100 %) [117] | 34 classes / 4,5 (100 %) [121] | 34 classes / 4,5,6,9 (100 %) [124] | 34 classes / 4,5,6,8,9,10 (100 %) [127] | 34 classes / 23 classes (100 %) [130] |
| 128.000 | – | – | – | 34 classes / 4,5,6,9 (100 %) [157] | 34 classes / 4,5,6,8,9 (100 %) [161] | 34 classes / 18 classes (100 %) [164] | 34 classes / 29 classes (100 %) [167] | 34 classes / 33 classes (100 %) [171] |
| 181.019 | – | – | – | – | – | – | – | 34 classes / 34 classes (100 %) [223] |

## 3. complex0.1, calibrated: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | none / none (0 %) [7] | none / none (0 %) [9] | 4,5 / 4 (16 %) [12] | 25 classes / 18 classes (85 %) [15] | 34 classes / 34 classes (100 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | none / none (0 %) [9] | none / none (0 %) [10] | none / none (0 %) [12] | 4,5,6 / 4,5 (29 %) [15] | 33 classes / 30 classes (99 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | none / none (0 %) [11] | none / none (0 %) [12] | none / none (0 %) [13] | none / none (0 %) [15] | 18 classes / 4,5,6,8,9 (62 %) [20] | 33 classes / 30 classes (99 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 8.000 | none / none (0 %) [14] | none / none (0 %) [16] | none / none (0 %) [17] | none / none (0 %) [18] | 4,5,6 / 4 (29 %) [20] | 19 classes / 4,5,6,8,9 (66 %) [24] | 33 classes / 31 classes (99 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 11.314 | none / none (0 %) [18] | none / none (0 %) [20] | none / none (0 %) [21] | none / none (0 %) [22] | 4 / none (3 %) [23] | 4,5,6 / 4 (29 %) [24] | 22 classes / 4,5,6,8,9,10 (72 %) [30] | 33 classes / 30 classes (99 %) [36] |
| 16.000 | none / none (0 %) [24] | none / none (0 %) [25] | none / none (0 %) [27] | 4 / none (3 %) [28] | 4,5 / none (16 %) [29] | 4,5,6 / 4 (29 %) [31] | 4,5,6,8 / 4,5 (35 %) [32] | 25 classes / 4,5,6,8,9 (85 %) [36] |
| 22.627 | none / none (0 %) [31] | none / none (0 %) [33] | none / none (0 %) [35] | 4 / none (3 %) [36] | 4,5 / 4 (16 %) [38] | 4,5,6 / 4 (29 %) [39] | 4,5,6,8 / 4,5 (35 %) [40] | 19 classes / 4,5,6 (66 %) [41] |
| 32.000 | none / none (0 %) [42] | none / none (0 %) [44] | none / none (0 %) [45] | 4 / none (3 %) [47] | 4,5,6 / 4 (29 %) [49] | 4,5,6 / 4 (29 %) [50] | 19 classes / 4,5 (66 %) [52] | 34 classes / 4,5,6 (100 %) [53] |
| 45.255 | none / none (0 %) [56] | none / none (0 %) [58] | 4 / none (3 %) [60] | 4,5 / none (16 %) [62] | 4,5,6,8 / 4 (35 %) [64] | 25 classes / 4 (85 %) [66] | 34 classes / 4,5 (100 %) [67] | 34 classes / 4,5,6 (100 %) [69] |
| 64.000 | – | 4,5 / none (16 %) [79] | 4,5,6 / none (29 %) [81] | 25 classes / 4 (85 %) [83] | 34 classes / 4 (100 %) [85] | 34 classes / 4,5 (100 %) [87] | 34 classes / 4,5 (100 %) [89] | 34 classes / 4,5,6 (100 %) [90] |
| 90.510 | – | – | 34 classes / 4 (100 %) [109] | 34 classes / 4,5 (100 %) [112] | 34 classes / 4,5 (100 %) [114] | 34 classes / 4,5,6 (100 %) [116] | 34 classes / 4,5,6,9 (100 %) [118] | 34 classes / 16 classes (100 %) [120] |
| 128.000 | – | – | – | – | 34 classes / 4,5,6,8,9 (100 %) [154] | 34 classes / 15 classes (100 %) [157] | 34 classes / 28 classes (100 %) [159] | 34 classes / 33 classes (100 %) [161] |
| 181.019 | – | – | – | – | – | – | – | 34 classes / 34 classes (100 %) [218] |

## 3. complex1, formula: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | 4,5,6 / none (29 %) [13] | 31 classes / 4,5,6,8,9 (96 %) [16] | 34 classes / 32 classes (100 %) [19] | 34 classes / 34 classes (100 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [29] | 34 classes / 34 classes (100 %) [32] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | 4,5 / none (16 %) [14] | 25 classes / 4,5,6 (85 %) [17] | 34 classes / 24 classes (100 %) [20] | 34 classes / 34 classes (100 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | 4,5 / none (16 %) [15] | 25 classes / 4 (85 %) [18] | 34 classes / 4,5,6,8,9 (100 %) [21] | 34 classes / 30 classes (100 %) [24] | 34 classes / 34 classes (100 %) [28] | 34 classes / 34 classes (100 %) [31] | 34 classes / 34 classes (100 %) [34] | 34 classes / 34 classes (100 %) [37] |
| 8.000 | 19 classes / none (66 %) [18] | – | 34 classes / 4,5,6,8,9 (100 %) [24] | 34 classes / 26 classes (100 %) [27] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] | 34 classes / 34 classes (100 %) [39] |
| 11.314 | 34 classes / none (100 %) [22] | – | – | 34 classes / 26 classes (100 %) [30] | 34 classes / 34 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] | 34 classes / 34 classes (100 %) [39] | 34 classes / 34 classes (100 %) [42] |
| 16.000 | – | – | – | 34 classes / 33 classes (100 %) [36] | – | – | 34 classes / 34 classes (100 %) [44] | 34 classes / 34 classes (100 %) [47] |

## 3. complex1, S7: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | 34 classes / 25 classes (100 %) [19] | 34 classes / 34 classes (100 %) [25] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [37] | 34 classes / 34 classes (100 %) [44] | 34 classes / 34 classes (100 %) [52] | 34 classes / 34 classes (100 %) [56] | 34 classes / 34 classes (100 %) [65] |
| 4.000 | 34 classes / 27 classes (100 %) [23] | 34 classes / 34 classes (100 %) [29] | 34 classes / 34 classes (100 %) [35] | 34 classes / 34 classes (100 %) [41] | 34 classes / 34 classes (100 %) [48] | 34 classes / 34 classes (100 %) [54] | 34 classes / 34 classes (100 %) [61] | 34 classes / 34 classes (100 %) [70] |
| 5.657 | 34 classes / 33 classes (100 %) [29] | 34 classes / 34 classes (100 %) [35] | 34 classes / 34 classes (100 %) [43] | 34 classes / 34 classes (100 %) [48] | 34 classes / 34 classes (100 %) [54] | 34 classes / 34 classes (100 %) [61] | 34 classes / 34 classes (100 %) [66] | 34 classes / 34 classes (100 %) [73] |
| 8.000 | 34 classes / 34 classes (100 %) [36] | – | 34 classes / 34 classes (100 %) [51] | 34 classes / 34 classes (100 %) [57] | 34 classes / 34 classes (100 %) [64] | 34 classes / 34 classes (100 %) [71] | 34 classes / 34 classes (100 %) [77] | 34 classes / 34 classes (100 %) [82] |
| 11.314 | – | – | – | 34 classes / 34 classes (100 %) [65] | 34 classes / 34 classes (100 %) [72] | 34 classes / 34 classes (100 %) [79] | 34 classes / 34 classes (100 %) [85] | 34 classes / 34 classes (100 %) [92] |
| 16.000 | – | – | – | – | – | – | 34 classes / 34 classes (100 %) [104] | 34 classes / 34 classes (100 %) [110] |

## 3. complex1, calibrated: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | none / none (0 %) [7] | none / none (0 %) [9] | 4,5,6 / none (29 %) [12] | 34 classes / 4,5,6,8,9,10 (100 %) [15] | 34 classes / 34 classes (100 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | none / none (0 %) [9] | none / none (0 %) [10] | 4,5 / none (16 %) [12] | 31 classes / 4,5 (96 %) [15] | 34 classes / 32 classes (100 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | none / none (0 %) [11] | none / none (0 %) [12] | 4,5,6 / none (29 %) [13] | 29 classes / none (93 %) [15] | 34 classes / 18 classes (100 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 8.000 | – | – | 34 classes / none (100 %) [17] | 34 classes / 4 (100 %) [18] | 34 classes / 4,5,6 (100 %) [20] | 34 classes / 28 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 11.314 | – | – | – | 34 classes / 4,5 (100 %) [22] | 34 classes / 4,5,6,8,9 (100 %) [23] | 34 classes / 17 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 16.000 | – | – | – | – | – | – | 34 classes / 34 classes (100 %) [32] | 34 classes / 34 classes (100 %) [36] |

## 3. real, formula: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | 4 / 4 (3 %) [13] | 19 classes / 4,5,6,8,9,10 (66 %) [16] | 33 classes / 32 classes (99 %) [19] | 34 classes / 34 classes (100 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [29] | 34 classes / 34 classes (100 %) [32] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | none / none (0 %) [14] | 4,5,6 / 4,5,6 (29 %) [17] | 25 classes / 19 classes (85 %) [20] | 33 classes / 33 classes (99 %) [23] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [33] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | none / none (0 %) [15] | 4 / 4 (3 %) [18] | 4,5,6,8 / 4,5,6 (35 %) [21] | 25 classes / 22 classes (85 %) [24] | 33 classes / 33 classes (99 %) [28] | 34 classes / 34 classes (100 %) [31] | 34 classes / 34 classes (100 %) [34] | 34 classes / 34 classes (100 %) [37] |
| 8.000 | none / none (0 %) [18] | none / none (0 %) [21] | 4,5,6 / 4,5 (29 %) [24] | 4,5,6,8,9,10 / 4,5,6,8,9 (56 %) [27] | 25 classes / 22 classes (85 %) [30] | 33 classes / 31 classes (99 %) [33] | 34 classes / 33 classes (100 %) [36] | 34 classes / 34 classes (100 %) [39] |
| 11.314 | none / none (0 %) [22] | none / none (0 %) [25] | 4 / 4 (3 %) [27] | 4,5,6 / 4,5,6 (29 %) [30] | 4,5,6,8,9,10 / 4,5,6,8,9 (56 %) [33] | 25 classes / 19 classes (85 %) [36] | 31 classes / 25 classes (96 %) [39] | 33 classes / 32 classes (99 %) [42] |
| 13.532 | none / none (0 %) [25] | none / none (0 %) [27] | 4 / 4 (3 %) [30] | 4,5,6 / 4,5 (29 %) [33] | 4,5,6,8,9 / 4,5,6,8 (49 %) [36] | 18 classes / 4,5,6,8,9,10 (62 %) [38] | 25 classes / 22 classes (85 %) [41] | 31 classes / 26 classes (96 %) [44] |
| 16.000 | none / none (0 %) [28] | none / none (0 %) [31] | 4 / none (3 %) [33] | 4,5,6 / 4,5 (29 %) [36] | 4,5,6,8 / 4,5,6 (35 %) [39] | 4,5,6,8,9 / 4,5,6,8,9 (49 %) [41] | 19 classes / 18 classes (66 %) [44] | 25 classes / 25 classes (85 %) [47] |
| 22.627 | none / none (0 %) [36] | none / none (0 %) [39] | none / none (0 %) [41] | 4,5 / 4 (16 %) [44] | 4,5,6 / 4,5 (29 %) [47] | 4,5,6 / 4,5,6 (29 %) [49] | 4,5,6,8,9 / 4,5,6,8 (49 %) [52] | 18 classes / 4,5,6,8,9,10 (62 %) [55] |
| 27.063 | none / none (0 %) [41] | none / none (0 %) [44] | none / none (0 %) [47] | 4 / 4 (3 %) [50] | 4,5 / 4,5 (16 %) [52] | 4,5,6 / 4,5,6 (29 %) [55] | 4,5,6,8 / 4,5,6 (35 %) [58] | 4,5,6,8,9,10 / 4,5,6,8,9 (56 %) [60] |
| 32.000 | none / none (0 %) [47] | none / none (0 %) [50] | none / none (0 %) [53] | 4 / 4 (3 %) [56] | 4,5 / 4,5 (16 %) [59] | 4,5,6 / 4,5 (29 %) [61] | 4,5,6 / 4,5,6 (29 %) [64] | 4,5,6,8,9 / 4,5,6,8 (49 %) [67] |
| 45.255 | none / none (0 %) [62] | none / none (0 %) [65] | none / none (0 %) [69] | 4 / 4 (3 %) [72] | 4,5 / 4 (16 %) [75] | 4,5 / 4,5 (16 %) [78] | 4,5,6 / 4,5,6 (29 %) [81] | 4,5,6 / 4,5,6 (29 %) [83] |
| 54.127 | none / none (0 %) [72] | none / none (0 %) [75] | none / none (0 %) [79] | 4 / none (3 %) [82] | 4 / 4 (3 %) [85] | 4,5 / 4 (16 %) [88] | 4,5,6 / 4,5 (29 %) [91] | 4,5,6 / 4,5,6 (29 %) [94] |
| 64.000 | none / none (0 %) [83] | none / none (0 %) [87] | none / none (0 %) [90] | 4 / none (3 %) [94] | 4 / 4 (3 %) [97] | 4,5 / 4 (16 %) [100] | 4,5 / 4,5 (16 %) [103] | 4,5,6 / 4,5 (29 %) [106] |
| 90.510 | none / none (0 %) [111] | none / none (0 %) [116] | none / none (0 %) [120] | none / none (0 %) [124] | 4 / 4 (3 %) [128] | 4 / 4 (3 %) [131] | 4,5 / 4 (16 %) [135] | 4,5 / 4,5 (16 %) [138] |
| 108.253 | none / none (0 %) [130] | none / none (0 %) [135] | none / none (0 %) [140] | none / none (0 %) [144] | 4 / none (3 %) [148] | 4 / 4 (3 %) [151] | 4 / 4 (3 %) [155] | 4,5 / 4 (16 %) [158] |
| 128.000 | none / none (0 %) [151] | none / none (0 %) [157] | none / none (0 %) [161] | none / none (0 %) [166] | 4 / none (3 %) [170] | 4 / 4 (3 %) [174] | 4 / 4 (3 %) [177] | 4,5 / 4 (16 %) [181] |
| 181.019 | none / none (0 %) [207] | none / none (0 %) [213] | none / none (0 %) [218] | none / none (0 %) [223] | none / none (0 %) [228] | 4 / none (3 %) [232] | 4 / 4 (3 %) [236] | 4 / 4 (3 %) [241] |
| 256.000 | none / none (0 %) [286] | none / none (0 %) [292] | none / none (0 %) [298] | none / none (0 %) [303] | none / none (0 %) [309] | none / none (0 %) [314] | 4 / none (3 %) [318] | 4 / 4 (3 %) [323] |
| 362.039 | none / none (0 %) [395] | none / none (0 %) [403] | none / none (0 %) [409] | none / none (0 %) [415] | none / none (0 %) [421] | none / none (0 %) [427] | none / none (0 %) [432] | 4 / none (3 %) [437] |
| 400.000 | none / none (0 %) [435] | none / none (0 %) [442] | none / none (0 %) [449] | none / none (0 %) [455] | none / none (0 %) [461] | none / none (0 %) [467] | none / none (0 %) [472] | none / none (0 %) [478] |

## 3. real, S7: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | none / none (0 %) [11] | 19 classes / 4,5,6,8,9,10 (66 %) [16] | 34 classes / 34 classes (100 %) [21] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [34] | 34 classes / 34 classes (100 %) [40] | 34 classes / 34 classes (100 %) [47] | 34 classes / 34 classes (100 %) [53] |
| 4.000 | none / none (0 %) [11] | 4 / 4 (3 %) [15] | 25 classes / 19 classes (85 %) [20] | 34 classes / 34 classes (100 %) [26] | 34 classes / 34 classes (100 %) [31] | 34 classes / 34 classes (100 %) [38] | 34 classes / 34 classes (100 %) [44] | 34 classes / 34 classes (100 %) [50] |
| 5.657 | none / none (0 %) [12] | none / none (0 %) [16] | 4,5,6 / 4,5,6 (29 %) [20] | 31 classes / 25 classes (96 %) [26] | 34 classes / 34 classes (100 %) [32] | 34 classes / 34 classes (100 %) [38] | 34 classes / 34 classes (100 %) [45] | 34 classes / 34 classes (100 %) [51] |
| 8.000 | none / none (0 %) [15] | none / none (0 %) [17] | none / none (0 %) [20] | 4,5,6,8,9 / 4,5,6,8,9 (49 %) [26] | 27 classes / 25 classes (92 %) [31] | 34 classes / 33 classes (100 %) [37] | 34 classes / 34 classes (100 %) [44] | 34 classes / 34 classes (100 %) [50] |
| 11.314 | none / none (0 %) [19] | none / none (0 %) [21] | none / none (0 %) [22] | 4,5 / 4 (16 %) [27] | 4,5,6,8,9 / 4,5,6,8 (49 %) [32] | 25 classes / 22 classes (85 %) [37] | 33 classes / 31 classes (99 %) [43] | 34 classes / 34 classes (100 %) [49] |
| 13.532 | none / none (0 %) [23] | none / none (0 %) [25] | none / none (0 %) [28] | 4,5,6 / 4,5,6 (29 %) [34] | 19 classes / 4,5,6,8,9,10 (66 %) [40] | 29 classes / 25 classes (93 %) [46] | 33 classes / 33 classes (99 %) [52] | 34 classes / 34 classes (100 %) [58] |
| 16.000 | none / none (0 %) [24] | none / none (0 %) [26] | none / none (0 %) [28] | none / none (0 %) [30] | 4,5 / 4 (16 %) [33] | 4,5,6,8 / 4,5,6 (35 %) [38] | 19 classes / 18 classes (66 %) [44] | 29 classes / 25 classes (93 %) [50] |
| 22.627 | none / none (0 %) [31] | none / none (0 %) [33] | none / none (0 %) [36] | none / none (0 %) [38] | 4 / none (3 %) [40] | 4,5 / 4 (16 %) [43] | 4,5,6 / 4,5,6 (29 %) [47] | 4,5,6,8,9,10 / 4,5,6,8,9 (56 %) [53] |
| 27.063 | none / none (0 %) [38] | none / none (0 %) [41] | none / none (0 %) [43] | none / none (0 %) [45] | 4 / 4 (3 %) [47] | 4,5 / 4 (16 %) [49] | 4,5,6 / 4,5 (29 %) [52] | 4,5,6,8,9 / 4,5,6,8 (49 %) [58] |
| 32.000 | none / none (0 %) [41] | none / none (0 %) [44] | none / none (0 %) [46] | none / none (0 %) [49] | 4 / none (3 %) [51] | 4 / 4 (3 %) [53] | 4,5 / 4 (16 %) [55] | 4,5,6 / 4,5 (29 %) [57] |
| 45.255 | none / none (0 %) [55] | none / none (0 %) [58] | none / none (0 %) [61] | none / none (0 %) [64] | none / none (0 %) [66] | 4 / 4 (3 %) [69] | 4 / 4 (3 %) [71] | 4,5 / 4,5 (16 %) [73] |
| 54.127 | none / none (0 %) [68] | none / none (0 %) [71] | none / none (0 %) [74] | none / none (0 %) [76] | 4 / none (3 %) [79] | 4 / 4 (3 %) [81] | 4 / 4 (3 %) [83] | 4,5 / 4 (16 %) [85] |
| 64.000 | none / none (0 %) [75] | none / none (0 %) [79] | none / none (0 %) [82] | none / none (0 %) [85] | none / none (0 %) [88] | 4 / none (3 %) [90] | 4 / 4 (3 %) [93] | 4,5 / 4 (16 %) [95] |
| 90.510 | none / none (0 %) [103] | none / none (0 %) [107] | none / none (0 %) [110] | none / none (0 %) [114] | none / none (0 %) [117] | none / none (0 %) [120] | 4 / none (3 %) [123] | 4 / 4 (3 %) [126] |
| 108.253 | none / none (0 %) [126] | none / none (0 %) [130] | none / none (0 %) [133] | none / none (0 %) [136] | none / none (0 %) [139] | none / none (0 %) [142] | 4 / 4 (3 %) [145] | 4 / 4 (3 %) [147] |
| 128.000 | none / none (0 %) [141] | none / none (0 %) [146] | none / none (0 %) [150] | none / none (0 %) [153] | none / none (0 %) [157] | none / none (0 %) [161] | none / none (0 %) [164] | 4 / 4 (3 %) [167] |
| 181.019 | none / none (0 %) [195] | none / none (0 %) [200] | none / none (0 %) [205] | none / none (0 %) [209] | none / none (0 %) [213] | none / none (0 %) [217] | none / none (0 %) [220] | 4 / none (3 %) [224] |
| 256.000 | none / none (0 %) [272] | none / none (0 %) [278] | none / none (0 %) [284] | none / none (0 %) [288] | none / none (0 %) [293] | none / none (0 %) [297] | none / none (0 %) [300] | none / none (0 %) [304] |
| 362.039 | none / none (0 %) [388] | none / none (0 %) [394] | none / none (0 %) [399] | none / none (0 %) [404] | none / none (0 %) [408] | none / none (0 %) [412] | none / none (0 %) [416] | none / none (0 %) [420] |
| 400.000 | none / none (0 %) [427] | none / none (0 %) [433] | none / none (0 %) [438] | none / none (0 %) [443] | none / none (0 %) [448] | none / none (0 %) [452] | none / none (0 %) [456] | none / none (0 %) [460] |

## 3. real, calibrated: classes on RCR per (k a, ε)

Each cell: the classes (|t|²/w²) whose diagonal form fails, by the bound E ≤ θ ε (C_T and θ of §2) / by the measurement (the rounding of f64, L = 2P − 2, against RCR above ε; the truncation is the rule's, the same as RCR's), and the share of an interior box's 189 V pairs on RCR by the bound; p in brackets; `–` where the level is pruned or the unit is missing.

| k a | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.828 | none / none (0 %) [7] | none / none (0 %) [9] | 4 / 4 (3 %) [12] | 22 classes / 17 classes (72 %) [15] | 34 classes / 34 classes (100 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 4.000 | none / none (0 %) [9] | none / none (0 %) [10] | none / none (0 %) [12] | 4,5,6 / 4,5 (29 %) [15] | 31 classes / 26 classes (96 %) [20] | 34 classes / 34 classes (100 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 5.657 | none / none (0 %) [11] | none / none (0 %) [12] | none / none (0 %) [13] | none / none (0 %) [15] | 4,5,6,8,9 / 4,5,6,8,9 (49 %) [20] | 31 classes / 25 classes (96 %) [24] | 34 classes / 34 classes (100 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 8.000 | none / none (0 %) [14] | none / none (0 %) [16] | none / none (0 %) [17] | none / none (0 %) [18] | 4 / 4 (3 %) [20] | 4,5,6,8,9 / 4,5,6,8,9 (49 %) [24] | 31 classes / 25 classes (96 %) [30] | 34 classes / 34 classes (100 %) [36] |
| 11.314 | none / none (0 %) [18] | none / none (0 %) [20] | none / none (0 %) [21] | none / none (0 %) [22] | none / none (0 %) [23] | 4 / 4 (3 %) [24] | 4,5,6,8,9 / 4,5,6,8 (49 %) [30] | 27 classes / 25 classes (92 %) [36] |
| 13.532 | none / none (0 %) [21] | none / none (0 %) [22] | none / none (0 %) [24] | none / none (0 %) [25] | none / none (0 %) [26] | 4 / 4 (3 %) [27] | 4,5,6 / 4,5 (29 %) [30] | 19 classes / 4,5,6,8,9,10 (66 %) [36] |
| 16.000 | none / none (0 %) [24] | none / none (0 %) [25] | none / none (0 %) [27] | none / none (0 %) [28] | none / none (0 %) [29] | 4 / 4 (3 %) [31] | 4,5 / 4,5 (16 %) [32] | 4,5,6,8 / 4,5,6 (35 %) [36] |
| 22.627 | none / none (0 %) [31] | none / none (0 %) [33] | none / none (0 %) [35] | none / none (0 %) [36] | none / none (0 %) [38] | 4 / none (3 %) [39] | 4,5 / 4 (16 %) [40] | 4,5 / 4,5 (16 %) [41] |
| 27.063 | none / none (0 %) [36] | none / none (0 %) [38] | none / none (0 %) [40] | none / none (0 %) [41] | none / none (0 %) [43] | 4 / none (3 %) [44] | 4,5 / 4 (16 %) [46] | 4,5 / 4 (16 %) [47] |
| 32.000 | none / none (0 %) [42] | none / none (0 %) [44] | none / none (0 %) [45] | none / none (0 %) [47] | none / none (0 %) [49] | 4 / none (3 %) [50] | 4 / 4 (3 %) [52] | 4,5 / 4 (16 %) [53] |
| 45.255 | none / none (0 %) [56] | none / none (0 %) [58] | none / none (0 %) [60] | none / none (0 %) [62] | none / none (0 %) [64] | none / none (0 %) [66] | 4 / none (3 %) [67] | 4 / 4 (3 %) [69] |
| 54.127 | none / none (0 %) [66] | none / none (0 %) [68] | none / none (0 %) [70] | none / none (0 %) [72] | none / none (0 %) [74] | none / none (0 %) [76] | 4 / none (3 %) [77] | 4 / 4 (3 %) [79] |
| 64.000 | none / none (0 %) [76] | none / none (0 %) [79] | none / none (0 %) [81] | none / none (0 %) [83] | none / none (0 %) [85] | none / none (0 %) [87] | 4 / none (3 %) [89] | 4 / 4 (3 %) [90] |
| 90.510 | none / none (0 %) [104] | none / none (0 %) [107] | none / none (0 %) [109] | none / none (0 %) [112] | none / none (0 %) [114] | none / none (0 %) [116] | none / none (0 %) [118] | 4 / none (3 %) [120] |
| 108.253 | none / none (0 %) [123] | none / none (0 %) [126] | none / none (0 %) [128] | none / none (0 %) [131] | none / none (0 %) [133] | none / none (0 %) [135] | none / none (0 %) [138] | 4 / none (3 %) [140] |
| 128.000 | none / none (0 %) [143] | none / none (0 %) [146] | none / none (0 %) [149] | none / none (0 %) [152] | none / none (0 %) [154] | none / none (0 %) [157] | none / none (0 %) [159] | none / none (0 %) [161] |
| 181.019 | none / none (0 %) [198] | none / none (0 %) [201] | none / none (0 %) [205] | none / none (0 %) [208] | none / none (0 %) [210] | none / none (0 %) [213] | none / none (0 %) [216] | none / none (0 %) [218] |
| 256.000 | none / none (0 %) [275] | none / none (0 %) [279] | none / none (0 %) [282] | none / none (0 %) [286] | none / none (0 %) [289] | none / none (0 %) [292] | none / none (0 %) [295] | none / none (0 %) [298] |
| 362.039 | none / none (0 %) [383] | none / none (0 %) [388] | none / none (0 %) [392] | none / none (0 %) [395] | none / none (0 %) [399] | none / none (0 %) [402] | none / none (0 %) [406] | none / none (0 %) [409] |
| 400.000 | none / none (0 %) [422] | none / none (0 %) [426] | none / none (0 %) [431] | none / none (0 %) [434] | none / none (0 %) [438] | none / none (0 %) [442] | none / none (0 %) [445] | none / none (0 %) [448] |

## 3b. §3.6's example: kD = 500, ε = 10⁻⁶, the formula's p

The rounding (diagonal against RCR, worst placement) on the axis at |t| = 2w and 3w, with T rounded from double-double (f64) and built in f64 (f64b, the seed's model), against the 1-D seed of Appendix B.1 (§3.6: below 10⁻⁸ at 3w on every level).

| level | k a | p | seed, 2w | f64, 2w | f64b, 2w | f64, 3w | f64b, 3w | E (§2), 2w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 108.253 | 144 | 1e-7 | 4.0e-10 | 2.9e-8 | 2.4e-15 | 3.7e-14 | 1.8e-8 |
| 3 | 54.127 | 82 | 8e-5 | 2.6e-7 | 1.7e-5 | 1.9e-15 | 2.9e-14 | 5.4e-6 |
| 4 | 27.063 | 50 | 1.5e-2 | 1.6e-4 | 7.4e-3 | 5.9e-15 | 1.4e-13 | 1.9e-3 |
| 5 | 13.532 | 33 | ≈ 10 | 1.3e-1 | 4.2e0 | 7.9e-11 | 2.5e-9 | 6.4e-1 |

## 4. f32

Per kind and k a: the smallest f32 total error over the classes and bandwidths at any p run, and the classes (|t|²) where f32 at L = 2P − 2 meets 10⁻³ at the p of the calibrated rule at 10⁻³.

| kind | k a | best f32 total (class, band, p) | f32 meets 10⁻³ at calibrated p |
| --- | --- | --- | --- |
| complex0.1 | 2.828 | 6.9e-7 (27, f32:P-1, 15) | {9, 18, 19, 22} |
| complex0.1 | 4.000 | 5.0e-7 (19, f32:P-1, 20) | {8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 5.657 | 6.0e-7 (18, f32:P-1, 23) | {6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 8.000 | 4.0e-7 (17, f32:P-1, 30) | {5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 11.314 | 3.9e-7 (13, f32:P-1, 37) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 16.000 | 4.1e-7 (27, f32:3(P-1)/2, 31) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 22.627 | 5.9e-7 (18, f32:P-1, 62) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 32.000 | 7.4e-7 (17, f32:3(P-1)/2, 52) | {5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 45.255 | 1.4e-6 (17, f32:3(P-1)/2, 67) | {5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| complex0.1 | 64.000 | 2.6e-6 (18, f32:3(P-1)/2, 89) | none |
| complex0.1 | 90.510 | 1.3e-5 (18, f32:3(P-1)/2, 120) | none |
| complex0.1 | 128.000 | 5.3e-5 (18, f32:3(P-1)/2, 154) | none |
| complex0.1 | 181.019 | 1.1e-3 (18, f32:3(P-1)/2, 218) | none |
| complex1 | 2.828 | 2.5e-5 (22, f32:P-1, 19) | none |
| complex1 | 4.000 | 2.8e-5 (22, f32:3(P-1)/2, 14) | none |
| complex1 | 5.657 | 2.7e-5 (22, f32:P-1, 24) | none |
| complex1 | 8.000 | 2.7e-5 (17, f32:P-1, 30) | none |
| complex1 | 11.314 | 1.1e-4 (18, f32:P-1, 36) | none |
| complex1 | 16.000 | 7.8e-4 (18, f32:P-1, 44) | none |
| real | 2.828 | 2.9e-7 (22, f32:P-1, 15) | {8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 4.000 | 2.5e-7 (22, f32:P-1, 20) | {5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 5.657 | 2.8e-7 (18, f32:P-1, 28) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 8.000 | 1.9e-7 (17, f32:P-1, 30) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 11.314 | 2.0e-7 (14, f32:3(P-1)/2, 27) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 13.532 | 2.4e-7 (14, f32:3(P-1)/2, 27) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 16.000 | 2.6e-7 (10, f32:P-1, 50) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 22.627 | 2.8e-7 (10, f32:3(P-1)/2, 41) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 27.063 | 3.5e-7 (11, f32:3(P-1)/2, 49) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 32.000 | 3.1e-7 (14, f32:3(P-1)/2, 53) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 45.255 | 3.8e-7 (18, f32:3(P-1)/2, 78) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 54.127 | 4.2e-7 (13, f32:3(P-1)/2, 85) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 64.000 | 4.5e-7 (10, f32:3(P-1)/2, 106) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 90.510 | 5.5e-7 (14, f32:2P-2, 118) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 108.253 | 5.7e-7 (14, f32:2P-2, 133) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 128.000 | 5.9e-7 (6, f32:2P-2, 150) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 181.019 | 6.6e-7 (13, f32:2P-2, 208) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 256.000 | 8.6e-7 (18, f32:2P-2, 298) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 362.039 | 9.8e-7 (14, f32:2P-2, 421) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |
| real | 400.000 | 1.0e-6 (13, f32:2P-2, 422) | {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27} |

## 5. T interpolated in ŝ · t̂, and T built in f64

Ratio of the rounding error to that of T rounded from double-double (f64, 2P − 2), over the points where the latter exceeds 10⁻¹³, and the largest ratio to the bound E of §2:

| kind | variant | points | ratio min / median / max | max measured / E |
| --- | --- | --- | --- | --- |
| complex0.1 | f64i:2P-2 | 5641 | 0.46 / 1.15 / 4.52 | 2.25 |
| complex0.1 | f64b:2P-2 | 5641 | 0.88 / 37.72 / 786.21 | 17.66 |
| complex1 | f64i:2P-2 | 3209 | 0.31 / 1.27 / 6.12 | 0.00 |
| complex1 | f64b:2P-2 | 3209 | 0.96 / 21.71 / 680.73 | 0.08 |
| real | f64i:2P-2 | 3556 | 0.33 / 1.20 / 4.11 | 2.95 |
| real | f64b:2P-2 | 3556 | 0.75 / 54.58 / 866.19 | 47.00 |

## 6. The bandwidth of T (f64, formula p)

The worst total error over the classes, and the nearest class (|t|² = 4), for L = P − 1, ⌈3(P − 1)/2⌉ and 2P − 2, at the formula's p, against ε; real k.

| k a | ε | p | worst: P−1 / 3(P−1)/2 / 2P−2 | nearest: P−1 / 3(P−1)/2 / 2P−2 | truncation (RCR) |
| --- | --- | --- | --- | --- | --- |
| 2.828 | 1e-4 | 16 | 2.6e-3 / 3.6e-4 / 1.5e3 | 2.6e-3 / 3.6e-4 / 1.5e3 | 2.3e-5 |
| 2.828 | 1e-6 | 23 | 4.8e-4 / 2.0e6 / 6.9e17 | 4.8e-4 / 2.0e6 / 6.9e17 | 7.5e-7 |
| 2.828 | 1e-8 | 29 | 1.0e0 / 1.3e16 / 6.3e32 | 1.0e0 / 1.3e16 / 6.3e32 | 8.5e-8 |
| 4.000 | 1e-4 | 17 | 2.7e-3 / 3.0e-4 / 6.4e-1 | 2.7e-3 / 3.0e-4 / 6.4e-1 | 1.2e-5 |
| 4.000 | 1e-6 | 23 | 6.5e-4 / 8.4e0 / 1.6e11 | 6.5e-4 / 8.4e0 / 1.6e11 | 9.5e-7 |
| 4.000 | 1e-8 | 30 | 2.2e-4 / 1.3e10 / 7.5e25 | 2.2e-4 / 1.3e10 / 7.5e25 | 4.9e-8 |
| 5.657 | 1e-4 | 18 | 3.0e-3 / 2.7e-4 / 2.7e-4 | 3.0e-3 / 2.7e-4 / 2.7e-4 | 6.3e-6 |
| 5.657 | 1e-6 | 24 | 5.9e-4 / 2.0e-4 / 8.5e5 | 5.9e-4 / 2.0e-4 / 8.5e5 | 4.6e-7 |
| 5.657 | 1e-8 | 31 | 1.1e-4 / 1.2e5 / 7.0e18 | 1.1e-4 / 1.2e5 / 7.0e18 | 4.1e-8 |
| 8.000 | 1e-4 | 21 | 2.0e-3 / 6.2e-5 / 2.0e-5 | 2.0e-3 / 6.2e-5 / 2.0e-5 | 1.8e-6 |
| 8.000 | 1e-6 | 27 | 2.5e-4 / 7.8e-6 / 7.6e3 | 2.5e-4 / 7.8e-6 / 7.6e3 | 9.1e-8 |
| 8.000 | 1e-8 | 33 | 5.1e-5 / 7.0e0 / 2.1e13 | 5.1e-5 / 7.0e0 / 2.1e13 | 6.1e-9 |
| 11.314 | 1e-4 | 25 | 8.1e-3 / 4.8e-5 / 7.4e-7 | 8.1e-3 / 4.8e-5 / 7.4e-7 | 7.2e-7 |
| 11.314 | 1e-6 | 30 | 6.9e-4 / 8.4e-6 / 6.1e-1 | 6.9e-4 / 8.4e-6 / 6.1e-1 | 7.2e-8 |
| 11.314 | 1e-8 | 36 | 8.8e-5 / 9.9e-5 / 9.1e7 | 8.8e-5 / 9.9e-5 / 9.1e7 | 5.1e-9 |
| 13.532 | 1e-4 | 27 | 3.0e-2 / 3.0e-5 / 3.1e-7 | 2.0e-2 / 3.0e-5 / 3.1e-7 | 3.1e-7 |
| 13.532 | 1e-6 | 33 | 6.1e-4 / 2.8e-6 / 1.3e-1 | 6.1e-4 / 2.8e-6 / 1.3e-1 | 9.6e-9 |
| 13.532 | 1e-8 | 38 | 7.6e-5 / 1.1e-6 / 3.7e5 | 7.6e-5 / 1.1e-6 / 3.7e5 | 1.7e-9 |
| 16.000 | 1e-4 | 31 | 3.8e-2 / 1.5e-5 / 7.6e-8 | 2.1e-2 / 1.5e-5 / 7.6e-8 | 6.5e-8 |
| 16.000 | 1e-6 | 36 | 2.2e-3 / 2.2e-6 / 1.1e-2 | 1.4e-3 / 2.2e-6 / 1.1e-2 | 5.8e-9 |
| 16.000 | 1e-8 | 41 | 1.1e-4 / 3.3e-7 / 1.1e4 | 1.1e-4 / 3.3e-7 / 1.1e4 | 9.0e-10 |
| 22.627 | 1e-4 | 39 | 2.2e-1 / 7.8e-6 / 4.3e-8 | 1.1e-1 / 7.8e-6 / 2.3e-8 | 4.3e-8 |
| 22.627 | 1e-6 | 44 | 2.8e-2 / 8.4e-7 / 1.9e-4 | 1.4e-2 / 8.4e-7 / 1.9e-4 | 6.4e-10 |
| 22.627 | 1e-8 | 49 | 2.0e-3 / 1.2e-7 / 2.5e1 | 9.8e-4 / 1.2e-7 / 2.5e1 | 5.8e-11 |
| 27.063 | 1e-4 | 44 | 2.7e-1 / 1.3e-5 / 1.0e-7 | 1.9e-1 / 1.0e-5 / 1.7e-8 | 1.0e-7 |
| 27.063 | 1e-6 | 50 | 7.2e-2 / 4.2e-7 / 1.6e-4 | 2.9e-2 / 4.2e-7 / 1.6e-4 | 9.5e-11 |
| 27.063 | 1e-8 | 55 | 1.0e-2 / 4.3e-8 / 1.0e1 | 3.2e-3 / 4.3e-8 / 1.0e1 | 1.2e-11 |
| 32.000 | 1e-4 | 50 | 4.1e-1 / 2.5e-5 / 5.8e-8 | 3.2e-1 / 1.5e-5 / 1.4e-8 | 5.8e-8 |
| 32.000 | 1e-6 | 56 | 1.6e-1 / 4.0e-7 / 6.1e-5 | 8.8e-2 / 4.0e-7 / 6.1e-5 | 9.2e-11 |
| 32.000 | 1e-8 | 61 | 3.3e-2 / 3.1e-8 / 1.7e0 | 1.5e-2 / 3.1e-8 / 1.7e0 | 3.2e-12 |
| 45.255 | 1e-4 | 65 | 5.9e-1 / 2.2e-4 / 8.5e-8 | 5.9e-1 / 1.1e-4 / 1.6e-8 | 8.5e-8 |
| 45.255 | 1e-6 | 72 | 4.5e-1 / 5.0e-7 / 2.1e-6 | 4.5e-1 / 5.0e-7 / 2.1e-6 | 7.3e-11 |
| 45.255 | 1e-8 | 78 | 2.0e-1 / 1.2e-8 / 1.0e-1 | 1.8e-1 / 1.2e-8 / 1.0e-1 | 8.3e-14 |
| 54.127 | 1e-4 | 75 | 6.5e-1 / 6.1e-4 / 6.1e-8 | 5.7e-1 / 2.4e-4 / 1.1e-8 | 6.1e-8 |
| 54.127 | 1e-6 | 82 | 4.2e-1 / 3.1e-6 / 2.6e-7 | 4.2e-1 / 1.2e-6 / 2.6e-7 | 1.1e-10 |
| 54.127 | 1e-8 | 88 | 3.7e-1 / 1.4e-8 / 4.6e-3 | 3.2e-1 / 1.4e-8 / 4.6e-3 | 2.5e-13 |
| 64.000 | 1e-4 | 87 | 6.3e-1 / 1.2e-3 / 5.0e-8 | 6.3e-1 / 5.5e-4 / 4.2e-9 | 5.0e-8 |
| 64.000 | 1e-6 | 94 | 4.7e-1 / 1.1e-5 / 9.5e-8 | 4.7e-1 / 4.3e-6 / 9.5e-8 | 8.6e-11 |
| 64.000 | 1e-8 | 100 | 4.5e-1 / 4.9e-8 / 9.8e-4 | 4.5e-1 / 2.6e-8 / 9.8e-4 | 1.5e-13 |
| 90.510 | 1e-4 | 116 | 6.9e-1 / 1.4e-2 / 2.4e-8 | 6.1e-1 / 8.5e-3 / 2.1e-9 | 2.4e-8 |
| 90.510 | 1e-6 | 124 | 5.9e-1 / 3.6e-4 / 2.3e-9 | 5.6e-1 / 1.1e-4 / 2.3e-9 | 6.7e-11 |
| 90.510 | 1e-8 | 131 | 4.6e-1 / 2.8e-6 / 2.0e-5 | 4.5e-1 / 4.9e-7 / 2.0e-5 | 1.5e-13 |
| 108.253 | 1e-4 | 135 | 6.9e-1 / 3.0e-2 / 5.9e-8 | 6.9e-1 / 2.6e-2 / 2.5e-9 | 5.9e-8 |
| 108.253 | 1e-6 | 144 | 6.5e-1 / 2.6e-3 / 4.0e-10 | 6.4e-1 / 6.9e-4 / 4.0e-10 | 2.9e-11 |
| 108.253 | 1e-8 | 151 | 5.8e-1 / 4.8e-5 / 1.5e-6 | 5.0e-1 / 6.2e-6 / 1.5e-6 | 6.6e-14 |
| 128.000 | 1e-4 | 157 | 7.2e-1 / 4.8e-2 / 2.3e-8 | 6.6e-1 / 3.3e-2 / 8.9e-10 | 2.3e-8 |
| 128.000 | 1e-6 | 166 | 7.0e-1 / 6.4e-3 / 5.7e-11 | 6.3e-1 / 2.2e-3 / 5.7e-11 | 3.2e-11 |
| 128.000 | 1e-8 | 174 | 6.2e-1 / 3.0e-4 / 3.0e-7 | 5.5e-1 / 5.2e-5 / 3.0e-7 | 4.2e-14 |
| 181.019 | 1e-4 | 213 | 7.5e-1 / 1.4e-1 / 2.1e-8 | 6.7e-1 / 1.0e-1 / 5.8e-10 | 2.1e-8 |
| 181.019 | 1e-6 | 223 | 7.3e-1 / 3.3e-2 / 2.1e-11 | 6.4e-1 / 2.9e-2 / 6.9e-13 | 2.1e-11 |
| 181.019 | 1e-8 | 232 | 7.2e-1 / 1.3e-2 / 6.5e-10 | 6.5e-1 / 3.9e-3 / 6.5e-10 | 5.1e-14 |
| 256.000 | 1e-4 | 292 | 7.3e-1 / 2.9e-1 / 2.3e-8 | 6.6e-1 / 2.0e-1 / 2.2e-10 | 2.3e-8 |
| 256.000 | 1e-6 | 303 | 6.9e-1 / 1.8e-1 / 2.8e-11 | 6.4e-1 / 8.2e-2 / 2.0e-13 | 2.8e-11 |
| 256.000 | 1e-8 | 314 | 7.0e-1 / 7.7e-2 / 2.6e-12 | 6.1e-1 / 3.0e-2 / 2.6e-12 | 1.8e-14 |
| 362.039 | 1e-4 | 403 | 6.7e-1 / 2.8e-1 / 7.8e-9 | 5.9e-1 / 2.3e-1 / 4.3e-11 | 7.8e-9 |
| 362.039 | 1e-6 | 415 | 6.7e-1 / 2.3e-1 / 1.6e-11 | 5.6e-1 / 1.4e-1 / 4.6e-14 | 1.6e-11 |
| 362.039 | 1e-8 | 427 | 6.4e-1 / 1.6e-1 / 2.3e-14 | 5.5e-1 / 7.8e-2 / 9.8e-15 | 2.3e-14 |
| 400.000 | 1e-4 | 442 | 7.3e-1 / 4.6e-1 / 8.0e-9 | 7.3e-1 / 3.2e-1 / 4.6e-11 | 8.0e-9 |
| 400.000 | 1e-6 | 455 | 7.3e-1 / 3.6e-1 / 7.6e-12 | 7.3e-1 / 2.3e-1 / 3.2e-14 | 7.6e-12 |
| 400.000 | 1e-8 | 467 | 7.4e-1 / 2.2e-1 / 9.8e-15 | 7.1e-1 / 1.3e-1 / 6.0e-15 | 9.5e-15 |

## 7. Damped k

measured / (u 10^term) (f64, 2P − 2, C_T = 1) per kind and Im(k) a band, without a damping factor, with e^{Im(k)|t|} and with e^{Im(k)(|t| + 2a)}, over the points where the last bound lies above the floor (10⁻¹³): median / max of each.

| kind | Im(k) a | points | no factor | e^{Im(k)|t|} | e^{Im(k)(|t| + 2a)} |
| --- | --- | --- | --- | --- | --- |
| complex0.1 | [0.25, 0.5) | 1287 | 4.5e-2 / 1.2e0 | 1.2e-2 / 2.9e-1 | 5.9e-3 / 1.6e-1 |
| complex0.1 | [0.5, 1) | 1203 | 1.7e-1 / 5.3e0 | 1.3e-2 / 1.5e-1 | 3.4e-3 / 3.7e-2 |
| complex0.1 | [1, 2) | 1292 | 1.8e0 / 3.3e2 | 1.1e-2 / 1.2e-1 | 6.8e-4 / 1.3e-2 |
| complex0.1 | [2, 4) | 1258 | 3.6e2 / 9.5e6 | 9.8e-3 / 4.4e-2 | 3.5e-5 / 4.5e-4 |
| complex0.1 | [4, 8) | 1156 | 1.2e7 / 8.4e15 | 2.1e-2 / 4.3e-1 | 5.2e-7 / 1.2e-5 |
| complex0.1 | [8, 16) | 918 | 4.1e16 / 8.6e34 | 2.6e-1 / 3.1e2 | 2.9e-10 / 5.3e-8 |
| complex0.1 | [16, 32) | 102 | 4.4e31 / 6.5e50 | 5.4e1 / 1.6e5 | 1.0e-14 / 3.1e-11 |
| complex1 | [2, 4) | 714 | 6.2e2 / 2.2e6 | 1.1e-2 / 1.6e-1 | 3.8e-5 / 5.7e-4 |
| complex1 | [4, 8) | 1428 | 2.0e6 / 7.4e13 | 1.9e-2 / 7.0e-1 | 1.3e-6 / 7.7e-5 |
| complex1 | [8, 16) | 952 | 3.9e14 / 7.4e30 | 1.4e-1 / 1.2e2 | 1.7e-9 / 4.4e-7 |
| complex1 | [16, 32) | 204 | 1.2e27 / 1.6e45 | 2.3e1 / 3.8e4 | 2.9e-13 / 4.8e-10 |
| real | 0 | 7184 | 1.1e-2 / 3.1e-1 | 1.1e-2 / 3.1e-1 | 1.1e-2 / 3.1e-1 |

Sweep cost: 188.7 core-hours over 23188 classes run (the per-class wall times summed).
