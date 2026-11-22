//! F16×F32 NEON (aarch64) matmul kernel.
//!
//! Uses aarch64's native f16 NEON support (`float16x8_t` + `vcvt_f32_f16`)
//! to convert 8 f16 weight lanes to 8 f32 lanes per iteration, then FMA
//! against the f32 input. 4-wide per inner loop, hsum via `vaddvq_f32`
//! at row end.
//!
//! TODO-005: validate against the scalar reference in `super::scalar`
//! once a CI target exposes aarch64.

#![cfg(target_arch = "aarch64")]

#[target_feature(enable = "neon")]
pub unsafe fn matmul_f16_vs_f32_neon(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    let weight_ptr = weight.as_ptr();
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for (output_index, row) in (row_start..row_end).enumerate() {
        let row_byte = row * n_in * 2;
        let mut sum = vdupq_n_f32(0.0);
        let mut index = 0;

        while index + 8 <= n_in {
            // Load 8 f16 weight lanes (16 bytes).
            let f16_bits = vld1q_u16(weight_ptr.add(row_byte + index * 2).cast());
            // Reinterpret as f16x8 and convert to f32x8.
            let f16_lanes: float16x8_t = vreinterpretq_f16_u16(f16_bits);
            let input_lanes = vld1q_f32(input_ptr.add(index));
            let low_values = vcvt_f32_f16(vget_low_f16(f16_lanes));
            sum = vfmaq_f32(sum, low_values, input_lanes);
            let high_values = vcvt_f32_f16(vget_high_f16(f16_lanes));
            sum = vfmaq_f32(sum, high_values, vld1q_f32(input_ptr.add(index + 4)));
            index += 8;
        }
        while index + 4 <= n_in {
            let f16_bits = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
            let f16_lanes: float16x4_t = vreinterpret_f16_u16(f16_bits);
            let values = vcvt_f32_f16(f16_lanes);
            sum = vfmaq_f32(sum, values, vld1q_f32(input_ptr.add(index)));
            index += 4;
        }

        let mut total = vaddvq_f32(sum);
        while index < n_in {
            let offset = row_byte + index * 2;
            let bits = u16::from_le_bytes([*weight_ptr.add(offset), *weight_ptr.add(offset + 1)]);
            total += crate::ops::f16_to_f32(bits) * *input_ptr.add(index);
            index += 1;
        }
        *output_ptr.add(output_index) = total;
    }
}

/// Batched F16×F32 matmul that reuses each loaded weight vector for four
/// input rows. The per-row vector accumulation order matches the single-row
/// kernel, so this only removes repeated weight loads.
#[target_feature(enable = "neon")]
pub unsafe fn matmul_f16_vs_f32_neon_batched(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(weight.len(), n_in * n_out * 2);
    debug_assert_eq!(input.len(), n_tokens * n_in);
    debug_assert_eq!(output.len(), n_tokens * n_out);

    let weight_ptr = weight.as_ptr();
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for row in 0..n_out {
        let row_byte = row * n_in * 2;
        for token_group in (0..n_tokens).step_by(4) {
            let group_len = (n_tokens - token_group).min(4);
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut index = 0;

            while index + 4 <= n_in {
                let f16_bits = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
                let values = vcvt_f32_f16(vreinterpret_f16_u16(f16_bits));
                for token in 0..group_len {
                    let input_row = input_ptr.add((token_group + token) * n_in + index);
                    acc[token] = vfmaq_f32(acc[token], values, vld1q_f32(input_row));
                }
                index += 4;
            }

            let mut totals = [0.0f32; 4];
            for token in 0..group_len {
                totals[token] = vaddvq_f32(acc[token]);
            }
            while index < n_in {
                let offset = row_byte + index * 2;
                let bits =
                    u16::from_le_bytes([*weight_ptr.add(offset), *weight_ptr.add(offset + 1)]);
                let value = crate::ops::f16_to_f32(bits);
                for token in 0..group_len {
                    totals[token] += value * *input_ptr.add((token_group + token) * n_in + index);
                }
                index += 1;
            }

            for token in 0..group_len {
                *output_ptr.add((token_group + token) * n_out + row) = totals[token];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{matmul_f16_vs_f32_neon, matmul_f16_vs_f32_neon_batched};
    use crate::ops::kernel::f16::scalar::forward_f16_rows;

    #[test]
    fn neon_matches_scalar_with_tail() {
        let n_in = 13;
        let n_out = 3;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32 * 0.07).sin()).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let mut actual = vec![0.0; n_out];
        let mut expected = vec![0.0; n_out];

        unsafe {
            matmul_f16_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, n_out);
        }
        forward_f16_rows(&weight, &input, &mut expected, n_in, n_out, 0, 1);

        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() < 1e-3, "{a} != {e}");
        }
    }

    #[test]
    fn neon_includes_high_half_of_eight_lane_chunk() {
        let n_in = 8;
        let weight: Vec<u8> = (0..n_in)
            .map(|index| crate::ops::f32_to_f16(if index < 4 { 0.0 } else { 1.0 }))
            .flat_map(u16::to_le_bytes)
            .collect();
        let input = vec![1.0; n_in];
        let mut actual = [0.0];
        let mut expected = [0.0];

        unsafe {
            matmul_f16_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, 1);
        }
        forward_f16_rows(&weight, &input, &mut expected, n_in, 1, 0, 1);

        assert_eq!(actual[0], 4.0);
        assert_eq!(actual[0], expected[0]);
    }

    #[test]
    fn neon_batched_matches_single_rows_bitwise() {
        let n_tokens = 4;
        let n_in = 13;
        let n_out = 3;
        let weight: Vec<u8> = (0..n_in * n_out)
            .map(|index| crate::f32_to_f16((index as f32 * 0.07).sin()))
            .flat_map(u16::to_le_bytes)
            .collect();
        let input: Vec<f32> = (0..n_tokens * n_in)
            .map(|index| (((index * 29 + 7) % 73) as f32 - 36.0) * 0.017)
            .collect();
        let mut actual = vec![0.0f32; n_tokens * n_out];
        let mut expected = vec![0.0f32; n_tokens * n_out];

        unsafe {
            matmul_f16_vs_f32_neon_batched(&weight, &input, &mut actual, n_in, n_out, n_tokens);
            for token in 0..n_tokens {
                matmul_f16_vs_f32_neon(
                    &weight,
                    &input[token * n_in..(token + 1) * n_in],
                    &mut expected[token * n_out..(token + 1) * n_out],
                    n_in,
                    0,
                    n_out,
                );
            }
        }

        assert_eq!(actual, expected);
    }
}
