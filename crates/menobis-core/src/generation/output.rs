//! Shared generation output type.

/// Sparse network output from a generation run.
#[derive(Clone, Debug, Default)]
pub struct SampledNetwork {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub occ_nums: Vec<crate::OccNum>,
}

/// Concatenate parallel chunk results into one sampled network.
pub fn merge_samples(chunks: Vec<SampledNetwork>) -> SampledNetwork {
    let total_edges = chunks.iter().map(|chunk| chunk.sources.len()).sum();
    let mut result = SampledNetwork {
        sources: Vec::with_capacity(total_edges),
        targets: Vec::with_capacity(total_edges),
        occ_nums: Vec::with_capacity(total_edges),
    };
    for mut chunk in chunks {
        result.sources.append(&mut chunk.sources);
        result.targets.append(&mut chunk.targets);
        result.occ_nums.append(&mut chunk.occ_nums);
    }
    result
}
