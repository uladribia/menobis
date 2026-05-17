//! Candidate pair providers shared by generation, filtering, and future stats.

use crate::distribution::{
    strength_degree_distribution, strength_edges_distribution, PairDistribution,
};
use std::collections::HashMap;

pub const PARALLEL_PAIR_THRESHOLD: usize = 1_000_000;
pub const ROW_CHUNK_SIZE: usize = 128;
pub const SPARSE_CHUNK_SIZE: usize = 65_536;

/// Candidate-pair support for a null model.
pub enum CandidateSupport<'a> {
    AllPairs {
        node_count: usize,
        self_loops: bool,
    },
    SparsePairs {
        sources: &'a [u64],
        targets: &'a [u64],
    },
}

/// Provider of pair-level null distributions.
pub trait PairDistributionProvider: Sync {
    fn support(&self) -> CandidateSupport<'_>;
    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution>;

    fn distribution_at(
        &self,
        _index: usize,
        source: usize,
        target: usize,
    ) -> Option<PairDistribution> {
        self.distribution(source, target)
    }
}

/// Deterministic seed mixer for parallel chunks.
#[must_use]
pub fn chunk_seed(seed: u64, chunk_index: usize) -> u64 {
    splitmix64(seed ^ ((chunk_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)))
}

#[must_use]
pub fn row_ranges(node_count: usize) -> Vec<(usize, usize)> {
    (0..node_count)
        .step_by(ROW_CHUNK_SIZE)
        .map(|start| (start, (start + ROW_CHUNK_SIZE).min(node_count)))
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct FixedStrengthPoissonProvider<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub self_loops: bool,
}

impl PairDistributionProvider for FixedStrengthPoissonProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::AllPairs {
            node_count: self.x.len(),
            self_loops: self.self_loops,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        if !self.self_loops && source == target {
            return None;
        }
        let rate = self.x[source] * self.y[target];
        (rate > 0.0).then_some(PairDistribution::Poisson { rate })
    }
}

pub struct StrengthCostPoissonProvider<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub gamma: f64,
    pub costs: &'a HashMap<(usize, usize), f64>,
    pub self_loops: bool,
}

impl PairDistributionProvider for StrengthCostPoissonProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::AllPairs {
            node_count: self.x.len(),
            self_loops: self.self_loops,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        if !self.self_loops && source == target {
            return None;
        }
        let rate = self.x[source]
            * self.y[target]
            * (-self.gamma * self.costs.get(&(source, target)).copied().unwrap_or(0.0)).exp();
        (rate > 0.0).then_some(PairDistribution::Poisson { rate })
    }
}

pub struct SparsePoissonRateProvider<'a> {
    pub sources: &'a [u64],
    pub targets: &'a [u64],
    pub rates: &'a [f64],
}

impl PairDistributionProvider for SparsePoissonRateProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::SparsePairs {
            sources: self.sources,
            targets: self.targets,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        self.sources
            .iter()
            .zip(self.targets.iter())
            .zip(self.rates.iter())
            .find_map(|((&src, &tgt), &rate)| {
                (src as usize == source && tgt as usize == target && rate > 0.0)
                    .then_some(PairDistribution::Poisson { rate })
            })
    }

    fn distribution_at(
        &self,
        index: usize,
        source: usize,
        target: usize,
    ) -> Option<PairDistribution> {
        if self.sources.get(index).copied()? as usize != source
            || self.targets.get(index).copied()? as usize != target
        {
            return None;
        }
        self.rates
            .get(index)
            .copied()
            .filter(|&rate| rate > 0.0)
            .map(|rate| PairDistribution::Poisson { rate })
    }
}

pub struct NormalizedSparsePoissonProvider<'a> {
    pub sources: &'a [u64],
    pub targets: &'a [u64],
    pub probabilities: &'a [f64],
    pub total_events: u64,
    pub probability_sum: f64,
}

impl PairDistributionProvider for NormalizedSparsePoissonProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::SparsePairs {
            sources: self.sources,
            targets: self.targets,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        self.sources
            .iter()
            .zip(self.targets.iter())
            .zip(self.probabilities.iter())
            .find_map(|((&src, &tgt), &probability)| {
                (src as usize == source && tgt as usize == target && probability > 0.0).then_some(
                    PairDistribution::Poisson {
                        rate: self.total_events as f64 * probability / self.probability_sum,
                    },
                )
            })
    }

    fn distribution_at(
        &self,
        index: usize,
        source: usize,
        target: usize,
    ) -> Option<PairDistribution> {
        if self.probability_sum <= 0.0
            || self.sources.get(index).copied()? as usize != source
            || self.targets.get(index).copied()? as usize != target
        {
            return None;
        }
        self.probabilities
            .get(index)
            .copied()
            .filter(|&probability| probability > 0.0)
            .map(|probability| PairDistribution::Poisson {
                rate: self.total_events as f64 * probability / self.probability_sum,
            })
    }
}

pub struct SparsePoissonRateMapProvider<'a> {
    pub sources: &'a [u64],
    pub targets: &'a [u64],
    pub map: &'a HashMap<(usize, usize), f64>,
}

impl PairDistributionProvider for SparsePoissonRateMapProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::SparsePairs {
            sources: self.sources,
            targets: self.targets,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        self.map
            .get(&(source, target))
            .copied()
            .filter(|&rate| rate > 0.0)
            .map(|rate| PairDistribution::Poisson { rate })
    }
}

pub struct DegreeEventsZipProvider<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub positive_weight_rate: f64,
    pub self_loops: bool,
}

impl PairDistributionProvider for DegreeEventsZipProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::AllPairs {
            node_count: self.x.len(),
            self_loops: self.self_loops,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        if !self.self_loops && source == target {
            return None;
        }
        let z = self.x[source] * self.y[target];
        let occupation = z / (1.0 + z);
        Some(PairDistribution::ZipPoisson {
            occupation,
            rate: self.positive_weight_rate,
        })
    }
}

pub struct StrengthEdgesZipProvider<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub lambda: f64,
    pub self_loops: bool,
}

impl PairDistributionProvider for StrengthEdgesZipProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::AllPairs {
            node_count: self.x.len(),
            self_loops: self.self_loops,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        if !self.self_loops && source == target {
            return None;
        }
        Some(strength_edges_distribution(
            self.x[source],
            self.y[target],
            self.lambda,
        ))
    }
}

pub struct StrengthDegreeZipProvider<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub z: &'a [f64],
    pub w: &'a [f64],
    pub self_loops: bool,
}

impl PairDistributionProvider for StrengthDegreeZipProvider<'_> {
    fn support(&self) -> CandidateSupport<'_> {
        CandidateSupport::AllPairs {
            node_count: self.x.len(),
            self_loops: self.self_loops,
        }
    }

    fn distribution(&self, source: usize, target: usize) -> Option<PairDistribution> {
        if !self.self_loops && source == target {
            return None;
        }
        Some(strength_degree_distribution(
            self.x[source],
            self.y[target],
            self.z[source],
            self.w[target],
        ))
    }
}
