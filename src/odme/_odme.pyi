"""Type stubs for the ODME native extension."""


def rust_core_version() -> str: ...


def directed_strengths(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[int], list[int]]: ...


def directed_degrees(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[int], list[int]]: ...


def compute_all_node_stats(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]: ...


def weight_distribution(
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[int], list[int]]: ...


def fit_masked_binary_degrees(
    degree_out: list[float],
    degree_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def fit_masked_strength_degree_me(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], list[float], list[float], bool, int]: ...


def fit_masked_strength(
    strength_out: list[float],
    strength_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def fit_strength_cost(
    strength_out: list[float],
    strength_in: list[float],
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...


def fit_binary_degrees(
    degree_out: list[float],
    degree_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def fit_strength_edges_me(
    strength_out: list[float],
    strength_in: list[float],
    target_edges: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...


def fit_strength_degree_me(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], list[float], list[float], bool, int]: ...


def fit_weighted_factors(
    excess_out: list[float],
    excess_in: list[float],
    degree_x: list[float],
    degree_y: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def fit_balance_no_self_loops(
    s_out: list[float],
    s_in: list[float],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def sample_microcanonical(
    strength_out: list[int],
    strength_in: list[int],
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_custom_pij_events_poisson(
    sources: list[int],
    targets: list[int],
    probabilities: list[float],
    total_events: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_custom_pij_events_multinomial(
    sources: list[int],
    targets: list[int],
    probabilities: list[float],
    total_events: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_poisson_multinomial(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_strength_edges_me(
    x: list[float],
    y: list[float],
    lam: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_strength_cost_me(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_poisson(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_fixed_degree_events_me(
    x: list[float],
    y: list[float],
    total_events: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_strength_degree_me(
    degree_x: list[float],
    degree_y: list[float],
    excess_x: list[float],
    excess_y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_multinomial(
    x: list[float],
    y: list[float],
    total_events: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def filter_fixed_strength_poisson(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_fixed_strength_poisson(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_custom_poisson_rates(
    rate_sources: list[int],
    rate_targets: list[int],
    rates: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_custom_poisson_rates(
    rate_sources: list[int],
    rate_targets: list[int],
    rates: list[float],
    sources: list[int],
    targets: list[int],
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_strength_edges_zip(
    x: list[float],
    y: list[float],
    lam: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_strength_edges_zip(
    x: list[float],
    y: list[float],
    lam: float,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_strength_cost_poisson(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_strength_cost_poisson(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_strength_degree_zip(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_strength_degree_zip(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_degree_events_zip(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_degree_events_zip(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def benjamini_hochberg(pvalues: list[float], alpha: float) -> list[bool]: ...


def sample_geometric(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def sample_neg_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...


def fit_binomial_strength(
    strength_out: list[float],
    strength_in: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def fit_masked_binomial_strength(
    strength_out: list[float],
    strength_in: list[float],
    mask: list[bool],
    layers: int,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...


def filter_geometric(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_geometric(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def filter_neg_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...


def absent_neg_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...


def clustering_coefficients(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> list[float]: ...


def weighted_clustering_coefficients(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> list[float]: ...
