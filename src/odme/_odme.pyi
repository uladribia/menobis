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
def fit_masked_degree_bernoulli(
    degree_out: list[float],
    degree_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...
def fit_masked_strength_degree_poisson(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], list[float], list[float], bool, int]: ...
def fit_masked_strength_poisson(
    strength_out: list[float],
    strength_in: list[float],
    mask: list[bool],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...
def fit_strength_cost_poisson(
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
def fit_strength_cost_poisson_coordinates(
    strength_out: list[float],
    strength_in: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...
def fit_strength_cost_binomial_coordinates(
    strength_out: list[float],
    strength_in: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...
def fit_strength_cost_w_coordinates(
    strength_out: list[float],
    strength_in: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...
def fit_degree_bernoulli(
    degree_out: list[float],
    degree_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...
def fit_strength_edges_poisson(
    strength_out: list[float],
    strength_in: list[float],
    target_edges: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...
def fit_strength_degree_poisson(
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
def fit_strength_poisson_no_self_loops(
    s_out: list[float],
    s_in: list[float],
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...
def sample_strength_stub_matching(
    strength_out: list[int],
    strength_in: list[int],
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_custom_poisson(
    sources: list[int],
    targets: list[int],
    probabilities: list[float],
    total_events: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_custom_multinomial(
    sources: list[int],
    targets: list[int],
    probabilities: list[float],
    total_events: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_poisson_multinomial(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_edges_poisson(
    x: list[float],
    y: list[float],
    lam: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_cost_poisson(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_poisson(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_degree_events_poisson(
    x: list[float],
    y: list[float],
    total_events: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_degree_poisson(
    degree_x: list[float],
    degree_y: list[float],
    excess_x: list[float],
    excess_y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_multinomial(
    x: list[float],
    y: list[float],
    total_events: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_strength_poisson(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_poisson(
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
def filter_custom_poisson(
    rate_sources: list[int],
    rate_targets: list[int],
    rates: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_custom_poisson(
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
def filter_strength_edges_poisson(
    x: list[float],
    y: list[float],
    lam: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_edges_poisson(
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
def filter_strength_degree_poisson(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_degree_poisson(
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
def filter_degree_events_poisson(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_degree_events_poisson(
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
def sample_strength_geometric(
    x: list[float],
    y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_negative_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def fit_strength_binomial(
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
def filter_strength_geometric(
    x: list[float],
    y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_geometric(
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
def filter_strength_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_binomial(
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
def filter_strength_negative_binomial(
    x: list[float],
    y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_negative_binomial(
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
def sample_strength_cost_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_edges_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_degree_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_degree_events_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_strength_cost_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def filter_strength_edges_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_edges_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def filter_strength_degree_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_degree_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def filter_degree_events_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_degree_events_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...

# Additional native APIs exposed by the current Rust extension.
def fit_strength_poisson(
    strength_out: list[float],
    strength_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], bool, int]: ...
def fit_strength_geometric(
    strength_out: list[float],
    strength_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    int,
    str,
    float,
    int,
    float,
    float,
    float,
    float,
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_negative_binomial(
    strength_out: list[float],
    strength_in: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    int,
    str,
    float,
    int,
    float,
    float,
    float,
    float,
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_degree_geometric(
    s_out: list[float],
    s_in: list[float],
    k_out: list[float],
    k_in: list[float],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_degree_negative_binomial(
    s_out: list[float],
    s_in: list[float],
    k_out: list[float],
    k_in: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_edges_geometric(
    strength_out: list[float],
    strength_in: list[float],
    target_edges: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    float,
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_edges_negative_binomial(
    strength_out: list[float],
    strength_in: list[float],
    target_edges: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    float,
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_cost_geometric(
    strength_out: list[float],
    strength_in: list[float],
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    float,
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_strength_cost_negative_binomial(
    strength_out: list[float],
    strength_in: list[float],
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    target_cost: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    list[float],
    list[float],
    float,
    int,
    str,
    float,
    int,
    tuple[float, float, float, float, float],
    tuple[int, int, int, int, int, int],
]: ...
def fit_degree_events_geometric(
    degree_out: list[float],
    degree_in: list[float],
    total_events: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, float, bool, int]: ...
def fit_degree_events_negative_binomial(
    degree_out: list[float],
    degree_in: list[float],
    total_events: int,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, float, bool, int]: ...
def sample_strength_cost_geometric(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_cost_negative_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_edges_geometric(
    x: list[float],
    y: list[float],
    lam: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_edges_negative_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_degree_geometric(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_degree_negative_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_degree_events_geometric(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_degree_events_negative_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_strength_cost_geometric(
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
def absent_strength_cost_geometric(
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
def filter_strength_cost_negative_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_negative_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def filter_strength_edges_geometric(
    x: list[float],
    y: list[float],
    lam: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_strength_edges_negative_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_strength_degree_geometric(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_strength_degree_negative_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_degree_events_geometric(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_degree_events_negative_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def fit_partial_strength_poisson_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_weights: list[int],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_degree_poisson_full(
    degree_out: list[float],
    degree_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_strength_degree_poisson_full(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_weights: list[int],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_strength_edges_poisson_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_weights: list[float],
    target_edges: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_strength_cost_poisson_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_weights: list[float],
    cost_sources: list[int],
    cost_targets: list[int],
    cost_values: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_cost_poisson_coordinates_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_weights: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def absent_strength_edges_geometric(
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
def absent_strength_edges_negative_binomial(
    x: list[float],
    y: list[float],
    lam: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def absent_strength_degree_geometric(
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
def absent_strength_degree_negative_binomial(
    x: list[float],
    y: list[float],
    z: list[float],
    w: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def absent_degree_events_geometric(
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
def absent_degree_events_negative_binomial(
    x: list[float],
    y: list[float],
    positive_weight_rate: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
