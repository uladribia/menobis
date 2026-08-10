"""Type stubs for the MENoBiS native extension."""

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
def occupation_distribution(
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
def fit_strength_edges_binomial(
    strength_out: list[float],
    strength_in: list[float],
    target_edges: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], list[float], float, bool, int]: ...
def fit_strength_degree_binomial(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    layers: int,
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
def sample_b_fixed_et(
    node_count: int,
    self_loops: bool,
    layers: int,
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_b_fixed_et_explicit(
    admissible_sources: list[int],
    admissible_targets: list[int],
    layers: int,
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_me_fixed_et(

    node_count: int,
    self_loops: bool,
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_me_fixed_et_explicit(
    admissible_sources: list[int],
    admissible_targets: list[int],
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_w_fixed_et(
    node_count: int,
    self_loops: bool,
    layers: int,
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_w_fixed_et_explicit(
    admissible_sources: list[int],
    admissible_targets: list[int],
    layers: int,
    residual_edges: int,
    residual_total: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_fixed_strength(
    family: str,
    strength_out: list[int],
    strength_in: list[int],
    self_loops: bool,
    fixed_sources: list[int],
    fixed_targets: list[int],
    fixed_occnums: list[int],
    layers: int,
    burn_in_sweeps: int,
    sweeps_per_sample: int,
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
def sample_strength_edges_poisson(
    x: list[float],
    y: list[float],
    lam: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_cost_poisson_coordinates(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
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
    coord_x: list[float],
    coord_y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_poisson(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_degree_events_poisson(
    x: list[float],
    y: list[float],
    positive_intensity: float,
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
def occupation_clustering_coefficients(
    node_count: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> list[float]: ...
def sample_strength_cost_binomial_coordinates(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_strength_cost_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_degree_events_binomial(
    x: list[float],
    y: list[float],
    positive_intensity: float,
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
def sample_strength_cost_geometric_coordinates(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_strength_cost_negative_binomial_coordinates(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def sample_degree_events_negative_binomial(
    x: list[float],
    y: list[float],
    positive_intensity: float,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_strength_cost_geometric(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_geometric(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    coord_x: list[float],
    coord_y: list[float],
    layers: int,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_strength_cost_negative_binomial(
    x: list[float],
    y: list[float],
    gamma: float,
    coord_x: list[float],
    coord_y: list[float],
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
    positive_intensity: float,
    sources: list[int],
    targets: list[int],
    weights: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def filter_degree_events_negative_binomial(
    x: list[float],
    y: list[float],
    positive_intensity: float,
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
    known_occnums: list[int],
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
    known_occnums: list[int],
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_strength_edges_poisson_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_occnums: list[float],
    target_edges: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...
def fit_partial_strength_cost_poisson_coordinates_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_occnums: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_cost_binomial_coordinates_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_occnums: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_cost_w_coordinates_full(
    strength_out: list[float],
    strength_in: list[float],
    known_sources: list[int],
    known_targets: list[int],
    known_occnums: list[float],
    coord_x: list[float],
    coord_y: list[float],
    target_cost: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_binomial_full(
    strength_out: list[float],
    strength_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_edges_binomial_full(
    strength_out: list[float],
    strength_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    target_edges: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_degree_binomial_full(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_edges_w_full(
    strength_out: list[float],
    strength_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    target_edges: float,
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_w_full(
    strength_out: list[float],
    strength_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[int], list[int], list[float], bool, int]: ...

def fit_partial_strength_degree_w_full(
    strength_out: list[float],
    strength_in: list[float],
    degree_out: list[float],
    degree_in: list[float],
    known_src: list[int],
    known_tgt: list[int],
    known_occnum: list[float],
    layers: int,
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
    positive_intensity: float,
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
    positive_intensity: float,
    layers: int,
    sources: list[int],
    targets: list[int],
    self_loops: bool,
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
def fit_edges_events(
    family: str,
    total_edges: float,
    total_events: int,
    n_pairs: int,
    layers: int,
    max_iterations: int,
) -> tuple[float, float, float, float, bool, int]: ...
def sample_edges_events(
    node_count: int,
    q: float,
    occupation: float,
    family: str,
    layers: int,
    self_loops: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]: ...
def filter_edges_events(
    node_count: int,
    q: float,
    occupation: float,
    family: str,
    layers: int,
    self_loops: bool,
    sources: list[int],
    targets: list[int],
    occ_nums: list[int],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def absent_edges_events(
    node_count: int,
    q: float,
    occupation: float,
    family: str,
    layers: int,
    self_loops: bool,
    sources: list[int],
    targets: list[int],
    alpha_lower: float,
    min_occupation: float,
    min_expected: float,
    max_absent: int | None,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]: ...
