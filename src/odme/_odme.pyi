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
