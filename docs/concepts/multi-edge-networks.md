---
description: Multi-edge networks represent integer-valued weighted directed graphs.
---

# Multi-edge networks

## TL;DR

A multi-edge (ME) network is a directed weighted graph where each edge weight
$t_{ij}$ is a non-negative integer representing the number of distinguishable
events (trips, interactions, messages) from node $i$ to node $j$.

## Definitions

Given $N$ nodes, the network is fully described by the **occupation matrix**
$\mathbf{T}$ with entries $t_{ij} \in \{0, 1, 2, \ldots\}$.

Key quantities:

| Symbol | Name | Definition |
|--------|------|------------|
| $T$ | Total events | $T = \sum_{ij} t_{ij}$ |
| $s_i^{out}$ | Outgoing strength | $s_i^{out} = \sum_j t_{ij}$ |
| $s_j^{in}$ | Incoming strength | $s_j^{in} = \sum_i t_{ij}$ |
| $k_i^{out}$ | Outgoing degree | $k_i^{out} = \sum_j \Theta(t_{ij})$ |
| $k_j^{in}$ | Incoming degree | $k_j^{in} = \sum_i \Theta(t_{ij})$ |
| $E$ | Binary edges | $E = \sum_{ij} \Theta(t_{ij})$ |

where $\Theta(x) = 1$ if $x > 0$, else $0$.

## Strength-degree invariant

Since each present edge contributes at least one event:

$$
s_i^{out} \ge k_i^{out}, \quad s_j^{in} \ge k_j^{in}.
$$

MENoBiS enforces this at the data boundary by rejecting fractional weights and
validating $s \ge k$ before coupled model fitting.

## Higher-order statistics

| Statistic | Formula |
|-----------|---------|
| Y2 disparity | $Y_{2,i}^{out} = \sum_j t_{ij}^2 / (s_i^{out})^2$ |
| Weighted k-nn | $k_{nn,i}^{w,out} = \sum_j t_{ij} k_j^{in} / s_i^{out}$ |
| Weighted s-nn | $s_{nn,i}^{w,out} = \sum_j t_{ij} s_j^{in} / s_i^{out}$ |
| Weight distribution | $P(w) = $ fraction of edges with weight $w$ |

These are computed in a single Rust pass via `compute_all_stats`.
