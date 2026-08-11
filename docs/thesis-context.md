---
description: Scientific references and citation metadata for MENoBiS.
---

# Thesis and citations

## TL;DR

MENoBiS implements the maximum-entropy framework for non-binary network
ensembles: distinguishable events (ME), aggregated binary layers (B), and
indistinguishable events (W).

!!! note "Primary thesis"
    The thesis title is **Non-binary maximum entropy network ensembles and their
    application to the study of urban mobility**. Handle:
    <https://hdl.handle.net/10803/400560>.

## CITATION.cff content

| Field | Value |
|---|---|
| `cff-version` | `1.2.0` |
| `title` | `MENoBiS: Max Entropy NOn Binary Suite` |
| `type` | `software` |
| `version` | `1.0.1` |
| `license` | `GPL-3.0-or-later` |
| `repository-code` | `https://github.com/uladribia/menobis` |
| `url` | `https://uladribia.github.io/menobis/` |
| `message` | `If you use MENoBiS, please cite the software and the scientific references below.` |

Abstract from `CITATION.cff`:

> MENoBiS is a Rust and Python library for fitting, sampling, filtering, and
> analysing maximum-entropy null models of directed non-binary networks with
> non-negative integer node-pair occupations.

GitHub renders this metadata through the **Cite this repository** button.

## Core references

| Topic | Reference |
|---|---|
| Thesis | O. Sagarra, *Non-binary maximum entropy network ensembles and their application to the study of urban mobility*, PhD thesis, 2015. <https://hdl.handle.net/10803/400560> |
| Multi-edge framework | O. Sagarra, C. J. Pérez Vicente, A. Díaz-Guilera, *Statistical mechanics of multiedge networks*, Phys. Rev. E 88, 062806. DOI: `10.1103/PhysRevE.88.062806` |
| Fixed-strength magnitudes | O. Sagarra, F. Font-Clos, C. J. Pérez-Vicente, A. Díaz-Guilera, *The configuration multi-edge model*, EPL 107, 38002. DOI: `10.1209/0295-5075/107/38002` |
| Urban mobility reconstruction | O. Sagarra, M. Szell, P. Santi, A. Díaz-Guilera, C. Ratti, *Supersampling and Network Reconstruction of Urban Mobility*, PLOS ONE. DOI: `10.1371/journal.pone.0134508` |
| Same constraints with different event nature leads to different statistics | O. Sagarra, C. J. Pérez Vicente, A. Díaz-Guilera, *Role of adjacency-matrix degeneracy in maximum-entropy-weighted network models*, Phys. Rev. E 92, 052816. DOI: `10.1103/PhysRevE.92.052816` |

## Complementary references

These works establish the microcanonical ensemble framework that MENoBiS extends
to non-binary (weighted) networks with ME, B, and W families.

| Topic | Reference |
|---|---|
| **PRIMARY** – Non-binary max-entropy (Sagarra thesis) | See above — the primary scientific reference. |
| **COMPLEMENTARY** – Microcanonical tailored graph ensembles | A. Annibale, A. C. C. Coolen, L. P. Fernandes, F. Fraternali, J. Kleinjung, *Tailored graph ensembles as proxies for biological network data*, J. Phys. A: Math. Theor. 42, 485001 (2009). DOI: `10.1088/1751-8113/42/48/485001` — Foundational work on microcanonical ensemble null models for binary networks. |
| **COMPLEMENTARY** – Monograph on network ensemble theory | A. C. C. Coolen, A. Annibale, E. Roberts, *Generating Random Networks and Bipartite Graphs*, Oxford University Press (2017). ISBN 978-0-19-879016-9 — Comprehensive treatment of microcanonical sampling, backbone swaps, and ensemble equivalence. |

!!! note "MENoBiS extension"
    The Coolen framework targets **binary** networks (edge-presence ensembles).
    MENoBiS generalises this to **non-binary** occupation numbers with three
    distinct event natures (ME, B, W), each carrying a different per-pair
    degeneracy. The microcanonical sampling backends (4-cycle Metropolis
    for fixed strengths, pair-Gibbs for fixed totals) are the non-binary
    analogue of Coolen-style backbone swaps.

## Mapping to MENoBiS names

| Thesis idea | MENoBiS name |
|---|---|
| distinguishable events | `ModelFamily.ME` |
| aggregated binary layers | `ModelFamily.B` with `layers=M` |
| indistinguishable events | `ModelFamily.W` with `layers=M` |
| grand-canonical independent pairs | `Ensemble.GRAND_CANONICAL` |
| fixed total ME events | `Ensemble.CANONICAL` |
| fixed ME strength stubs | `Ensemble.MICROCANONICAL` |

See [Equations](concepts/equations.md) for the compact formula map.
