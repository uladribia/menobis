---
description: Primary thesis, core papers, complementary microcanonical references, and citation instructions for MENoBiS.
---

# References and thesis

## TL;DR

MENoBiS implements the theoretical framework of Oleguer Sagarra's doctoral
thesis on **non-binary maximum-entropy network ensembles**, extended with
exact microcanonical samplers. Cite the software (via `CITATION.cff`) and
the scientific references relevant to your use.

## Primary thesis

O. Sagarra, *Non-binary maximum entropy network ensembles and their
application to the study of urban mobility*, PhD thesis, 2015.

<https://hdl.handle.net/10803/400560>

This is the primary scientific reference for the model ontology: the
ME/B/W occupation families, the grand-canonical constraint framework, and
the urban-mobility applications.

## Core papers

| Topic | Reference |
|---|---|
| Multi-edge framework | O. Sagarra, C. J. Pérez Vicente, A. Díaz-Guilera, *Statistical mechanics of multiedge networks*, Phys. Rev. E 88, 062806. DOI: `10.1103/PhysRevE.88.062806` |
| Fixed-strength magnitudes | O. Sagarra, F. Font-Clos, C. J. Pérez-Vicente, A. Díaz-Guilera, *The configuration multi-edge model*, EPL 107, 38002. DOI: `10.1209/0295-5075/107/38002` |
| Urban mobility reconstruction | O. Sagarra, M. Szell, P. Santi, A. Díaz-Guilera, C. Ratti, *Supersampling and Network Reconstruction of Urban Mobility*, PLOS ONE. DOI: `10.1371/journal.pone.0134508` |
| Event nature and statistics | O. Sagarra, C. J. Pérez Vicente, A. Díaz-Guilera, *Role of adjacency-matrix degeneracy in maximum-entropy-weighted network models*, Phys. Rev. E 92, 052816. DOI: `10.1103/PhysRevE.92.052816` |

## Complementary microcanonical references

| Topic | Reference |
|---|---|
| Tailored graph ensembles | A. Annibale, A. C. C. Coolen, L. P. Fernandes, F. Fraternali, J. Kleinjung, *Tailored graph ensembles as proxies for biological network data*, J. Phys. A: Math. Theor. 42, 485001 (2009). DOI: `10.1088/1751-8113/42/48/485001` — foundational microcanonical null modelling for binary networks |

## Citation instructions

The repository ships a `CITATION.cff`; GitHub renders it through the
**Cite this repository** button. When publishing results obtained with
MENoBiS, cite the software and the scientific references above that ground
the models you used (at minimum the primary thesis, plus the papers matching
the families/constraints you employed).

## Historical terminology

The thesis and related literature call MENoBiS objects **weighted
networks** and speak of edge **weights**. Within the MENoBiS documentation
the terminology is *non-binary networks*, *occupation numbers*, and
*occupied pairs*; the historical wording is literature terminology, not the
current canonical vocabulary (see [Notation](notation.md)).

## Mapping to the current ontology

The thesis concepts map one-to-one onto the current model ontology, which is
documented in these pages rather than in a stale mapping table:

- families ME / B / W → [Event families](event-families.md);
- constraints → [Constraints](constraints.md);
- ensembles → [Ensembles](ensembles.md);
- filtering → [Filtering statistics](filtering.md);
- supported combinations → [Supported models](../guide/supported-models.md).