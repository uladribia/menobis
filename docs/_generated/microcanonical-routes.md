<!-- GENERATED FILE. DO NOT EDIT BY HAND. -->

# Microcanonical sampling routes

> Generated from MENoBiS' capability registry (`menobis.capabilities`). The exactness and backend columns reflect the current source tree.

_Registry source SHA: `35cea70` · generated: 2026-08-31T16:24:01+00:00_

| Constraint | Family | Exact / controlled | Backend |
| Required arguments | Exactness |
|---|---|---|---|---|---|
| strength | ME | strengths exact | `microcanonical_fixed_strength` | strength_in, strength_out | exact stationary MCMC; strengths exact |
| strength_cost | ME | strengths exact; cost expected (gamma) | `microcanonical_fixed_strength_cost` | coord_x, coord_y, strength_in, strength_out, target_cost | exact stationary MCMC; strengths exact, cost matched in expectation |
| strength_edges | ME | strengths, E exact | `microcanonical_fixed_strength_edges` | strength_in, strength_out, target_edges | exact stationary MCMC; strengths and occupied-pair count E exact |
| strength_degree | ME | strengths, degree sequences exact | `microcanonical_fixed_strength_degree` | degree_in, degree_out, strength_in, strength_out | exact stationary MCMC; strengths and degree sequences exact |
| degree_events | ME | degree sequences, T exact | `microcanonical_fixed_kt` | degree_in, degree_out, total_events | exact stationary MCMC; degree sequences and total events T exact |
| edges_events | ME | E, T exact | `microcanonical_fixed_et` | node_count, target_edges, total_events | exact direct; occupied-pair count E and total events T exact |
| strength | B | strengths exact | `microcanonical_fixed_strength` | strength_in, strength_out | exact stationary MCMC; strengths exact |
| strength_cost | B | strengths exact; cost expected (gamma) | `microcanonical_fixed_strength_cost` | coord_x, coord_y, strength_in, strength_out, target_cost | exact stationary MCMC; strengths exact, cost matched in expectation |
| strength_edges | B | strengths, E exact | `microcanonical_fixed_strength_edges` | strength_in, strength_out, target_edges | exact stationary MCMC; strengths and occupied-pair count E exact |
| strength_degree | B | strengths, degree sequences exact | `microcanonical_fixed_strength_degree` | degree_in, degree_out, strength_in, strength_out | exact stationary MCMC; strengths and degree sequences exact |
| degree_events | B | degree sequences, T exact | `microcanonical_fixed_kt` | degree_in, degree_out, layers, total_events | exact stationary MCMC; degree sequences and total events T exact |
| edges_events | B | E, T exact | `microcanonical_fixed_et` | layers, node_count, target_edges, total_events | exact direct; occupied-pair count E and total events T exact |
| strength | W | strengths exact | `microcanonical_fixed_strength` | strength_in, strength_out | exact stationary MCMC; strengths exact |
| strength_cost | W | strengths exact; cost expected (gamma) | `microcanonical_fixed_strength_cost` | coord_x, coord_y, strength_in, strength_out, target_cost | exact stationary MCMC; strengths exact, cost matched in expectation |
| strength_edges | W | strengths, E exact | `microcanonical_fixed_strength_edges` | strength_in, strength_out, target_edges | exact stationary MCMC; strengths and occupied-pair count E exact |
| strength_degree | W | strengths, degree sequences exact | `microcanonical_fixed_strength_degree` | degree_in, degree_out, strength_in, strength_out | exact stationary MCMC; strengths and degree sequences exact |
| degree_events | W | degree sequences, T exact | `microcanonical_fixed_kt` | degree_in, degree_out, layers, total_events | exact stationary MCMC; degree sequences and total events T exact |
| edges_events | W | E, T exact | `microcanonical_fixed_et` | layers, node_count, target_edges, total_events | exact direct; occupied-pair count E and total events T exact |
