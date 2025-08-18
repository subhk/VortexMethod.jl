# Changelog

All notable changes to this project will be documented in this file.

## 2.0.0-DEV

- Breaking: Rename `dom` to `domain` everywhere in the public API and codebase.
  - Function parameters now use `domain::DomainSpec` instead of `dom::DomainSpec`.
  - Keyword arguments `; dom=...` are now `; domain=...`.
  - Checkpoint loader NamedTuple field `dom` is now `domain`.
  - Examples, tests, README, and docs updated accordingly.
- Rationale: Improve readability and consistency; avoid ambiguity with unrelated substrings.
- Migration:
  - Replace all usage of `dom` parameter/keyword with `domain`.
  - When accessing checkpoint snapshots, use `ck.domain` and `snap.domain` instead of `ck.dom`.
  - Update helper calls such as `grid_vectors(domain, gr)`, `wrap_point(..., domain)`, `rk2_step!(..., domain, gr, ...)`.

## 1.x (pre-release)

- Initial development versions.
