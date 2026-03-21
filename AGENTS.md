# AGENTS.md

## Repository expectations

- Keep changes minimal and targeted.
- Do not change public APIs, filenames, or build targets unless the task requires it.
- Prefer fixing build and include-path issues over rewriting working logic.

## Build and validation

- Primary validation command: `make all`
- After changing any `.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.mk`, or `Makefile`, run `make all`.
- If the build fails, report the exact failing command and error.

## Eigen handling

- This repo may run in environments where `Eigen/Dense` is not on the default compiler include path.
- Prefer an existing vendored Eigen copy if the repo already contains one, such as `third_party/eigen`.
- Otherwise preserve or use an include path equivalent to:
  - `CPPFLAGS += -I/usr/include/eigen3`
  - or `make all CPPFLAGS+='-I/usr/include/eigen3'`
- Do not remove Eigen usage just to make the build pass.
- If the environment itself is missing Eigen, note that Codex cloud setup must install `libeigen3-dev`.

## Makefile policy

- Prefer small Makefile fixes that keep both local builds and Codex builds working.
- Prefer a fallback pattern like:
  - use vendored Eigen if present
  - otherwise use `/usr/include/eigen3`

## Change review checklist

Before finishing:
- run `make all`
- keep diffs small
- avoid unrelated formatting churn
- summarize exactly what changed and why
