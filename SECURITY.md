# Security policy

Keep is pre-release software and has no supported versions yet. We still
want to hear about security problems.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability
reporting ("Report a vulnerability" on the Security tab) for this repository.
Include the affected version or commit, steps to reproduce and the impact you
expect. We aim to acknowledge reports within five working days.

## Scope notes

- The credentials in `compose.yaml` are for local development only and bind to
  loopback. Do not reuse them anywhere else.
- API keys for model providers are read from environment variables named in a
  manifest. They are never written to run artifacts; report any case where
  one is.
- Benchmark data and run artifacts may contain conversation text. They are
  gitignored (`data/`, `results/`); keep them out of commits and issues.
