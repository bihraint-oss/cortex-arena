# Security policy

## Supported version

Security fixes currently target the latest `0.1.x` release on `main`.

## Reporting

Please use GitHub's private vulnerability reporting for issues that could cause command execution, unsafe archive contents, dependency compromise, or control outside the intended Lux process.

## Trust boundary

Cortex Arena launches the official `luxai-s3` runner and two local Python agent processes. Treat third-party opponent agents as untrusted code and run them in a container or separate account. The bundled submission builder includes only the project's Python sources and license notices; it does not download code or follow symlinks.
