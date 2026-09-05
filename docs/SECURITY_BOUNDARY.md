# CB16 Remote CI Relay – Security Boundary

| Entity | Holds |
|---|---|
| GitHub | source code + sanitized CI evidence |
| Cloudflare | ingress/auth boundary; service token |
| Japan OCI | orchestration, secrets, full logs |
| Shanxi | compute, datasets, weights |

- Model weights never leave Shanxi.
- OCI CI port only listens on `127.0.0.1`.
- `/webhook/github` HMAC-validated.
- `/api/v1/worker/*` requires Cloudflare Service Token + CB16 bearer.
- No secrets, IPs, or model artifacts are committed to this repository.
