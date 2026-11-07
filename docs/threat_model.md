# Threat model and privacy controls

## Assets, actors and trust boundaries

- **Assets:** CSI recordings, participant metadata, model checkpoints, API keys, inference results, cloud logs, artifact registries and AWS permissions.
- **Actors:** authorized developer/researcher, end-user operator, accidental public internet client, compromised deployment credential, malicious file/data contributor, network attacker, and inappropriate internal dataset consumer.
- **Trust boundaries:** dataset root into Python file loader; manifest path into filesystem; checkpoint into torch deserializer; public HTTPS ALB into private ECS task; secret manager into task environment; inference response into external systems.

## Implemented safeguards

1. NPZ files are loaded without pickle and must contain only the declared two arrays. Manifest paths are confined to the intended data root with resolved-path checking.
2. Input dimensions, finite values, channel bounds and timestamps are verified before training/inference. The API bounds single/batch window sizes and checks a declared HTTP content-length limit.
3. Predictor artifacts use `torch.load(..., weights_only=True)` and refuse contract mismatches.
4. Production mode fails startup unless an API key exists; comparisons use a timing-safe equality function.
5. The API does not log raw CSI request bodies. Docker Compose binds to `127.0.0.1`, uses a non-root UID, read-only filesystem and removes Linux capabilities.
6. Terraform uses an ACM-backed TLS ALB and Fargate private subnets, restricted container ingress, a scoped task-execution Secrets Manager permission, and an immutable container digest.

## Important residual risks — not solved

- **API key is not complete identity/auth authorization:** keys can be shared, leaked, replayed or stolen from environment variables. Implement authentication gateway, quotas, per-client scopes, revocation and auditing before public availability.
- **HTTP body memory budget:** application checks declared Content-Length but untrusted chunked/streamed bodies require actual-byte caps at an upstream gateway/proxy. Add WAF/rate limiting, bounded queues and load tests.
- **Model integrity:** SHA-256 identifies a checkpoint but does not sign it. Adopt signed artifacts, a trusted release registry, image provenance and approved training sources.
- **Channel semantics:** wrong channel mapping or a silent unit change can yield confidently wrong predictions even if shape checks pass. Add source-device/channel-order fingerprints.
- **Privacy:** CSI can disclose sensitive behavior and occupancy. Ensure consent, GDPR/data protection assessment where applicable, minimization, retention, access logs and deletion procedures. No raw individual traces are shipped.
- **Cloud metadata/secret risks:** ECS environment variables are visible to application processes and operators with suitable access; rotating a Secrets Manager value requires restarting tasks to pick up the new version. Configure KMS permissions as needed when using customer-managed CMKs.
- **Adversarial/robustness threats:** evasion, poisoning, physical interference and distribution shifts are not systematically evaluated.
- **Native process limits:** C++ CLI rejects malformed/oversized individual rows but is not a sandbox. Treat unknown files as untrusted and isolate workflows in CI/containers.

## Release gate checklist

- [ ] Threat model reviewed and approved by deployment owner.
- [ ] Dataset ownership/consent verified.
- [ ] Model trained and evaluated on approved real data.
- [ ] API rate/body limits tested under load.
- [ ] Certificate/DNS and AWS network isolation validated.
- [ ] Secret rotation and emergency revocation exercised.
- [ ] Dependency licenses and vulnerabilities audited.
- [ ] Operational alerting, data retention and incident response added.
