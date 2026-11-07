# Optional AWS deployment playbook — **nothing is provisioned by this repository**

## Deployment design and cost warning

This is an **account- and release-specific reference** for an inference service, not evidence that CrossGuard-Fi has been deployed or tested in AWS. It creates real billable infrastructure only if an operator deliberately runs AWS and Terraform commands. In particular, **ALB, Fargate, NAT gateways/VPC endpoints, CloudWatch Logs, Secrets Manager and ECR storage are not generally free**. Use local Docker for portfolio verification; do not run Terraform just to demonstrate the project in interviews.

Target path:

```text
External HTTPS client
       |
ACM certificate + public Application Load Balancer (:443)
       |
Security-group-scoped traffic to private ECS/Fargate task (:8000)
       |
CGF_ENV=production, CGF_MODEL_PATH=/opt/crossguard/models/model.pt
       |
Secrets Manager -> task execution role -> CGF_API_KEY environment
       |
CloudWatch logs (application does not log CSI payloads)
```

The image must already contain an **approved, contract-compatible checkpoint**. No build step trains a real model. The existing VPC must include two public ALB subnets and two private task subnets in distinct availability zones. Private task networking needs NAT or correctly configured VPC endpoints (including ECR API/DKR, S3, logs and Secrets Manager); this Terraform template does **not** create or pay for networking. AWS documents these requirements: [Fargate task networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-task-networking.html) and [ECR endpoints](https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html).

## Prerequisites

- AWS account/permissions, AWS CLI v2, Terraform >=1.6, Docker.
- Approved dataset licensing, a trained/evaluated checkpoint and model release sign-off.
- An existing VPC, two public and two private subnets, a domain name and a validated ACM HTTPS certificate.
- A pre-created **plaintext string** secret (API key only, not a JSON blob) in AWS Secrets Manager. The service injects the full secret value. [AWS secret injection guidance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/secrets-envvar-secrets-manager.html). If using your own KMS key, explicitly grant scoped `kms:Decrypt` to the task execution role.
- Budget alerts and cost estimates; review IAM permissions and TLS before apply.

## Stage A — offline release verification

```bash
pytest -q
python scripts/smoke.py
crossguard validate --manifest /path/to/approved/manifest.jsonl --data-root /path/to/approved
# Train and evaluate REAL data separately, with its ontology and approval gate.
cp /path/to/approved/best_model.pt artifacts/model.pt
export CGF_API_KEY="use-a-unique-secret-on-your-own-machine"
docker compose up --build
```

`artifacts/model.pt` is excluded from Git and should not contain personal data. Confirm readiness only when the validated model loads, and use `scripts/build_api_request.py` for a sample API request.

## Stage B — create a private ECR repository and publish an image

These commands are intentionally **templates**, not automatic scripts; they can incur charges and must not be run until approved.

```bash
export AWS_REGION=eu-central-1
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export REPO=crossguard-fi
aws ecr create-repository --repository-name "$REPO" --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true --region "$AWS_REGION"
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker build -t "$REPO:staging" .
docker tag "$REPO:staging" \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:staging"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:staging"
aws ecr describe-images --repository-name "$REPO" --image-ids imageTag=staging \
  --query 'imageDetails[0].imageDigest' --output text
```

The digest should be in the form `sha256:<64 lowercase hex>`. Use the repository URI **with that digest**, not a mutable tag, for `container_image_uri`.

## Stage C — provision optional ECS service

1. Copy `infra/terraform/terraform.tfvars.example` to a private, **ignored** `terraform.tfvars` and fill valid AWS IDs, certificate and secret. Avoid placing credentials in tfvars or the repository. Set the ECR digest from Stage B.
2. Configure a secure remote Terraform state backend with access controls and locking before collaboration (not bundled here because it is account-specific).
3. Run `terraform fmt -check`, `terraform init`, `terraform validate`, `terraform plan -out=staging.tfplan`, review cost/resource changes, then **only with approval** `terraform apply staging.tfplan` from `infra/terraform`.
4. Point a **custom DNS name** matching the ACM certificate to the ALB (an ACM certificate does not generally cover the raw `*.elb.amazonaws.com` hostname). Validate HTTPS end-to-end, `/readyz`, authentication, throttling, API limits, logs, and rollback.
5. Check AWS Health, ECS service events, CloudWatch metrics and log group; ensure no raw CSI inputs appear in logs.

### Versioning and operations

The task definition references an immutable digest; publishing a new tag is not enough to roll out a new model. Rebuild and pin a new approved digest, review `terraform plan` and update the ECS service. For rotated Secrets Manager environment secrets, **start new tasks**; existing containers do not automatically receive updated values.

### Network and IAM considerations

- Public internet traffic terminates at the **HTTPS** ALB; private Fargate tasks have no public IP.
- Security groups allow only ALB-to-task port 8000, and task HTTPS outbound access for required AWS endpoints.
- The execution role uses the managed ECS task-execution policy plus a narrowly scoped single-secret `GetSecretValue` permission. If using custom KMS CMK, add scoped decrypt permissions after review.
- No application task role is assigned because inference does not access AWS resources at request time.
- The ALB uses `/readyz`, not `/healthz`, for healthy target determination. The task also has a readiness healthcheck.
- Add AWS WAF/rate limits or API Gateway, alarms, access logs and incident response before broad internet exposure. This template intentionally stays small and inspectable.

## Teardown and spend control

Before deletion, preserve any required logs/checkpoints and remove dependent DNS routes. Review `terraform destroy` and the remaining resources (ECR images, VPC/NAT, ACM, Secrets Manager, remote state) created outside this Terraform stack. Disable deletion protection if enabled, then destroy with care. Deleting only the ECS service **does not** necessarily stop ALB, NAT, ECR or log charges.
