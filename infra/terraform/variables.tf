variable "aws_region" {
  description = "AWS region with existing VPC, subnets, certificate and secret"
  type        = string
}
variable "environment" {
  type    = string
  default = "staging"
}
variable "vpc_id" {
  description = "Existing VPC identifier"
  type        = string
}
variable "public_subnet_ids" {
  description = "Two or more public subnets in different AZs for ALB"
  type        = list(string)
  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "At least two public subnets are required"
  }
}
variable "private_subnet_ids" {
  description = "Private subnets with NAT or ECR/S3/Logs/Secrets VPC endpoints"
  type        = list(string)
  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "At least two private subnets are required"
  }
}
variable "tls_certificate_arn" {
  description = "Existing ACM certificate ARN for HTTPS ALB listener"
  type        = string
}
variable "api_key_secret_arn" {
  description = "Existing Secrets Manager plaintext API-key secret ARN (not a JSON object)"
  type        = string
}
variable "container_image_uri" {
  description = "ECR image pinned to SHA-256 digest, containing a validated model.pt"
  type        = string
  validation {
    condition     = can(regex("@sha256:[a-f0-9]{64}$", var.container_image_uri))
    error_message = "Use an immutable image digest, not a mutable image tag"
  }
}
variable "desired_count" {
  type    = number
  default = 1
  validation {
    condition     = var.desired_count >= 1
    error_message = "At least one task required for a healthy ALB"
  }
}
variable "enable_deletion_protection" {
  type    = bool
  default = false
}
