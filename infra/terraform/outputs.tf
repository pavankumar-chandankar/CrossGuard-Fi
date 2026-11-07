output "https_endpoint" {
  value       = "https://${aws_lb.api.dns_name}"
  description = "DNS endpoint; configure a custom DNS name matching the ACM certificate"
}
output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}
output "ecs_service_name" {
  value = aws_ecs_service.api.name
}
output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}
