# outputs.tf
output "api_endpoint" {
  description = "Base URL for the deploy API"
  value       = aws_api_gateway_stage.prod.invoke_url
}

output "api_key" {
  description = "API key to distribute to students"
  value       = aws_api_gateway_api_key.class_key.value
  sensitive   = true
}

output "cluster_name" {
  description = "ECS cluster name (for cleanup script)"
  value       = aws_ecs_cluster.students.name
}

output "security_group_id" {
  description = "Security group ID created for student tasks"
  value       = aws_security_group.student_tasks.id
}
