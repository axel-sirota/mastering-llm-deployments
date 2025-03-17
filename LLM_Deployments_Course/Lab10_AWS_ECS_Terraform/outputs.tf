# outputs.tf
output "ecs_cluster_id" {
  value = aws_ecs_cluster.llm_cluster.id
}
output "ecs_service_name" {
  value = aws_ecs_service.llm_service.name
}
