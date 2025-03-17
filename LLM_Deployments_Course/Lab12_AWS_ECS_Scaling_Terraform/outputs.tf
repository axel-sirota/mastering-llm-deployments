# outputs.tf
output "ecs_cluster_id" {
  value = aws_ecs_cluster.llm_cluster.id
}
output "load_balancer_dns" {
  value = aws_lb.llm_lb.dns_name
}
