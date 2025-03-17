# variables.tf
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}
variable "execution_role_arn" {
  description = "ARN of the ECS task execution role"
  type        = string
}
variable "docker_image" {
  description = "Docker image URI"
  type        = string
}
variable "subnets" {
  description = "List of subnet IDs"
  type        = list(string)
}
variable "security_groups" {
  description = "List of security group IDs"
  type        = list(string)
}
