# variables.tf
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for the security group"
  type        = string
}

variable "subnets" {
  description = "List of public subnet IDs for Fargate tasks"
  type        = list(string)
}
