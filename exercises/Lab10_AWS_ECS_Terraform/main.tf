# main.tf

resource "aws_ecs_cluster" "llm_cluster" {
  name = "llm-deployments-cluster"
}

resource "aws_ecs_task_definition" "llm_task" {
  family                   = "llm-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = var.execution_role_arn
  container_definitions    = jsonencode([
    {
      name      = "llm-container"
      image     = var.docker_image
      essential = true
      portMappings = [
        {
          containerPort = 7860,
          hostPort      = 7860
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "llm_service" {
  name            = "llm-service"
  cluster         = aws_ecs_cluster.llm_cluster.id
  task_definition = aws_ecs_task_definition.llm_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.subnets
    security_groups = var.security_groups
    assign_public_ip = true
  }
}
