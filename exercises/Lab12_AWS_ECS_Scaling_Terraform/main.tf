# main.tf

resource "aws_ecs_cluster" "llm_cluster" {
  name = "llm-deployments-cluster-scaled"
}

resource "aws_ecs_task_definition" "llm_task" {
  family                   = "llm-task-scaled"
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

resource "aws_lb" "llm_lb" {
  name               = "llm-lb"
  internal           = false
  load_balancer_type = "application"
  subnets            = var.subnets
  security_groups    = var.security_groups
}

resource "aws_lb_target_group" "llm_tg" {
  name     = "llm-tg"
  port     = 7860
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  health_check {
    path                = "/"
    protocol            = "HTTP"
    matcher             = "200-299"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "llm_listener" {
  load_balancer_arn = aws_lb.llm_lb.arn
  port              = "80"
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.llm_tg.arn
  }
}

resource "aws_ecs_service" "llm_service" {
  name            = "llm-service-scaled"
  cluster         = aws_ecs_cluster.llm_cluster.id
  task_definition = aws_ecs_task_definition.llm_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.subnets
    security_groups = var.security_groups
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.llm_tg.arn
    container_name   = "llm-container"
    container_port   = 7860
  }
}
