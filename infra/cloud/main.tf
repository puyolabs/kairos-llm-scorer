# SPDX-License-Identifier: Apache-2.0
# AWS deployment: ECR + Fargate behind an internet-facing ALB.
# Applies out-of-the-box against the account's default VPC.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "cpu" {
  type    = number
  default = 256 # Fargate task CPU units (.25 vCPU)
}

variable "memory" {
  type    = number
  default = 512 # Fargate task memory (MiB)
}

variable "scorer_model" {
  type    = string
  default = "claude-sonnet-4-6"
}

variable "scorer_effort" {
  type    = string
  default = "medium" # screener thinking depth: low | medium | high | max
}

variable "build_sha" {
  type    = string
  default = "dev"
}

variable "log_retention_days" {
  type    = number
  default = 14
}

# Sensitive secret values. Defaults are empty: when empty we create the
# Secrets Manager secret *container* but not a version, so the value is set
# out-of-band (`aws secretsmanager put-secret-value ...`). When non-empty
# (e.g. passed via TF_VAR_*), we manage the version too. This keeps secrets
# out of state by default while still allowing a fully-automated apply.
variable "anthropic_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "scorer_api_keys" {
  type      = string
  default   = ""
  sensitive = true
}

variable "enable_https" {
  type    = bool
  default = false # set true together with acm_certificate_arn to add HTTPS:443
}

variable "acm_certificate_arn" {
  type    = string
  default = ""
}

variable "ingress_cidrs" {
  type        = list(string)
  default     = ["0.0.0.0/0"] # open to the internet; restrict to known CIDRs to harden
  description = "Source CIDRs allowed to reach the ALB on 80/443"
}

provider "aws" {
  region = var.region
}

locals {
  name = "kairos-llm-scorer"
}

# ---------------------------------------------------------------------------
# Networking: reuse the account's default VPC + subnets (no VPC to manage)
# ---------------------------------------------------------------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------------------------------------------------------------------------
# ECR: image repository with a 10-image retention lifecycle policy
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# Secrets Manager: secret containers for the two app secrets. Versions are
# only created when the corresponding variable is non-empty (count guard).
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name = "${local.name}/anthropic-api-key"
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  count         = var.anthropic_api_key != "" ? 1 : 0
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key
}

resource "aws_secretsmanager_secret" "scorer_api_keys" {
  name = "${local.name}/scorer-api-keys"
}

resource "aws_secretsmanager_secret_version" "scorer_api_keys" {
  count         = var.scorer_api_keys != "" ? 1 : 0
  secret_id     = aws_secretsmanager_secret.scorer_api_keys.id
  secret_string = var.scorer_api_keys
}

# ---------------------------------------------------------------------------
# IAM: ECS task execution role — pull from ECR, ship logs, read the secrets
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# AWS-managed policy: ECR pull + CloudWatch Logs write for the agent.
resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Inline policy: allow reading exactly the two secrets at task start.
data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.anthropic_api_key.arn,
      aws_secretsmanager_secret.scorer_api_keys.arn,
    ]
  }
}

resource "aws_iam_role_policy" "read_secrets" {
  name   = "${local.name}-read-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# ---------------------------------------------------------------------------
# CloudWatch: container log group
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

# ---------------------------------------------------------------------------
# Security groups: ALB open on 80 (and optionally 443); service on 8000 only
# from the ALB.
# ---------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "ALB: inbound HTTP(S) from the internet"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }

  dynamic "ingress" {
    for_each = var.enable_https ? [1] : []
    content {
      description = "HTTPS"
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = var.ingress_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "Fargate task: inbound 8000 only from the ALB"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "App port from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# Load balancer: internet-facing ALB -> target group (8000) with /health check
# ---------------------------------------------------------------------------

resource "aws_lb" "app" {
  name               = local.name
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "app" {
  name        = local.name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip" # awsvpc/Fargate tasks register by IP

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# Optional HTTPS listener; only created when enable_https is set.
resource "aws_lb_listener" "https" {
  count             = var.enable_https ? 1 : 0
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ---------------------------------------------------------------------------
# ECS: cluster, task definition, and Fargate service
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "app" {
  name = local.name
}

resource "aws_ecs_task_definition" "app" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn

  container_definitions = jsonencode([{
    name      = "scorer"
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    # Plain (non-secret) configuration.
    environment = [
      { name = "KAIROS_SCORER_MODEL", value = var.scorer_model },
      { name = "KAIROS_SCORER_EFFORT", value = var.scorer_effort },
      { name = "KAIROS_BUILD_SHA", value = var.build_sha },
    ]
    # Secrets injected from Secrets Manager at container start.
    secrets = [
      { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic_api_key.arn },
      { name = "KAIROS_SCORER_API_KEYS", valueFrom = aws_secretsmanager_secret.scorer_api_keys.arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = local.name
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true # tasks live in default public subnets; needed to pull image
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "scorer"
    container_port   = 8000
  }

  # Don't create the service until the listener (and thus the ALB) is ready.
  depends_on = [aws_lb_listener.http]
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "alb_dns_name" {
  description = "Public URL base for the service"
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecr_repository_url" {
  description = "Push images here"
  value       = aws_ecr_repository.app.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.app.name
}

output "service_name" {
  value = aws_ecs_service.app.name
}
