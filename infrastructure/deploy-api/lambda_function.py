"""
Deploy API Lambda — lets students deploy Docker images to the instructor's ECS cluster.

Endpoints:
  POST /deploy  — body: {"student_name": "john", "image": "john/gradio-app:latest"}
  GET  /status  — query: ?student=john
  DELETE /deploy — body: {"student_name": "john"}
"""
import json
import os
import boto3
from botocore.exceptions import ClientError

ecs = boto3.client("ecs")
ec2 = boto3.client("ec2")

CLUSTER = os.environ["CLUSTER_NAME"]
SUBNETS = os.environ["SUBNETS"].split(",")
SECURITY_GROUPS = os.environ["SECURITY_GROUPS"].split(",")
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]


def respond(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def deploy(student_name, image):
    family = f"student-{student_name}"
    service_name = f"student-{student_name}"

    ecs.register_task_definition(
        family=family,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu="512",
        memory="1024",
        executionRoleArn=EXECUTION_ROLE_ARN,
        containerDefinitions=[
            {
                "name": "app",
                "image": image,
                "essential": True,
                "portMappings": [
                    {"containerPort": 7860, "hostPort": 7860, "protocol": "tcp"}
                ],
            }
        ],
    )

    try:
        ecs.create_service(
            cluster=CLUSTER,
            serviceName=service_name,
            taskDefinition=family,
            desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNETS,
                    "securityGroups": SECURITY_GROUPS,
                    "assignPublicIp": "ENABLED",
                }
            },
        )
    except ClientError as e:
        err = str(e)
        if "already exists" in err or "still Draining" in err:
            # Service exists or is draining from a previous delete — update it
            if "still Draining" in err:
                # Wait briefly for drain to complete, then retry create
                import time
                time.sleep(5)
                try:
                    ecs.create_service(
                        cluster=CLUSTER,
                        serviceName=service_name,
                        taskDefinition=family,
                        desiredCount=1,
                        launchType="FARGATE",
                        networkConfiguration={
                            "awsvpcConfiguration": {
                                "subnets": SUBNETS,
                                "securityGroups": SECURITY_GROUPS,
                                "assignPublicIp": "ENABLED",
                            }
                        },
                    )
                except ClientError:
                    return respond(409, {
                        "status": "DRAINING",
                        "student": student_name,
                        "message": "Previous service is still shutting down. Wait 30 seconds and try again.",
                    })
            else:
                ecs.update_service(
                    cluster=CLUSTER,
                    service=service_name,
                    taskDefinition=family,
                    forceNewDeployment=True,
                )
        else:
            raise

    return respond(200, {
        "status": "DEPLOYING",
        "student": student_name,
        "message": "Task is starting. Check /status?student={} in ~60 seconds.".format(student_name),
    })


def status(student_name):
    service_name = f"student-{student_name}"

    try:
        tasks = ecs.list_tasks(cluster=CLUSTER, serviceName=service_name)
    except ClientError:
        return respond(404, {"status": "NOT_FOUND", "student": student_name})

    if not tasks.get("taskArns"):
        return respond(200, {"status": "PENDING", "student": student_name, "url": None})

    described = ecs.describe_tasks(cluster=CLUSTER, tasks=tasks["taskArns"])
    for task in described.get("tasks", []):
        if task["lastStatus"] != "RUNNING":
            return respond(200, {
                "status": task["lastStatus"],
                "student": student_name,
                "url": None,
            })

        for attachment in task.get("attachments", []):
            for detail in attachment.get("details", []):
                if detail["name"] == "networkInterfaceId":
                    eni_id = detail["value"]
                    eni = ec2.describe_network_interfaces(
                        NetworkInterfaceIds=[eni_id]
                    )
                    public_ip = (
                        eni["NetworkInterfaces"][0]
                        .get("Association", {})
                        .get("PublicIp")
                    )
                    if public_ip:
                        return respond(200, {
                            "status": "RUNNING",
                            "student": student_name,
                            "url": f"http://{public_ip}:7860",
                        })

    return respond(200, {"status": "STARTING", "student": student_name, "url": None})


def teardown(student_name):
    service_name = f"student-{student_name}"
    try:
        ecs.update_service(cluster=CLUSTER, service=service_name, desiredCount=0)
        ecs.delete_service(cluster=CLUSTER, service=service_name)
    except ClientError as e:
        return respond(404, {"error": str(e)})
    return respond(200, {"status": "DELETED", "student": student_name})


def handler(event, context):
    method = event.get("httpMethod", "")
    path = event.get("path", "")

    if method == "POST" and path == "/deploy":
        body = json.loads(event.get("body", "{}"))
        student_name = body.get("student_name", "").strip().lower()
        image = body.get("image", "").strip()
        if not student_name or not image:
            return respond(400, {"error": "student_name and image are required"})
        if not student_name.isalnum():
            return respond(400, {"error": "student_name must be alphanumeric"})
        return deploy(student_name, image)

    elif method == "GET" and path == "/status":
        params = event.get("queryStringParameters") or {}
        student_name = params.get("student", "").strip().lower()
        if not student_name:
            return respond(400, {"error": "student query parameter is required"})
        return status(student_name)

    elif method == "DELETE" and path == "/deploy":
        body = json.loads(event.get("body", "{}"))
        student_name = body.get("student_name", "").strip().lower()
        if not student_name:
            return respond(400, {"error": "student_name is required"})
        return teardown(student_name)

    return respond(404, {"error": "Not found"})
