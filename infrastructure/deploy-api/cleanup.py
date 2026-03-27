"""
Cleanup script — run after class to tear down all student ECS services.

Usage:
    python cleanup.py --cluster llm-deployments-cluster
    python cleanup.py --cluster llm-deployments-cluster --dry-run
"""
import argparse
import boto3

ecs = boto3.client("ecs")


def cleanup(cluster_name, dry_run=False):
    paginator = ecs.get_paginator("list_services")
    for page in paginator.paginate(cluster=cluster_name):
        for service_arn in page["serviceArns"]:
            service_name = service_arn.split("/")[-1]
            if not service_name.startswith("student-"):
                continue

            if dry_run:
                print(f"[DRY RUN] Would delete: {service_name}")
                continue

            print(f"Scaling down {service_name}...")
            ecs.update_service(
                cluster=cluster_name,
                service=service_name,
                desiredCount=0,
            )
            print(f"Deleting {service_name}...")
            ecs.delete_service(
                cluster=cluster_name,
                service=service_name,
            )
            print(f"Deleted {service_name}")

    print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up student ECS services")
    parser.add_argument("--cluster", required=True, help="ECS cluster name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()
    cleanup(args.cluster, args.dry_run)
