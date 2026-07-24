"""
Railway API Extension for Kimi Bridge
Add these endpoints to your existing FastAPI app.
Requires RAILWAY_API_TOKEN env var.
"""

import os
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/railway", tags=["railway"])

RAILWAY_TOKEN = os.getenv("RAILWAY_API_TOKEN")
RAILWAY_GRAPHQL = "https://backboard.railway.app/graphql/v2"

headers = {
    "Authorization": f"Bearer {RAILWAY_TOKEN}",
    "Content-Type": "application/json"
}

def railway_query(query: str, variables: dict = None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(RAILWAY_GRAPHQL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    if data.get("errors"):
        raise HTTPException(status_code=400, detail=data["errors"])
    return data["data"]


@router.get("/projects")
def list_projects():
    """List all Railway projects."""
    # NOTE: do NOT wrap this in `me { ... }`. A team/project-scoped
    # RAILWAY_API_TOKEN can read projects and services but is Not Authorized
    # for `me`, so the wrapper breaks this endpoint for exactly the tokens
    # most people deploy with. Top-level `projects` works for both.
    query = """
    query {
        projects {
            edges {
                node {
                    id
                    name
                    description
                }
            }
        }
    }
    """
    return railway_query(query)


@router.get("/project/{project_id}/services")
def list_services(project_id: str):
    """List services in a project."""
    # `source` was removed from type Service in Railway's current schema
    # (GRAPHQL_VALIDATION_FAILED: Cannot query field "source" on type "Service").
    query = """
    query($id: String!) {
        project(id: $id) {
            services {
                edges {
                    node {
                        id
                        name
                    }
                }
            }
        }
    }
    """
    return railway_query(query, {"id": project_id})


@router.get("/service/{service_id}/deployments")
def list_deployments(service_id: str, limit: int = 10):
    """List recent deployments for a service."""
    query = """
    query($id: String!, $limit: Int!) {
        service(id: $id) {
            deployments(last: $limit) {
                edges {
                    node {
                        id
                        status
                        createdAt
                        updatedAt
                    }
                }
            }
        }
    }
    """
    return railway_query(query, {"id": service_id, "limit": limit})


@router.get("/deployment/{deployment_id}/logs")
def get_logs(deployment_id: str, limit: int = 100):
    """Get logs for a specific deployment."""
    query = """
    query($id: String!, $limit: Int!) {
        deploymentLogs(deploymentId: $id, limit: $limit) {
            message
            timestamp
            severity
        }
    }
    """
    return railway_query(query, {"id": deployment_id, "limit": limit})


@router.get("/service/{service_id}/status")
def get_service_status(service_id: str):
    """Get current deployment status for a service.

    Queries the last 20 deployments and returns the most recent one
    by createdAt, avoiding stale FAILED results from `last: 1`.
    """
    query = """
    query($id: String!) {
        service(id: $id) {
            deployments(last: 20) {
                edges {
                    node {
                        id
                        status
                        createdAt
                        updatedAt
                    }
                }
            }
        }
    }
    """
    data = railway_query(query, {"id": service_id})
    edges = (
        data.get("service", {})
        .get("deployments", {})
        .get("edges", [])
    )
    if not edges:
        return data
    # Sort by createdAt descending and return the newest deployment
    newest = max(edges, key=lambda e: e.get("node", {}).get("createdAt", ""))
    return {
        "service": {
            "deployments": {
                "edges": [newest]
            }
        }
    }


@router.post("/service/{service_id}/redeploy")
def redeploy_service(service_id: str, environment: str = "production"):
    """Trigger a redeploy of a service.

    Current Railway schema: serviceInstanceDeploy(serviceId, environmentId)
    returns Boolean. The environment id is resolved by name (default
    'production') from the service's project.
    """
    # 1. find the service's project + environment id
    lookup = """
    query($id: String!) {
        service(id: $id) {
            project {
                environments {
                    edges { node { id name } }
                }
            }
        }
    }
    """
    data = railway_query(lookup, {"id": service_id})
    envs = (
        data.get("service", {})
        .get("project", {})
        .get("environments", {})
        .get("edges", [])
    )
    env_id = None
    for e in envs:
        if e.get("node", {}).get("name") == environment:
            env_id = e["node"]["id"]
            break
    if not env_id and envs:
        env_id = envs[0]["node"]["id"]
    if not env_id:
        return {"error": "no environment found for service", "raw": data}

    # 2. deploy
    mutation = """
    mutation($sid: String!, $eid: String!) {
        serviceInstanceDeploy(serviceId: $sid, environmentId: $eid)
    }
    """
    return railway_query(mutation, {"sid": service_id, "eid": env_id})


@router.get("/service/{service_id}/env")
def list_env_vars(service_id: str):
    """List environment variable names (values hidden)."""
    query = """
    query($id: String!) {
        service(id: $id) {
            serviceInstances {
                edges {
                    node {
                        id
                        domains {
                            serviceDomains {
                                domain
                            }
                        }
                    }
                }
            }
        }
    }
    """
    # Note: Railway GraphQL doesn't expose env var names directly via public API
    # This returns service instance metadata instead
    return railway_query(query, {"id": service_id})


# --- v1.2.0: generic GraphQL proxy (service configuration, volumes, domains) ---

from pydantic import BaseModel

class GqlRequest(BaseModel):
    query: str
    variables: dict = None

@router.post("/gql")
def railway_gql(req: GqlRequest):
    """Generic Railway GraphQL proxy. Body: {"query": "...", "variables": {...}}.
    Enables serviceInstanceUpdate (root dir, build/start), variableCollectionUpsert,
    volumeCreate, serviceDomainCreate, etc. without dedicated endpoints."""
    return railway_query(req.query, req.variables or {})


@router.post("/graphql")
def railway_graphql(req: GqlRequest):
    """Alias for /gql (same generic proxy, matches the /railway/graphql name
    used in the bridge-key handoff design doc)."""
    return railway_gql(req)


# --- v1.3.0: self-service key rotation support ---

BRIDGE_SERVICE_ID = os.getenv("BRIDGE_SERVICE_ID", "509d2aef-f1a5-4854-9a57-e44cf9c079a0")
BRIDGE_ENVIRONMENT_NAME = os.getenv("BRIDGE_ENVIRONMENT_NAME", "production")


def _resolve_project_and_env(service_id: str, environment_name: str):
    """Look up the projectId and environmentId for a service, by env name."""
    query = """
    query($id: String!) {
        service(id: $id) {
            projectId
            project {
                environments {
                    edges { node { id name } }
                }
            }
        }
    }
    """
    data = railway_query(query, {"id": service_id})
    svc = data.get("service") or {}
    project_id = svc.get("projectId")
    envs = svc.get("project", {}).get("environments", {}).get("edges", [])
    env_id = None
    for e in envs:
        if e.get("node", {}).get("name") == environment_name:
            env_id = e["node"]["id"]
            break
    if not env_id and envs:
        env_id = envs[0]["node"]["id"]
    return project_id, env_id


def set_service_variable(
    name: str,
    value: str,
    service_id: str = None,
    environment_name: str = None,
):
    """Upsert a single environment variable on a Railway service (defaults
    to this bridge's own service/environment). Used by POST /rotate_key in
    main.py to persist a freshly rotated BRIDGE_API_KEY so it survives a
    restart or redeploy, not just the current in-memory process."""
    service_id = service_id or BRIDGE_SERVICE_ID
    environment_name = environment_name or BRIDGE_ENVIRONMENT_NAME
    project_id, env_id = _resolve_project_and_env(service_id, environment_name)
    if not project_id or not env_id:
        raise HTTPException(
            status_code=502,
            detail="could not resolve project/environment for variable upsert",
        )
    mutation = """
    mutation($input: VariableUpsertInput!) {
        variableUpsert(input: $input)
    }
    """
    variables = {
        "input": {
            "projectId": project_id,
            "environmentId": env_id,
            "serviceId": service_id,
            "name": name,
            "value": value,
        }
    }
    return railway_query(mutation, variables)
