"""Kubernetes / OpenShift cluster analysis for LLM consumption.

Gathers cluster state via ``oc`` (preferred) or ``kubectl`` using a
user-supplied kubeconfig, then formats the output into a structured
report that the chain pipeline feeds to LLMs for expert analysis.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class K8sAnalysis:
    kubeconfig: str
    cli_tool: str = "oc"
    is_openshift: bool = False
    cluster_version: str = ""
    api_server_url: str = ""
    platform: str = ""
    nodes: str = ""
    namespaces: str = ""
    pods: str = ""
    unhealthy_pods: str = ""
    deployments: str = ""
    services: str = ""
    ingresses: str = ""
    events: str = ""
    pvs: str = ""
    storage_classes: str = ""
    resource_usage_nodes: str = ""
    resource_usage_pods: str = ""
    # OpenShift-specific
    cluster_operators: str = ""
    routes: str = ""
    machines: str = ""
    machine_config_pools: str = ""
    cluster_service_versions: str = ""
    infrastructure: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _run(
    args: list[str],
    kubeconfig: str,
    cli: str,
    timeout: int = 30,
) -> str:
    """Run a kubectl/oc command and return stdout, or empty string on failure."""
    cmd = [cli, f"--kubeconfig={kubeconfig}"] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return ""


def analyze_cluster(kubeconfig: str) -> K8sAnalysis:
    """Gather comprehensive cluster state from a kubeconfig file."""
    if not os.path.isfile(kubeconfig):
        print(f"Error: kubeconfig not found: {kubeconfig}", file=sys.stderr)
        sys.exit(1)

    cli = "oc" if shutil.which("oc") else "kubectl"
    if not shutil.which(cli):
        print(
            f"Error: neither 'oc' nor 'kubectl' found in PATH",
            file=sys.stderr,
        )
        sys.exit(1)

    a = K8sAnalysis(kubeconfig=kubeconfig, cli_tool=cli)

    # -- Detect OpenShift ---------------------------------------------------
    cv = _run(["get", "clusterversion", "-o", "wide"], kubeconfig, cli)
    if cv:
        a.is_openshift = True
        a.cluster_version = cv

    # -- Cluster info -------------------------------------------------------
    ver = _run(["version", "--short"], kubeconfig, cli, timeout=10)
    if not ver:
        ver = _run(["version"], kubeconfig, cli, timeout=10)
    a.platform = ver

    info = _run(["cluster-info"], kubeconfig, cli, timeout=10)
    if info:
        for line in info.splitlines():
            if "running at" in line.lower():
                a.api_server_url = line.strip()
                break

    # -- Nodes --------------------------------------------------------------
    a.nodes = _run(
        ["get", "nodes", "-o", "wide", "--show-labels"], kubeconfig, cli,
    )

    # -- Namespaces ---------------------------------------------------------
    a.namespaces = _run(["get", "namespaces"], kubeconfig, cli)

    # -- Pods ---------------------------------------------------------------
    a.pods = _run(
        ["get", "pods", "--all-namespaces", "-o", "wide", "--sort-by=.metadata.namespace"],
        kubeconfig, cli, timeout=60,
    )
    a.unhealthy_pods = _run(
        [
            "get", "pods", "--all-namespaces",
            "--field-selector=status.phase!=Running,status.phase!=Succeeded",
            "-o", "wide",
        ],
        kubeconfig, cli,
    )

    # -- Deployments --------------------------------------------------------
    a.deployments = _run(
        ["get", "deployments", "--all-namespaces", "-o", "wide"],
        kubeconfig, cli,
    )

    # -- Services -----------------------------------------------------------
    a.services = _run(
        ["get", "services", "--all-namespaces"], kubeconfig, cli,
    )

    # -- Ingresses ----------------------------------------------------------
    a.ingresses = _run(
        ["get", "ingress", "--all-namespaces"], kubeconfig, cli,
    )

    # -- Events (warnings only) ---------------------------------------------
    a.events = _run(
        [
            "get", "events", "--all-namespaces",
            "--sort-by=.lastTimestamp",
            "--field-selector=type=Warning",
        ],
        kubeconfig, cli, timeout=30,
    )

    # -- Storage ------------------------------------------------------------
    a.pvs = _run(["get", "pv"], kubeconfig, cli)
    a.storage_classes = _run(["get", "sc"], kubeconfig, cli)

    # -- Resource usage (requires metrics-server) ---------------------------
    a.resource_usage_nodes = _run(["top", "nodes"], kubeconfig, cli, timeout=15)
    a.resource_usage_pods = _run(
        ["top", "pods", "--all-namespaces", "--sort-by=memory"],
        kubeconfig, cli, timeout=15,
    )

    # -- OpenShift-specific -------------------------------------------------
    if a.is_openshift:
        a.cluster_operators = _run(
            ["get", "clusteroperators"], kubeconfig, cli,
        )
        a.routes = _run(
            ["get", "routes", "--all-namespaces"], kubeconfig, cli,
        )
        a.machines = _run(
            ["get", "machines", "-n", "openshift-machine-api", "-o", "wide"],
            kubeconfig, cli,
        )
        a.machine_config_pools = _run(
            ["get", "machineconfigpool"], kubeconfig, cli,
        )
        a.cluster_service_versions = _run(
            ["get", "csv", "--all-namespaces"],
            kubeconfig, cli, timeout=30,
        )
        a.infrastructure = _run(
            ["get", "infrastructure", "cluster", "-o", "yaml"],
            kubeconfig, cli,
        )

    # -- Detect issues ------------------------------------------------------
    if a.unhealthy_pods:
        for line in a.unhealthy_pods.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4:
                a.warnings.append(
                    f"Unhealthy pod: {parts[0]}/{parts[1]} — status: {parts[3]}"
                )

    if a.events:
        event_count = len([
            l for l in a.events.splitlines() if l.strip() and not l.startswith("NAMESPACE")
        ])
        if event_count > 20:
            a.warnings.append(
                f"{event_count} warning events detected — review events section"
            )

    if a.cluster_operators and a.is_openshift:
        for line in a.cluster_operators.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5:
                name, available, progressing, degraded = (
                    parts[0], parts[1], parts[2], parts[3],
                )
                if degraded.lower() == "true":
                    a.errors.append(f"ClusterOperator '{name}' is DEGRADED")
                if available.lower() == "false":
                    a.errors.append(f"ClusterOperator '{name}' is NOT AVAILABLE")

    return a


def format_analysis(a: K8sAnalysis) -> str:
    """Format K8sAnalysis into a structured text report for LLM consumption."""
    lines: list[str] = []
    platform = "OpenShift" if a.is_openshift else "Kubernetes"
    lines.append(f"=== {platform.upper()} CLUSTER ANALYSIS ===\n")
    lines.append(f"CLI tool: {a.cli_tool}")
    lines.append(f"Kubeconfig: {a.kubeconfig}")

    if a.api_server_url:
        lines.append(f"API Server: {a.api_server_url}")

    _add_section(lines, "CLUSTER VERSION", a.cluster_version or a.platform)
    _add_section(lines, "NODES", a.nodes)
    _add_section(lines, "NAMESPACES", a.namespaces)
    _add_section(lines, "PODS (all namespaces)", a.pods)

    if a.unhealthy_pods:
        _add_section(lines, "UNHEALTHY PODS", a.unhealthy_pods)

    _add_section(lines, "DEPLOYMENTS", a.deployments)
    _add_section(lines, "SERVICES", a.services)

    if a.ingresses:
        _add_section(lines, "INGRESSES", a.ingresses)

    if a.events:
        _add_section(lines, "WARNING EVENTS (recent)", a.events)

    if a.pvs:
        _add_section(lines, "PERSISTENT VOLUMES", a.pvs)

    if a.storage_classes:
        _add_section(lines, "STORAGE CLASSES", a.storage_classes)

    if a.resource_usage_nodes:
        _add_section(lines, "NODE RESOURCE USAGE", a.resource_usage_nodes)

    if a.resource_usage_pods:
        _add_section(lines, "TOP PODS BY MEMORY", a.resource_usage_pods)

    if a.is_openshift:
        _add_section(lines, "CLUSTER OPERATORS", a.cluster_operators)
        if a.routes:
            _add_section(lines, "ROUTES", a.routes)
        if a.machines:
            _add_section(lines, "MACHINES", a.machines)
        if a.machine_config_pools:
            _add_section(lines, "MACHINE CONFIG POOLS", a.machine_config_pools)
        if a.cluster_service_versions:
            _add_section(lines, "INSTALLED OPERATORS (CSV)", a.cluster_service_versions)
        if a.infrastructure:
            _add_section(lines, "INFRASTRUCTURE", a.infrastructure)

    if a.errors:
        lines.append(f"\n--- ERRORS DETECTED ({len(a.errors)}) ---")
        for e in a.errors:
            lines.append(f"  [ERR] {e}")

    if a.warnings:
        lines.append(f"\n--- WARNINGS ({len(a.warnings)}) ---")
        for w in a.warnings:
            lines.append(f"  [WARN] {w}")

    if not a.errors and not a.warnings:
        lines.append("\n--- No errors or warnings detected ---")

    return "\n".join(lines)


def _add_section(lines: list[str], title: str, content: str) -> None:
    if content and content.strip():
        lines.append(f"\n--- {title} ---")
        lines.append(content)
