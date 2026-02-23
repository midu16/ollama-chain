"""Unit tests for k8s.py — K8sAnalysis, _run, format_analysis, issue detection."""

from unittest.mock import patch, MagicMock

import pytest

from ollama_chain.k8s import K8sAnalysis, _run, format_analysis


# ---------------------------------------------------------------------------
# K8sAnalysis dataclass
# ---------------------------------------------------------------------------

class TestK8sAnalysis:
    def test_defaults(self):
        a = K8sAnalysis(kubeconfig="/path/kube")
        assert a.kubeconfig == "/path/kube"
        assert a.cli_tool == "oc"
        assert a.is_openshift is False
        assert a.errors == []
        assert a.warnings == []
        assert a.nodes == ""

    def test_openshift_fields(self):
        a = K8sAnalysis(
            kubeconfig="/kube",
            is_openshift=True,
            cluster_operators="op1 True False False",
        )
        assert a.is_openshift is True
        assert a.cluster_operators != ""


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------

class TestRun:
    @patch("ollama_chain.k8s.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="output data\n",
        )
        result = _run(["get", "nodes"], "/kube", "kubectl")
        assert result == "output data"

    @patch("ollama_chain.k8s.subprocess.run")
    def test_failure_returns_empty(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _run(["get", "nodes"], "/kube", "kubectl")
        assert result == ""

    @patch("ollama_chain.k8s.subprocess.run")
    def test_timeout_returns_empty(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="oc", timeout=30)
        result = _run(["get", "nodes"], "/kube", "oc")
        assert result == ""

    @patch("ollama_chain.k8s.subprocess.run")
    def test_file_not_found_returns_empty(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run(["get", "nodes"], "/kube", "oc")
        assert result == ""


# ---------------------------------------------------------------------------
# format_analysis
# ---------------------------------------------------------------------------

class TestFormatAnalysis:
    def test_basic_kubernetes(self):
        a = K8sAnalysis(
            kubeconfig="/kube",
            cli_tool="kubectl",
            nodes="node1 Ready",
            namespaces="default\nkube-system",
            pods="pod1 Running",
        )
        report = format_analysis(a)
        assert "KUBERNETES CLUSTER ANALYSIS" in report
        assert "kubectl" in report
        assert "node1" in report
        assert "pod1" in report

    def test_openshift_with_operators(self):
        a = K8sAnalysis(
            kubeconfig="/kube",
            cli_tool="oc",
            is_openshift=True,
            cluster_operators="auth True False False",
            routes="route1 example.com",
        )
        report = format_analysis(a)
        assert "OPENSHIFT CLUSTER ANALYSIS" in report
        assert "CLUSTER OPERATORS" in report
        assert "ROUTES" in report

    def test_errors_section(self):
        a = K8sAnalysis(kubeconfig="/k")
        a.errors = ["ClusterOperator 'dns' is DEGRADED"]
        report = format_analysis(a)
        assert "ERRORS DETECTED" in report
        assert "DEGRADED" in report

    def test_warnings_section(self):
        a = K8sAnalysis(kubeconfig="/k")
        a.warnings = ["Unhealthy pod: ns/pod1 — status: CrashLoopBackOff"]
        report = format_analysis(a)
        assert "WARNINGS" in report
        assert "CrashLoopBackOff" in report

    def test_no_issues(self):
        a = K8sAnalysis(kubeconfig="/k")
        report = format_analysis(a)
        assert "No errors or warnings detected" in report

    def test_optional_sections_omitted_when_empty(self):
        a = K8sAnalysis(kubeconfig="/k")
        report = format_analysis(a)
        assert "INGRESSES" not in report
        assert "PERSISTENT VOLUMES" not in report
        assert "WARNING EVENTS" not in report

    def test_optional_sections_included_when_present(self):
        a = K8sAnalysis(
            kubeconfig="/k",
            ingresses="ing1 host.example.com",
            pvs="pv1 10Gi Bound",
            storage_classes="gp2 default",
            events="Warning pod restarted",
        )
        report = format_analysis(a)
        assert "INGRESSES" in report
        assert "PERSISTENT VOLUMES" in report
        assert "STORAGE CLASSES" in report
        assert "WARNING EVENTS" in report

    def test_api_server_url(self):
        a = K8sAnalysis(
            kubeconfig="/k",
            api_server_url="https://api.cluster.example.com:6443",
        )
        report = format_analysis(a)
        assert "api.cluster.example.com" in report


# ---------------------------------------------------------------------------
# Issue detection logic
# ---------------------------------------------------------------------------

class TestIssueDetection:
    @patch("ollama_chain.k8s._run")
    @patch("ollama_chain.k8s.shutil.which", return_value="/usr/bin/kubectl")
    @patch("ollama_chain.k8s.os.path.isfile", return_value=True)
    def test_unhealthy_pods_detected(self, _isfile, _which, mock_run):
        def side_effect(args, kube, cli, timeout=30):
            if "field-selector" in str(args):
                return (
                    "NAMESPACE NAME READY STATUS\n"
                    "default   bad  0/1   CrashLoopBackOff"
                )
            return ""
        mock_run.side_effect = side_effect

        from ollama_chain.k8s import analyze_cluster
        a = analyze_cluster("/kube")
        assert any("Unhealthy pod" in w for w in a.warnings)

    @patch("ollama_chain.k8s._run")
    @patch("ollama_chain.k8s.shutil.which", return_value="/usr/bin/oc")
    @patch("ollama_chain.k8s.os.path.isfile", return_value=True)
    def test_degraded_operator_detected(self, _isfile, _which, mock_run):
        def side_effect(args, kube, cli, timeout=30):
            if args[:2] == ["get", "clusterversion"]:
                return "4.14.0 Completed"
            if args[:2] == ["get", "clusteroperators"]:
                return (
                    "NAME       AVAILABLE PROGRESSING DEGRADED SINCE\n"
                    "dns        True      False       True     5m"
                )
            return ""
        mock_run.side_effect = side_effect

        from ollama_chain.k8s import analyze_cluster
        a = analyze_cluster("/kube")
        assert any("DEGRADED" in e for e in a.errors)
