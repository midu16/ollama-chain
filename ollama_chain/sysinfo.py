"""Read-only system information gathering.

Collects live host metrics (CPU, memory, disk, network, processes, uptime,
etc.) and formats them as a text block suitable for injection into LLM
prompts.  Every function here is strictly observational — no state is
mutated and no commands are executed.
"""

import datetime
import os
import platform
import shutil
import socket
import subprocess


def _read_file(path: str) -> str:
    """Return file contents or empty string on any error."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _run(cmd: list[str], timeout: int = 5) -> str:
    """Run a read-only command and return stdout (empty on failure)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------------
# Individual collectors
# ------------------------------------------------------------------

def host_info() -> dict[str, str]:
    uname = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "system": uname.system,
        "kernel": uname.release,
        "arch": uname.machine,
        "python": platform.python_version(),
    }


def uptime() -> str:
    raw = _read_file("/proc/uptime")
    if not raw:
        return _run(["uptime", "-p"])
    secs = float(raw.split()[0])
    days, rem = divmod(int(secs), 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{mins}m")
    return " ".join(parts)


def cpu_info() -> dict[str, str]:
    info: dict[str, str] = {}
    cpuinfo = _read_file("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    info["cores"] = str(os.cpu_count() or "?")
    loadavg = _read_file("/proc/loadavg")
    if loadavg:
        parts = loadavg.split()
        info["load_1m"] = parts[0]
        info["load_5m"] = parts[1]
        info["load_15m"] = parts[2]
    return info


def memory_info() -> dict[str, str]:
    meminfo = _read_file("/proc/meminfo")
    if not meminfo:
        return {}
    vals: dict[str, int] = {}
    for line in meminfo.splitlines():
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if line.startswith(key + ":"):
                vals[key] = int(line.split()[1])
    def _fmt(kb: int) -> str:
        if kb >= 1_048_576:
            return f"{kb / 1_048_576:.1f} GB"
        return f"{kb / 1024:.0f} MB"

    info: dict[str, str] = {}
    if "MemTotal" in vals:
        info["total"] = _fmt(vals["MemTotal"])
    if "MemAvailable" in vals and "MemTotal" in vals:
        used = vals["MemTotal"] - vals["MemAvailable"]
        info["used"] = _fmt(used)
        info["available"] = _fmt(vals["MemAvailable"])
        info["pct_used"] = f"{used * 100 / vals['MemTotal']:.0f}%"
    if vals.get("SwapTotal", 0) > 0:
        swap_used = vals["SwapTotal"] - vals.get("SwapFree", 0)
        info["swap_total"] = _fmt(vals["SwapTotal"])
        info["swap_used"] = _fmt(swap_used)
    return info


def disk_info() -> list[dict[str, str]]:
    disks: list[dict[str, str]] = []
    for mount in ("/", "/home", "/tmp", "/var"):
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        disks.append({
            "mount": mount,
            "total": f"{usage.total / (1 << 30):.1f} GB",
            "used": f"{usage.used / (1 << 30):.1f} GB",
            "free": f"{usage.free / (1 << 30):.1f} GB",
            "pct_used": f"{usage.used * 100 / usage.total:.0f}%",
        })
    return disks


def network_interfaces() -> list[dict[str, str]]:
    raw = _run(["ip", "-br", "addr"])
    if not raw:
        return []
    ifaces: list[dict[str, str]] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            ifaces.append({
                "name": parts[0],
                "state": parts[1],
                "addrs": " ".join(parts[2:]),
            })
        elif len(parts) == 2:
            ifaces.append({"name": parts[0], "state": parts[1], "addrs": ""})
    return ifaces


def top_processes_text(n: int = 10) -> str:
    header_and_rows = _run(
        ["ps", "axo", "pid,user,%cpu,%mem,comm", "--sort=-%cpu"],
        timeout=5,
    )
    if not header_and_rows:
        return ""
    lines = header_and_rows.splitlines()
    return "\n".join(lines[: n + 1])


def logged_in_users() -> str:
    return _run(["who"])


def datetime_info() -> dict[str, str]:
    now = datetime.datetime.now(datetime.timezone.utc)
    local = datetime.datetime.now().astimezone()
    return {
        "utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "local": local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "timezone": str(local.tzinfo),
    }


def systemd_failed_units() -> str:
    return _run(["systemctl", "--no-pager", "--plain", "list-units", "--state=failed"])


def listening_ports() -> str:
    return _run(["ss", "-tlnp"])


# ------------------------------------------------------------------
# Aggregated snapshot
# ------------------------------------------------------------------

def collect_snapshot() -> str:
    """Return a formatted text block with the current system state."""
    sections: list[str] = []

    hi = host_info()
    sections.append(
        f"Host: {hi['hostname']}  |  {hi['system']} {hi['kernel']} ({hi['arch']})  |  Python {hi['python']}"
    )

    dt = datetime_info()
    sections.append(f"Time: {dt['utc']}  (local: {dt['local']})")
    sections.append(f"Uptime: {uptime()}")

    ci = cpu_info()
    cpu_line = f"CPU: {ci.get('model', '?')}  |  {ci['cores']} cores"
    if "load_1m" in ci:
        cpu_line += f"  |  load {ci['load_1m']} {ci['load_5m']} {ci['load_15m']}"
    sections.append(cpu_line)

    mi = memory_info()
    if mi:
        mem_line = f"Memory: {mi.get('used', '?')} / {mi.get('total', '?')} ({mi.get('pct_used', '?')})"
        if "swap_total" in mi:
            mem_line += f"  |  Swap: {mi['swap_used']} / {mi['swap_total']}"
        sections.append(mem_line)

    disks = disk_info()
    if disks:
        lines = [f"  {d['mount']}: {d['used']} / {d['total']} ({d['pct_used']})" for d in disks]
        sections.append("Disks:\n" + "\n".join(lines))

    nets = network_interfaces()
    if nets:
        lines = [f"  {n['name']}: {n['state']}  {n['addrs']}" for n in nets]
        sections.append("Network:\n" + "\n".join(lines))

    ports = listening_ports()
    if ports:
        sections.append(f"Listening ports:\n{ports}")

    procs = top_processes_text()
    if procs:
        sections.append(f"Top processes (by CPU):\n{procs}")

    users = logged_in_users()
    if users:
        sections.append(f"Logged-in users:\n{users}")

    failed = systemd_failed_units()
    if failed and "0 loaded" not in failed:
        sections.append(f"Failed systemd units:\n{failed}")

    return "\n\n".join(sections)
