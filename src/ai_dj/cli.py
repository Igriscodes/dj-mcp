import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from ai_dj.capabilities import get_system_capabilities
from ai_dj.config import get_default_config
from ai_dj.mcp.server import main as server_main


def run_doctor(as_json: bool = False) -> int:
    cfg = get_default_config()
    caps = get_system_capabilities()

    # Filesystem checks
    cache_writable = os.access(cfg.cache_dir, os.W_OK)
    temp_writable = os.access(cfg.temp_dir, os.W_OK)
    disk_free_mb = cfg.get_free_disk_space_mb()

    report = {
        "status": "healthy" if (caps["engines"]["ffmpeg"]["available"] and cache_writable) else "degraded",
        "system": caps["os"],
        "gpu": caps["gpu"],
        "engines": caps["engines"],
        "codecs": caps["codecs"],
        "filesystem": {
            "workspace_dir": str(cfg.workspace_dir),
            "cache_dir": str(cfg.cache_dir),
            "cache_writable": cache_writable,
            "temp_dir": str(cfg.temp_dir),
            "temp_writable": temp_writable,
            "disk_free_mb": disk_free_mb,
            "has_sufficient_disk_space": disk_free_mb > 500,
        },
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 60)
    print("DJ-MCP SYSTEM DOCTOR DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Overall Status   : {report['status'].upper()}")
    print(f"Operating System : {report['system']['system']} ({report['system']['architecture']})")
    print(f"Python Version   : {report['system']['python_version']}")
    print(f"CPU Cores        : {report['system']['cpu_cores']}")
    print(f"RAM Total/Avail  : {report['system']['total_ram_gb']} GB / {report['system']['available_ram_gb']} GB")
    print(f"GPU Available    : {report['gpu']['available']} ({report['gpu']['type']})")
    print("-" * 60)
    print("ENGINES & DSP:")
    for eng_name, eng in report["engines"].items():
        avail_str = "✓ INSTALLED" if eng["available"] else "✗ MISSING"
        ver_str = f" ({eng.get('version', '')})" if eng.get("version") else ""
        print(f"  [{avail_str:12}] {eng_name:14} : {eng['purpose']}{ver_str}")
    print("-" * 60)
    print("CODECS & FORMATS:")
    for codec, avail in report["codecs"].items():
        st = "✓ SUPPORTED" if avail else "✗ NOT AVAILABLE"
        print(f"  [{st:14}] {codec.upper()}")
    print("-" * 60)
    print(f"Disk Free Space  : {disk_free_mb} MB")
    print(f"Cache Directory  : {cfg.cache_dir} (Writable: {cache_writable})")
    print(f"Temp Directory   : {cfg.temp_dir} (Writable: {temp_writable})")
    print("=" * 60)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="dj-mcp", description="DJ MCP - Professional AI Audio Workstation")
    subparsers = parser.add_subparsers(dest="command")

    # doctor
    doc_parser = subparsers.add_parser("doctor", help="Run system diagnostics and capability checks")
    doc_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # server
    srv_parser = subparsers.add_parser("server", help="Start the MCP server")
    srv_parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="MCP transport protocol")
    srv_parser.add_argument("--host", default="127.0.0.1", help="Host for SSE server")
    srv_parser.add_argument("--port", type=int, default=8000, help="Port for SSE server")

    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(run_doctor(as_json=args.json))
    elif args.command == "server":
        sys.argv = ["dj-mcp.server", "--transport", args.transport]
        if args.transport == "sse":
            sys.argv.extend(["--host", args.host, "--port", str(args.port)])
        server_main()
    else:
        # Default to doctor if no args, or show help
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
