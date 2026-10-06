#!/usr/bin/env python3
"""Execute every code cell in a fresh Jupyter kernel and save real outputs."""

import json
import os
import sys
from pathlib import Path
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
path = root / "EDA_COCO_vs_VizWiz_Report.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
kernel = KernelManager(kernel_name="python3")
# Use the interpreter that invoked this script, irrespective of a global kernelspec.
kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
kernel_env = os.environ.copy()
kernel_env["IPYTHONDIR"] = str(root / ".ipython_runtime")
kernel_env["JUPYTER_RUNTIME_DIR"] = str(root / ".jupyter_runtime")
(root / ".ipython_runtime").mkdir(exist_ok=True)
(root / ".jupyter_runtime").mkdir(exist_ok=True)
kernel.start_kernel(cwd=str(root), env=kernel_env)
client = kernel.client()
client.start_channels()
try:
    client.wait_for_ready(timeout=60)
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cell["outputs"] = []
        cell["execution_count"] = None
        message_id = client.execute(source, store_history=True, stop_on_error=True)
        while True:
            message = client.get_iopub_msg(timeout=300)
            if message["parent_header"].get("msg_id") != message_id:
                continue
            kind = message["msg_type"]
            body = message["content"]
            if kind == "status" and body["execution_state"] == "idle":
                break
            if kind == "execute_input":
                cell["execution_count"] = body["execution_count"]
            elif kind == "stream":
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": body["name"],
                    "text": body["text"],
                })
            elif kind in ("display_data", "execute_result"):
                output = {
                    "output_type": kind,
                    "data": body["data"],
                    "metadata": body.get("metadata", {}),
                }
                if kind == "execute_result":
                    output["execution_count"] = body["execution_count"]
                cell["outputs"].append(output)
            elif kind == "error":
                cell["outputs"].append({
                    "output_type": "error",
                    "ename": body["ename"],
                    "evalue": body["evalue"],
                    "traceback": body["traceback"],
                })
                with path.open("w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(notebook, ensure_ascii=False, indent=1))
                raise RuntimeError(f"Cell {index} failed: {body['ename']}: {body['evalue']}")
        print(f"Executed cell {index}: {len(cell['outputs'])} output(s)", flush=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(notebook, ensure_ascii=False, indent=1))
finally:
    client.stop_channels()
    kernel.shutdown_kernel(now=True)
    del client
    del kernel
