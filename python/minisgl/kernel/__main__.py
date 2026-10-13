assert __name__ == "__main__"


def generate_clangd():
    import argparse
    import json
    import os
    import subprocess

    from minisgl.kernel.utils import DEFAULT_INCLUDE
    from minisgl.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    parser = argparse.ArgumentParser(description="Generate clangd configuration for CUDA kernels")
    parser.add_argument("--cuda-arch", help="CUDA architecture, e.g. sm_80 (auto-detected if available)")
    args = parser.parse_args()

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    cuda_arch = args.cuda_arch
    if cuda_arch is None:
        try:
            status = subprocess.run(
                args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                check=True,
                text=True,
            )
            major, minor = status.stdout.strip().splitlines()[0].split(".")
            cuda_arch = f"sm_{major}{minor}"
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
            cuda_arch = "sm_80"
            logger.warning(
                "Could not detect GPU architecture; using sm_80 for clangd. "
                "Pass --cuda-arch to override it."
            )
    if not cuda_arch.startswith("sm_") or not cuda_arch[3:].isdigit():
        parser.error("--cuda-arch must look like sm_80")

    flags = ["-xcuda", f"--cuda-gpu-arch={cuda_arch}", "-std=c++20", "-Wall", "-Wextra"]
    flags.extend(f"-isystem{path}" for path in dict.fromkeys(include_paths))
    clangd_content = "CompileFlags:\n  Add: " + json.dumps(flags) + "\n"
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()
