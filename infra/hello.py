"""Round-trip test module for infra.remote — proves module CLI execution works.

Prints python version / torch CUDA availability / argv, then writes AND reads
back a sentinel file under config.RESULTS_DIR (= /vol/results on Modal because
the image sets WM_ROOT=/vol; local results/ otherwise).

Run: python -m infra.hello --tag sometag
"""
import argparse
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="infra round-trip hello")
    parser.add_argument("--tag", default="hello")
    args = parser.parse_args()

    print("python:", sys.version.replace("\n", " "))
    print("argv:", sys.argv)

    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch: {torch.__version__} cuda_available: {cuda}")
        if cuda:
            print("cuda_device:", torch.cuda.get_device_name(0))
    except ImportError:
        print("torch: NOT INSTALLED cuda_available: False")

    import config

    sentinel = config.RESULTS_DIR / f"hello_{args.tag}.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        f"tag={args.tag} host={platform.node()} cuda={cuda} "
        f"python={platform.python_version()}\n"
    )
    print("sentinel_written:", sentinel)
    print("sentinel_readback:", sentinel.read_text().strip())
    print("HELLO_OK", args.tag)


if __name__ == "__main__":
    main()
