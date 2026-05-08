import argparse
import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "train": ("train", "Train the affinity model."),
    "predict": ("predict", "Predict affinity maps from HDF raw volumes."),
    "segment": ("segment", "Convert affinity maps to neuron instances."),
}


def print_help() -> None:
    parser = argparse.ArgumentParser(description="CREMI affinity segmentation utilities.")
    parser.add_argument("command", choices=COMMANDS, help="Command to run.")
    parser.print_help()
    print("\nCommands:")
    for command, (_, description) in COMMANDS.items():
        print(f"  {command:<8} {description}")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print_help()
        return

    command = args[0]
    if command not in COMMANDS:
        valid = ", ".join(COMMANDS)
        raise SystemExit(f'Unknown command "{command}". Expected one of: {valid}')

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    module.main(args[1:])


if __name__ == "__main__":
    main()
