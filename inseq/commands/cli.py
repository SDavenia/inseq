"""Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/commands/transformers_cli.py."""
import sys
from dotenv import load_dotenv

from ..utils import InseqArgumentParser
from .attribute import AttributeCommand
from .attribute_context import AttributeContextCommand
from .attribute_dataset import AttributeDatasetCommand
from .base import BaseCLICommand

COMMANDS: list[BaseCLICommand] = [AttributeCommand, AttributeDatasetCommand, AttributeContextCommand]


def main():
    load_dotenv()
    print(f"Loading environmental variables...")
    print(f"Entering main: ")
    parser = InseqArgumentParser(prog="Inseq CLI tool", usage="inseq <COMMAND> [<ARGS>]")
    command_parser = parser.add_subparsers(title="Inseq CLI command helpers")

    for command_type in COMMANDS:
        print(f"Command type: {command_type}")
        command_type.register_subcommand(command_parser)
    
    # Extract all args from the command line
    args = parser.parse_args()

    if not hasattr(args, "factory_method"):
        parser.print_help()
        sys.exit(1)

    # Extract command and check the args corresponding to that command
    command, command_args = args.factory_method(args)
    # print(f"Command is: {command}") 
    # print(f"Command args is: {command_args}")
    command.run(command_args)


if __name__ == "__main__":
    main()
