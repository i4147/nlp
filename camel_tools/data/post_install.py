# -*- coding: utf-8 -*-


from pathlib import Path
from typing import List

from muddler import unmuddle

__all__ = ("post_install",)


def parse_args(arg_types: List[str], args: List[str]):
    result = []
    for arg_type, arg in zip(arg_types, args):
        if arg_type == "string":
            result.append(arg)
        elif arg_type == "path":
            result.append(Path(arg))
        elif arg_type == "int":
            result.append(int(arg))
        elif arg_type == "float":
            result.append(float(arg))
        elif arg_type == "bool":
            arg_lower = arg.lower()
            if arg_lower == "true":
                result.append(True)
            elif arg_lower == "false":
                result.append(False)
            else:
                result.append(False)

    return result


def post_install(config: dict, package_path: Path, args: List[str]):
    parsed_args = parse_args(config["args"], args)

    for step in config["steps"]:
        if step["action"] == "unmuddle":
            source_path = parsed_args[step["source_path_index"]]
            muddled_path = Path(package_path, step["muddled_path"])
            output_path = Path(package_path, step["output_path"])
            unmuddle(str(source_path), str(muddled_path), str(output_path))
