# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from dataclasses import dataclass

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs, get_kwargs
from vllm.utils.argparse_utils import FlexibleArgumentParser


@dataclass
class ModelConfigGr(ModelConfig):
    catalog_path: str | None = None
    """ Optionally define a path for valid results file"""


origin_add_cli_args = EngineArgs.add_cli_args
origin_create_model_config = EngineArgs.create_model_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    model_kwargs = get_kwargs(ModelConfigGr)
    model_group = parser.add_argument_group(title="ModelConfig", description=ModelConfigGr.__doc__)
    model_group.add_argument("--catalog-path", **model_kwargs["catalog_path"])
    return parser


def create_model_config(self) -> ModelConfig:
    config = origin_create_model_config(self)
    config.__class__ = ModelConfigGr
    config.catalog_path = getattr(self, "catalog_path", None)
    return config


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    engine_args = origin_from_cli_args(cls, args)
    engine_args.catalog_path = None
    if hasattr(args, "catalog_path"):
        engine_args.catalog_path = args.catalog_path
    return engine_args


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_model_config = create_model_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
