from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Tuple

from ert._c_wrappers.enkf import EnsembleConfig, RunArg, SummaryConfig
from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum

from .load_status import LoadResult, LoadStatus

CallbackArgs = Tuple[RunArg, EnsembleConfig]
Callback = Callable[[RunArg, EnsembleConfig], LoadResult]

logger = logging.getLogger(__name__)


def _read_parameters(
    run_arg: RunArg, parameter_configuration: List[ParameterConfig]
) -> LoadResult:
    result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    error_msg = ""
    for config_node in parameter_configuration:
        if not config_node.forward_init:
            continue
        try:
            start_time = time.perf_counter()
            logger.info(f"Starting to load parameter: {config_node.name}")
            config_node.load(
                Path(run_arg.runpath), run_arg.iens, run_arg.ensemble_storage
            )
            logger.info(
                f"Saved {config_node.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except ValueError as err:
            error_msg += str(err)
            result = LoadResult(LoadStatus.LOAD_FAILURE, error_msg)
    return result


def _write_responses_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    errors = []
    for config in ens_config.response_configs.values():
        if isinstance(config, SummaryConfig):
            # Nothing to load, should not be handled here, should never be
            # added in the first place
            if not config.keys:
                continue
        try:
            start_time = time.perf_counter()
            logger.info(f"Starting to load response: {config.name}")
            ds = config.read_from_file(run_arg.runpath, run_arg.iens)
            run_arg.ensemble_storage.save_response(config.name, ds, run_arg.iens)
            logger.info(
                f"Saved {config.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except ValueError as err:
            errors.append(str(err))
    if errors:
        return LoadResult(LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def forward_model_ok(
    run_arg: RunArg,
    ens_conf: EnsembleConfig,
) -> LoadResult:
    parameters_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    response_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    try:
        # We only read parameters after the prior, after that, ERT
        # handles parameters
        if run_arg.itr == 0:
            parameters_result = _read_parameters(
                run_arg, ens_conf.parameter_configuration
            )

        if parameters_result.status == LoadStatus.LOAD_SUCCESSFUL:
            response_result = _write_responses_to_storage(ens_conf, run_arg)

    except Exception as err:
        logging.exception(f"Failed to load results for realization {run_arg.iens}")
        parameters_result = LoadResult(
            LoadStatus.LOAD_FAILURE,
            "Failed to load results for realization "
            f"{run_arg.iens}, failed with: {err}",
        )

    final_result = parameters_result
    if response_result.status != LoadStatus.LOAD_SUCCESSFUL:
        final_result = response_result

    run_arg.ensemble_storage.state_map[run_arg.iens] = (
        RealizationStateEnum.STATE_HAS_DATA
        if final_result.status == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )

    return final_result


def forward_model_exit(run_arg: RunArg, _: EnsembleConfig) -> LoadResult:
    run_arg.ensemble_storage.state_map[
        run_arg.iens
    ] = RealizationStateEnum.STATE_LOAD_FAILURE
    return LoadResult(None, "")
