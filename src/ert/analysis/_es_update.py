from __future__ import annotations

import logging
import time
from collections import UserDict
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import iterative_ensemble_smoother as ies
import numpy as np
import xarray as xr
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)
from tqdm import tqdm

from ert.config import Field, GenKwConfig, SurfaceConfig
from ert.realization_state import RealizationState

from .row_scaling import RowScaling
from .update import Parameter, RowScalingParameter

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.analysis.configuration import UpdateConfiguration, UpdateStep
    from ert.config import AnalysisConfig, AnalysisModule
    from ert.storage import EnsembleAccessor, EnsembleReader

_logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


@dataclass
class ObservationAndResponseSnapshot:
    obs_name: str
    obs_val: float
    obs_std: float
    response_mean: float
    response_std: float
    response_mean_mask: bool
    response_std_mask: bool

    def __post_init__(self) -> None:
        status = "Active"
        if np.isnan(self.response_mean):
            status = "Deactivated, missing response(es)"
        elif not self.response_std_mask:
            status = f"Deactivated, ensemble std ({self.response_std:.3f}) > STD_CUTOFF"
        elif not self.response_mean_mask:
            status = "Deactivated, outlier"
        self.status = status


@dataclass
class SmootherSnapshot:
    source_case: str
    target_case: str
    analysis_module: str
    analysis_configuration: Dict[str, Any]
    alpha: float
    std_cutoff: float
    update_step_snapshots: Dict[str, List[ObservationAndResponseSnapshot]] = field(
        default_factory=dict
    )


@dataclass
class Progress:
    task: Task
    sub_task: Optional[Task]

    def __str__(self) -> str:
        ret = f"tasks: {self.task}"
        if self.sub_task is not None:
            ret += f"\nsub tasks: {self.sub_task}"
        return ret


@dataclass
class Task:
    description: str
    current: int
    total: Optional[int]

    def __str__(self) -> str:
        ret: str = f"running #{self.current}"
        if self.total is not None:
            ret += f" of {self.total}"
        return ret + f" - {self.description}"


ProgressCallback = Callable[[Progress], None]


def noop_progress_callback(_: Progress) -> None:
    pass


class TempStorage(UserDict):  # type: ignore
    def __getitem__(self, key: str) -> npt.NDArray[np.double]:
        value: Union[npt.NDArray[np.double], xr.DataArray] = self.data[key]
        if not isinstance(value, xr.DataArray):
            return value
        ensemble_size = len(value.realizations)
        return value.values.reshape(ensemble_size, -1).T

    def __setitem__(
        self, key: str, value: Union[npt.NDArray[np.double], xr.DataArray]
    ) -> None:
        old_value = self.data.get(key)
        if isinstance(old_value, xr.DataArray):
            old_value.data = value.T.reshape(*old_value.shape)
            self.data[key] = old_value
        else:
            self.data[key] = value

    def get_xr_array(self, key: str, real: int) -> xr.DataArray:
        value = self.data[key]
        if isinstance(value, xr.DataArray):
            return value[real]
        else:
            raise ValueError(f"TempStorage has no xarray DataFrame with key={key}")


def _param_ensemble_for_projection(
    source_fs: EnsembleReader,
    iens_active_index: npt.NDArray[np.int_],
    ensemble_size: int,
    param_groups: List[str],
    tot_num_params: int,
) -> Optional[npt.NDArray[np.double]]:
    """Responses must be projected when num_params < ensemble_size - 1.
    The number of parameters here refers to the total number of parameters used to
    drive the model and not the number of parameters in a local update step.

    Scenario 1:
        - p < N - 1, meaning that projection needs to be done.
        - User wants to loop through each parameter and update it separately,
          perhaps to do distance based localization.
        In this case, we need to pass `param_ensemble` to the smoother
        so that the responses are projected using the same `param_ensemble`
        in every loop.
    Scenario 2:
        - p > N - 1, meaning that projection is not necessary.
        - User wants to loop through every parameter as in Scenario 1.
        In this case `param_ensemble` should be `None` which means
        that no projection will be done even when updating a single parameter.
    """
    if tot_num_params < ensemble_size - 1:
        temp_storage = TempStorage()
        for param_group in param_groups:
            _temp_storage = _create_temporary_parameter_storage(
                source_fs, iens_active_index, param_group
            )
            temp_storage[param_group] = _temp_storage[param_group]
        matrices = [temp_storage[p] for p in param_groups]
        return np.vstack(matrices) if matrices else None
    return None


def _get_params_with_row_scaling(
    temp_storage: TempStorage,
    parameters: List[RowScalingParameter],
) -> List[Tuple[npt.NDArray[np.double], RowScaling]]:
    matrices = []
    for p in parameters:
        if p.index_list is None:
            matrices.append((temp_storage[p.name].astype(np.double), p.row_scaling))
        else:
            matrices.append(
                (
                    temp_storage[p.name][p.index_list, :].astype(np.double),
                    p.row_scaling,
                ),
            )

    return matrices


def _save_to_temp_storage(
    temp_storage: TempStorage,
    parameters: Sequence[Union[Parameter, RowScalingParameter]],
    A: Optional["npt.NDArray[np.double]"],
) -> None:
    if A is None:
        return
    offset = 0
    for p in parameters:
        if p.index_list is None:
            rows = temp_storage[p.name].shape[0]
            temp_storage[p.name] = A[offset : offset + rows, :]
            offset += rows
        else:
            row_indices = p.index_list
            for i, row in enumerate(row_indices):
                temp_storage[p.name][row] = A[offset + i]
            offset += len(row_indices)


def _save_temp_storage_to_disk(
    target_fs: EnsembleAccessor,
    temp_storage: TempStorage,
    iens_active_index: npt.NDArray[np.int_],
) -> None:
    for key, matrix in temp_storage.items():
        config_node = target_fs.experiment.parameter_configuration[key]
        for i, realization in enumerate(iens_active_index):
            if isinstance(config_node, GenKwConfig):
                assert isinstance(matrix, np.ndarray)
                dataset = xr.Dataset(
                    {
                        "values": ("names", matrix[:, i]),
                        "transformed_values": (
                            "names",
                            config_node.transform(matrix[:, i]),
                        ),
                        "names": [e.name for e in config_node.transfer_functions],
                    }
                )
                target_fs.save_parameters(key, realization, dataset)
            elif isinstance(config_node, (Field, SurfaceConfig)):
                _matrix = temp_storage.get_xr_array(key, i)
                assert isinstance(_matrix, xr.DataArray)
                target_fs.save_parameters(key, realization, _matrix.to_dataset())
            else:
                raise NotImplementedError(f"{type(config_node)} is not supported")
    target_fs.sync()


def _create_temporary_parameter_storage(
    source_fs: Union[EnsembleReader, EnsembleAccessor],
    iens_active_index: npt.NDArray[np.int_],
    param_group: str,
) -> TempStorage:
    temp_storage = TempStorage()
    t_genkw = 0.0
    t_surface = 0.0
    t_field = 0.0
    _logger.debug("_create_temporary_parameter_storage() - start")
    config_node = source_fs.experiment.parameter_configuration[param_group]
    matrix: Union[npt.NDArray[np.double], xr.DataArray]
    if isinstance(config_node, GenKwConfig):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index).values.T
        t_genkw += time.perf_counter() - t
    elif isinstance(config_node, SurfaceConfig):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index)
        t_surface += time.perf_counter() - t
    elif isinstance(config_node, Field):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index)
        t_field += time.perf_counter() - t
    else:
        raise NotImplementedError(f"{type(config_node)} is not supported")
    temp_storage[param_group] = matrix
    _logger.debug(
        f"_create_temporary_parameter_storage() time_used gen_kw={t_genkw:.4f}s, \
                surface={t_surface:.4f}s, field={t_field:.4f}s"
    )
    return temp_storage


def _get_obs_and_measure_data(
    source_fs: EnsembleReader,
    selected_observations: List[Tuple[str, Optional[List[int]]]],
    iens_active_index: npt.NDArray[np.int_],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.str_],
]:
    measured_data = []
    observation_keys = []
    observation_values = []
    observation_errors = []
    observations = source_fs.experiment.observations
    for obs_key, obs_active_list in selected_observations:
        observation = observations[obs_key]
        group = observation.attrs["response"]
        if obs_active_list:
            index = observation.coords.to_index()[obs_active_list]
            sub_selection = {
                name: list(set(index.get_level_values(name))) for name in index.names
            }
            observation = observation.sel(sub_selection)
        ds = source_fs.load_response(group, tuple(iens_active_index))
        try:
            filtered_ds = observation.merge(ds, join="left")
        except KeyError as e:
            raise ErtAnalysisError(
                f"Mismatched index for: "
                f"Observation: {obs_key} attached to response: {group}"
            ) from e

        observation_keys.append([obs_key] * len(filtered_ds.observations.data.ravel()))
        observation_values.append(filtered_ds["observations"].data.ravel())
        observation_errors.append(filtered_ds["std"].data.ravel())
        measured_data.append(
            filtered_ds["values"]
            .transpose(..., "realization")
            .values.reshape((-1, len(filtered_ds.realization)))
        )
    source_fs.load_response.cache_clear()
    return (
        np.concatenate(measured_data, axis=0),
        np.concatenate(observation_values),
        np.concatenate(observation_errors),
        np.concatenate(observation_keys),
    )


def _load_observations_and_responses(
    source_fs: EnsembleReader,
    alpha: float,
    std_cutoff: float,
    global_std_scaling: float,
    iens_ative_index: npt.NDArray[np.int_],
    selected_observations: List[Tuple[str, Optional[List[int]]]],
) -> Any:
    S, observations, errors, obs_keys = _get_obs_and_measure_data(
        source_fs,
        selected_observations,
        iens_ative_index,
    )

    # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    # in for example evensen2018 - Analysis of iterative ensemble smoothers for
    # solving inverse problems.
    # `global_std_scaling` is 1.0 for ES.
    errors *= sqrt(global_std_scaling)

    ens_mean = S.mean(axis=1)
    ens_std = S.std(ddof=0, axis=1)

    ens_std_mask = ens_std > std_cutoff
    ens_mean_mask = abs(observations - ens_mean) <= alpha * (ens_std + errors)
    obs_mask = np.logical_and(ens_mean_mask, ens_std_mask)

    update_snapshot = []
    for (
        obs_name,
        obs_val,
        obs_std,
        response_mean,
        response_std,
        response_mean_mask,
        response_std_mask,
    ) in zip(
        obs_keys,
        observations,
        errors,
        ens_mean,
        ens_std,
        ens_mean_mask,
        ens_std_mask,
    ):
        update_snapshot.append(
            ObservationAndResponseSnapshot(
                obs_name=obs_name,
                obs_val=obs_val,
                obs_std=obs_std,
                response_mean=response_mean,
                response_std=response_std,
                response_mean_mask=response_mean_mask,
                response_std_mask=response_std_mask,
            )
        )

    for missing_obs in obs_keys[~obs_mask]:
        _logger.warning(f"Deactivating observation: {missing_obs}")

    return S[obs_mask], (
        observations[obs_mask],
        errors[obs_mask],
        update_snapshot,
    )


def _split_by_batchsize(
    arr: npt.NDArray[np.int_], batch_size: int
) -> List[npt.NDArray[np.int_]]:
    return np.array_split(arr, int((arr.shape[0] / batch_size)) + 1)


def _update_with_row_scaling(
    update_step: UpdateStep,
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    iens_active_index: npt.NDArray[np.int_],
    S: npt.NDArray[np.float_],
    observation_errors: npt.NDArray[np.float_],
    observation_values: npt.NDArray[np.float_],
    noise: npt.NDArray[np.float_],
    truncation: float,
    inversion: int,
    progress_callback: ProgressCallback,
) -> None:
    for param_group in update_step.row_scaling_parameters:
        source: Union[EnsembleReader, EnsembleAccessor]
        if target_fs.has_parameter_group(param_group.name):
            source = target_fs
        else:
            source = source_fs
        temp_storage = _create_temporary_parameter_storage(
            source, iens_active_index, param_group.name
        )
        params_with_row_scaling = _get_params_with_row_scaling(
            temp_storage, update_step.row_scaling_parameters
        )
        params_with_row_scaling = ensemble_smoother_update_step_row_scaling(
            S,
            params_with_row_scaling,
            observation_errors,
            observation_values,
            noise,
            truncation,
            ies.InversionType(inversion),
        )
        for row_scaling_parameter, (A, _) in zip(
            update_step.row_scaling_parameters, params_with_row_scaling
        ):
            _save_to_temp_storage(temp_storage, [row_scaling_parameter], A)

        progress_callback(Progress(Task("Storing data", 3, 3), None))
        _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)



#### TED-LASSO ####
def _parameter_ensemble(
    source_fs: EnsembleReader,
    iens_active_index: npt.NDArray[np.int_],
    param_groups: List[str],
) -> npt.NDArray[np.double]:
    temp_storage = TempStorage()
    for param_group in param_groups:
        _temp_storage = _create_temporary_parameter_storage(
            source_fs, iens_active_index, param_group
        )
        temp_storage[param_group] = _temp_storage[param_group]
    matrices = [temp_storage[p] for p in param_groups]
    return np.vstack(matrices) if matrices else None


def ensemble_gaussian_update(X, cov_yx, cov_y, cov_eps, Y, d):
    # X: nxp
    # cov_yx: mxp
    X_tmp = X.copy()
    n = X_tmp.shape[0]
    p = X_tmp.shape[1]
    m = cov_eps.shape[0]
    K = cov_yx @ np.linalg.inv(cov_y + cov_eps) # p x m
    for i in range(n):
        d_i = np.array([np.random.normal(loc=d[k], scale=np.sqrt(cov_eps[k,k])) for k in range(m)])
        X_tmp[i,:] = X_tmp[i,:] + K @ (Y[i:,] - d_i)
    return X_tmp


from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

def linear_l1_regression(X, Y):
    n, p = X.shape #  p: number of features
    n, m = Y.shape #  m: number of y responses

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    coefficients_matrix = np.zeros((p,m))

    # loop over features
    for j in range(m):
    # pick out y_j
        y_j = Y[:,j]
        
        # learn regularization
        eps = 1e-3  # the smaller it is the longer is the path
        max_iter = 10000
        model_cv = LassoCV(cv=10, fit_intercept=False, max_iter=max_iter, eps=eps,)
        model_cv.fit(X_scaled, y_j)
        alpha_optimal = model_cv.alpha_

        # learn lasso model
        model = Lasso(alpha=alpha_optimal, fit_intercept=False, max_iter=max_iter)
        model.fit(X_scaled, y_j)

        # extract coefficients
        coefficients_matrix[:,j] = model.coef_
    return coefficients_matrix, scaler

def compute_cross_covariance(coefficient_matrix, scaler):
    """Compute cross-covariance between parameters and responses"""
    p = coefficient_matrix.shape[0]
    cov_x = np.eye(p) # gen_kw originally sampled from standard normal
    inv_sd = np.diag(1.0/scaler.scale_)
    H = inv_sd @ coefficient_matrix
    return H @ cov_x


def analysis_ES(
    updatestep: UpdateConfiguration,
    rng: np.random.Generator,
    module: AnalysisModule,
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    progress_callback: ProgressCallback,
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    tot_num_params = sum(
        len(source_fs.experiment.parameter_configuration[key])
        for key in source_fs.experiment.parameter_configuration
    )
    param_groups = list(source_fs.experiment.parameter_configuration.keys())
    ensemble_size = ens_mask.sum()
    param_ensemble = _param_ensemble_for_projection(
        source_fs, iens_active_index, ensemble_size, param_groups, tot_num_params
    )

    for update_step in updatestep:
        progress_callback(Progress(Task("Loading data", 1, 3), None))
        try:
            S, (
                observation_values,
                observation_errors,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                alpha,
                std_cutoff,
                global_scaling,
                iens_active_index,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(e) from e

        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot

        num_obs = len(observation_values)
        if num_obs == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        smoother = ies.ES()
        truncation = module.get_truncation()
        noise = rng.standard_normal(size=(num_obs, ensemble_size))

        # TED-LASSO
        # input X, Y, d, eps
        X = _parameter_ensemble(
            source_fs, iens_active_index, param_groups
        ).T
        Y = S.T
        d = observation_values
        cov_y = np.cov(Y)
        cov_eps = np.diag(observation_errors)
        # learn cross-covariance
        coefficient_matrix, scaler = linear_l1_regression(X, Y)
        cov_yx = compute_cross_covariance(coefficient_matrix, scaler)

        for param_group in update_step.parameters:
            source: Union[EnsembleReader, EnsembleAccessor]
            if target_fs.has_parameter_group(param_group.name):
                source = target_fs
            else:
                source = source_fs
            temp_storage = _create_temporary_parameter_storage(
                source, iens_active_index, param_group.name
            )
            if module.localization():
                Y_prime = S - S.mean(axis=1, keepdims=True)
                Sigma_Y = np.std(S, axis=1, ddof=1)
                batch_size: int = 1000
                correlation_threshold = module.localization_correlation_threshold(
                    ensemble_size
                )
                # for parameter in update_step.parameters:
                num_params = temp_storage[param_group.name].shape[0]

                print(
                    (
                        f"Running localization on {num_params} parameters,",
                        f"{num_obs} responses and {ensemble_size} realizations...",
                    )
                )
                batches = _split_by_batchsize(np.arange(0, num_params), batch_size)
                for param_batch_idx in tqdm(batches):
                    X_local = temp_storage[param_group.name][param_batch_idx, :]
                    # Parameter standard deviations
                    Sigma_A = np.std(X_local, axis=1, ddof=1)
                    # Cross-covariance between parameters and measurements
                    A = X_local - X_local.mean(axis=1, keepdims=True)
                    C_AY = A @ Y_prime.T / (ensemble_size - 1)
                    # Cross-correlation between parameters and measurements
                    c_AY = np.abs(
                        (C_AY / Sigma_Y.reshape(1, -1)) / Sigma_A.reshape(-1, 1)
                    )
                    # Absolute values of the correlation matrix
                    c_bool = c_AY > correlation_threshold
                    # Some parameters might be significantly correlated
                    # to the exact same responses.
                    # We want to call the update only once per such parameter group
                    # to speed up computation.
                    # Here we create a collection of unique sets of parameter-to-observation
                    # correlations.
                    param_correlation_sets: npt.NDArray[np.bool_] = np.unique(
                        c_bool, axis=0
                    )
                    # Drop the correlation set that does not correlate to any responses.
                    row_with_all_false = np.all(~param_correlation_sets, axis=1)
                    param_correlation_sets = param_correlation_sets[~row_with_all_false]

                    for param_correlation_set in param_correlation_sets:
                        # Find the rows matching the parameter group
                        matching_rows = np.all(c_bool == param_correlation_set, axis=1)
                        # Get the indices of the matching rows
                        row_indices = np.where(matching_rows)[0]
                        X_chunk = temp_storage[param_group.name][param_batch_idx, :][
                            row_indices, :
                        ]
                        S_chunk = S[param_correlation_set, :]
                        observation_errors_loc = observation_errors[
                            param_correlation_set
                        ]
                        observation_values_loc = observation_values[
                            param_correlation_set
                        ]
                        smoother.fit(
                            S_chunk,
                            observation_errors_loc,
                            observation_values_loc,
                            noise=noise[param_correlation_set],
                            truncation=truncation,
                            inversion=ies.InversionType(module.inversion),
                            param_ensemble=param_ensemble,
                        )
                        temp_storage[param_group.name][
                            param_batch_idx[row_indices], :
                        ] = smoother.update(X_chunk)
            else:
                # smoother.fit(
                #     S,
                #     observation_errors,
                #     observation_values,
                #     noise=noise,
                #     truncation=truncation,
                #     inversion=ies.InversionType(module.inversion),
                #     param_ensemble=param_ensemble,
                # )
                if active_indices := param_group.index_list:
                    X_group = temp_storage[param_group.name][active_indices, :]
                    temp_storage[param_group.name][active_indices, :] = ensemble_gaussian_update(
                        X_group.T, cov_yx, cov_y, cov_eps, Y, d
                    ).T
                    # temp_storage[param_group.name][active_indices, :] = smoother.update(
                    #     temp_storage[param_group.name][active_indices, :]
                    # )
                else:
                    X_group = temp_storage[param_group.name]
                    temp_storage[param_group.name] = ensemble_gaussian_update(
                        X_group.T, cov_yx, cov_y, cov_eps, Y, d
                    ).T
                    # temp_storage[param_group.name] = smoother.update(
                    #     temp_storage[param_group.name]
                    # )

            progress_callback(Progress(Task("Storing data", 3, 3), None))
            _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)

        _update_with_row_scaling(
            update_step,
            source_fs,
            target_fs,
            iens_active_index,
            S,
            observation_errors,
            observation_values,
            noise,
            truncation,
            module.inversion,
            progress_callback,
        )


def analysis_IES(
    updatestep: UpdateConfiguration,
    rng: np.random.Generator,
    module: AnalysisModule,
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    iterative_ensemble_smoother: ies.SIES,
    progress_callback: ProgressCallback,
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    tot_num_params = sum(
        len(source_fs.experiment.parameter_configuration[key])
        for key in source_fs.experiment.parameter_configuration
    )
    param_groups = list(source_fs.experiment.parameter_configuration.keys())
    ensemble_size = ens_mask.sum()
    param_ensemble = _param_ensemble_for_projection(
        source_fs, iens_active_index, ensemble_size, param_groups, tot_num_params
    )

    for update_step in updatestep:
        progress_callback(Progress(Task("Loading data", 1, 3), None))
        try:
            S, (
                observation_values,
                observation_errors,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                alpha,
                std_cutoff,
                global_scaling,
                iens_active_index,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(str(e)) from e
        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        noise = rng.standard_normal(size=(len(observation_values), S.shape[1]))
        iterative_ensemble_smoother.fit(
            S,
            observation_errors,
            observation_values,
            noise=noise,
            ensemble_mask=ens_mask,
            inversion=ies.InversionType(module.inversion),
            truncation=module.get_truncation(),
            param_ensemble=param_ensemble,
        )
        for param_group in update_step.parameters:
            source: Union[EnsembleReader, EnsembleAccessor] = target_fs
            try:
                target_fs.load_parameters(group=param_group.name, realizations=0)
            except Exception:
                source = source_fs
            temp_storage = _create_temporary_parameter_storage(
                source, iens_active_index, param_group.name
            )
            progress_callback(Progress(Task("Updating data", 2, 3), None))
            if active_indices := param_group.index_list:
                temp_storage[param_group.name][
                    active_indices, :
                ] = iterative_ensemble_smoother.update(
                    temp_storage[param_group.name][active_indices, :]
                )
            else:
                temp_storage[param_group.name] = iterative_ensemble_smoother.update(
                    temp_storage[param_group.name]
                )

            progress_callback(Progress(Task("Storing data", 3, 3), None))
            _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)


def _write_update_report(
    path: Path, snapshot: SmootherSnapshot, run_id: str, global_scaling: float
) -> None:
    fname = path / f"{run_id}.txt"
    fname.parent.mkdir(parents=True, exist_ok=True)
    for update_step_name, update_step in snapshot.update_step_snapshots.items():
        with open(fname, "w", encoding="utf-8") as fout:
            fout.write("=" * 150 + "\n")
            timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            fout.write(f"Time: {timestamp}\n")
            fout.write(f"Parent ensemble: {snapshot.source_case}\n")
            fout.write(f"Target ensemble: {snapshot.target_case}\n")
            fout.write(f"Alpha: {snapshot.alpha}\n")
            fout.write(f"Global scaling: {global_scaling}\n")
            fout.write(f"Standard cutoff: {snapshot.std_cutoff}\n")
            fout.write(f"Run id: {run_id}\n")
            fout.write(f"Update step: {update_step_name:<10}\n")
            fout.write("-" * 150 + "\n")
            fout.write(
                "Observed history".rjust(56)
                + "|".rjust(12)
                + "Simulated data".rjust(27)
                + "|".rjust(13)
                + "Status".rjust(12)
                + "\n"
            )
            fout.write("-" * 150 + "\n")
            for nr, step in enumerate(update_step):
                fout.write(
                    f"{nr+1:^6}: {step.obs_name:20} {step.obs_val:>16.3f} +/- "
                    f"{step.obs_std:<16.3f} | {step.response_mean:>16.3f} +/- "
                    f"{step.response_std:<16.3f} {'|':<6} "
                    f"{step.status.capitalize()}\n"
                )


def _assert_has_enough_realizations(
    ens_mask: npt.NDArray[np.bool_], analysis_config: "AnalysisConfig"
) -> None:
    active_realizations = ens_mask.sum()
    if not analysis_config.have_enough_realisations(active_realizations):
        raise ErtAnalysisError(
            f"There are {active_realizations} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )


def _create_smoother_snapshot(
    prior_name: "str", posterior_name: "str", analysis_config: "AnalysisConfig"
) -> SmootherSnapshot:
    active_module = analysis_config.active_module()
    return SmootherSnapshot(
        prior_name,
        posterior_name,
        active_module.name,
        active_module.variable_value_dict(),
        analysis_config.enkf_alpha,
        analysis_config.std_cutoff,
    )


def smoother_update(
    prior_storage: EnsembleReader,
    posterior_storage: EnsembleAccessor,
    run_id: str,
    updatestep: UpdateConfiguration,
    analysis_config: AnalysisConfig,
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[ProgressCallback] = None,
    global_scaling: float = 1.0,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if not rng:
        rng = np.random.default_rng()
    alpha = analysis_config.enkf_alpha
    std_cutoff = analysis_config.std_cutoff
    ens_mask = prior_storage.get_realization_mask_from_state(
        [RealizationState.HAS_DATA]
    )
    _assert_has_enough_realizations(ens_mask, analysis_config)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name, posterior_storage.name, analysis_config
    )

    analysis_ES(
        updatestep,
        rng,
        analysis_config.active_module(),
        alpha,
        std_cutoff,
        global_scaling,
        smoother_snapshot,
        ens_mask,
        prior_storage,
        posterior_storage,
        progress_callback,
    )

    _write_update_report(
        Path(analysis_config.log_path),
        smoother_snapshot,
        run_id,
        global_scaling,
    )

    return smoother_snapshot


def iterative_smoother_update(
    prior_storage: EnsembleReader,
    posterior_storage: EnsembleAccessor,
    w_container: ies.SIES,
    run_id: str,
    updatestep: UpdateConfiguration,
    analysis_config: AnalysisConfig,
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if not rng:
        rng = np.random.default_rng()

    if len(updatestep) > 1:
        raise ErtAnalysisError(
            "Can not combine IES_ENKF modules with multi step updates"
        )

    alpha = analysis_config.enkf_alpha
    std_cutoff = analysis_config.std_cutoff
    ens_mask = prior_storage.get_realization_mask_from_state(
        [RealizationState.HAS_DATA]
    )

    _assert_has_enough_realizations(ens_mask, analysis_config)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name, posterior_storage.name, analysis_config
    )

    analysis_IES(
        updatestep,
        rng,
        analysis_config.active_module(),
        alpha,
        std_cutoff,
        1.0,
        smoother_snapshot,
        ens_mask,
        prior_storage,
        posterior_storage,
        w_container,
        progress_callback,
    )

    _write_update_report(
        Path(analysis_config.log_path),
        smoother_snapshot,
        run_id,
        global_scaling=1.0,
    )

    return smoother_snapshot
