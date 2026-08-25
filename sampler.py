"""初始化、执行并整理MCMC采样结果。"""

from __future__ import annotations
from dataclasses import dataclass,field
from multiprocessing import Pool
from typing import TYPE_CHECKING,Any
from uuid import uuid4
import numpy as np

from config import InputCard
if TYPE_CHECKING:
    from posterior import PosteriorCalculator

_WORKER_POSTERIOR: PosteriorCalculator | None = None
_WORKER_STUDY_NAME = ""

def _initialize_worker(posterior: PosteriorCalculator,study_name: str) -> None:
    """为每个工作进程持久保存后验计算器"""

    # 初始化函数结束后 后验计算器仍然保存在工作进程里 不会随着函数结束而丢失
    global _WORKER_POSTERIOR,_WORKER_STUDY_NAME
    _WORKER_POSTERIOR = posterior
    _WORKER_STUDY_NAME = study_name

def _worker_log_probability(theta: dict[str,float]) -> float:
    """在工作进程中计算对数后验"""

    # 多进程模式下真正调用的后验概率函数
    if _WORKER_POSTERIOR is None:
        raise RuntimeError("工作进程的后验计算器未初始化")
    procedure_file_label = (f"{_WORKER_STUDY_NAME}_{uuid4().hex}")

    return _WORKER_POSTERIOR(theta,procedure_file_label)

@dataclass
class SamplerResult:
    """采样结果与诊断信息。"""

    parameter_names: list[str]
    samples: np.ndarray
    log_probability: np.ndarray
    chain: np.ndarray
    log_probability_chain: np.ndarray
    steps: np.ndarray
    acceptance_fraction: np.ndarray
    autocorrelation_time: np.ndarray
    effective_sample_size: np.ndarray
    chain_long_enough: np.ndarray
    converged: bool
    metadata: dict[str,Any] = field(default_factory=dict)

class Sampler:
    """统一采样器入口与初始walker生成器"""

    def __init__(
        self,
        sampler_config: InputCard.Sampler,
        study_name: str,
    ):
        self.sampler_type = sampler_config.type
        if sampler_config.emcee is None:
            raise ValueError("sampler.emcee 不能为空")
        self.n_walkers = sampler_config.emcee.n_walkers
        self.initial_spread = sampler_config.initial_spread
        self.sampler_dict = {
            "emcee": EmceeSampler(
                sampler_config,
                study_name,
            ),
        }

    def sample(
        self,
        parameter_configs: list[InputCard.Parameter],
        posterior: PosteriorCalculator,
        random_seed: int,
    ) -> SamplerResult:
        """生成初态并调用指定采样器"""

        sampler = self.sampler_dict.get(self.sampler_type)
        if sampler is None:
            raise NotImplementedError(
                f"未实现的采样器: {self.sampler_type}"
            )

        random_state = np.random.RandomState(random_seed)
        parameter_names,initial_state = self._initial_state(parameter_configs,posterior,random_state)
        result = sampler.sample(
            parameter_names,
            initial_state,
            random_state.get_state(),
            posterior,
        )
        result.metadata["random_seed"] = random_seed

        return result

    def _initial_state(
        self,
        parameter_configs: list[InputCard.Parameter],
        posterior: PosteriorCalculator,
        random_state: np.random.RandomState,
    ) -> tuple[list[str],np.ndarray]:
        """在先验支撑集内生成walker初始位置"""

        parameter_names = [config.name for config in parameter_configs]
        center = np.asarray([config.initial for config in parameter_configs],dtype=float)
        spread = np.asarray([self.initial_spread[name] for name in parameter_names],dtype=float)
        dimension = len(parameter_names)
        initial_state = np.empty((self.n_walkers,dimension),dtype=float)

        accepted = 0
        attempts = 0
        maximum_attempts = 1000*self.n_walkers

        while accepted < self.n_walkers:
            candidate = center + spread*random_state.randn(dimension)
            theta = dict(zip(parameter_names,candidate,strict=True))

            attempts += 1
            if np.isfinite(posterior.prior.log_prior(theta)):
                initial_state[accepted] = candidate
                accepted += 1

            if attempts >= maximum_attempts:
                raise RuntimeError(
                    "无法在先验支撑集内生成足够的walker初态"
                    "请检查参数初始值与initial_spread"
                )

        return parameter_names,initial_state

class EmceeSampler:
    """emcee集成MCMC采样器"""

    def __init__(
        self,
        sampler_config: InputCard.Sampler,
        study_name: str,
    ):
        if sampler_config.emcee is None:
            raise ValueError("sampler.emcee 不能为空")

        self.study_name = study_name
        self.n_walkers = sampler_config.emcee.n_walkers
        self.move_weights = sampler_config.emcee.move_weights
        self.n_processes = sampler_config.n_processes
        self.burn_in = sampler_config.burn_in
        self.production_steps = sampler_config.production_steps
        self.thin = sampler_config.thin

    def sample(
        self,
        parameter_names: list[str],
        initial_state: np.ndarray,
        random_state: tuple,
        posterior: PosteriorCalculator,
    ) -> SamplerResult:
        """执行burn-in与正式采样"""

        if self.n_processes == 1:
            sampler = self._create_sampler(parameter_names,posterior,pool=None)
            return self._run(sampler,parameter_names,initial_state,random_state)

        with Pool(
            self.n_processes,
            initializer=_initialize_worker,
            initargs=(posterior,self.study_name),
        ) as pool:
            sampler = self._create_sampler(
                parameter_names,
                posterior=None,
                pool=pool,
            )
            result = self._run(
                sampler,
                parameter_names,
                initial_state,
                random_state,
            )

        return result

    def _create_sampler(
        self,
        parameter_names: list[str],
        posterior: PosteriorCalculator | None,
        pool,
    ):
        """创建emcee采样器"""

        import emcee

        if pool is None:
            log_probability = self._log_probability
            args = [posterior]
        else:
            log_probability = _worker_log_probability
            args = []

        return emcee.EnsembleSampler(
            self.n_walkers,
            len(parameter_names),
            log_probability,
            args=args,
            parameter_names=parameter_names,
            pool=pool,
            moves=self._moves(emcee),
        )

    def _moves(self,emcee):
        """按照配置权重构造emcee提议方法"""

        configured_moves = (
            (emcee.moves.StretchMove,self.move_weights.stretch),
            (emcee.moves.DEMove,self.move_weights.de),
            (emcee.moves.DESnookerMove,self.move_weights.de_snooker),
        )

        return [
            (move(),weight)
            for move,weight in configured_moves
            if weight > 0.0
        ]

    def _run(
        self,
        sampler,
        parameter_names: list[str],
        initial_state: np.ndarray,
        random_state: tuple,
    ) -> SamplerResult:
        """运行emcee并组装结果"""

        import emcee

        # 初始化
        state = emcee.State(initial_state,random_state=random_state)

        # burn-in阶段
        state = sampler.run_mcmc(
            state,
            self.burn_in,
            store=False,
            progress=True,
            progress_kwargs={"desc":"Burn-in"},
        )
        sampler.reset()

        # 正式采样阶段
        sampler.run_mcmc(
            state,
            self.production_steps,
            progress=True,
            progress_kwargs={"desc":"Sampling"},
        )

        return self._result(sampler,parameter_names)

    def _result(
        self,
        sampler,
        parameter_names: list[str],
    ) -> SamplerResult:
        """整理采样链与诊断信息"""

        dimension = len(parameter_names)
        chain = sampler.get_chain(thin=self.thin,flat=False)
        log_probability_chain = sampler.get_log_prob(thin=self.thin,flat=False)
        samples = chain.reshape(-1,dimension)
        log_probability = log_probability_chain.reshape(-1)
        steps = (self.burn_in + self.thin*np.arange(1,chain.shape[0]+1))
        # 正式采样阶段总接受次数/总采样次数
        acceptance_fraction = np.asarray(sampler.acceptance_fraction,dtype=float)
        # \tau_{i} = 1 + 2\sum^{\infty}_{k=1} \rho_{i}(k) i:参数编号 rho_{i}(k)=Cov(X_{i}^{t},X_{i}^{t+k})/Var(X_{i})
        autocorrelation_time = np.asarray(sampler.get_autocorr_time(tol=0),dtype=float)
        valid_autocorrelation = np.isfinite(autocorrelation_time) & (autocorrelation_time > 0.0)
        effective_sample_size = np.full(autocorrelation_time.shape,np.nan,dtype=float)
        # 有效样本数指N个总样本中大约能提供相当于N_{eff}个独立样本的信息 N_{eff}= N / \tau
        effective_sample_size[valid_autocorrelation] = self.n_walkers*self.production_steps / autocorrelation_time[valid_autocorrelation]
        # 100倍判据来自emcee的示例 https://emcee.readthedocs.io/en/latest/user/autocorr/
        chain_long_enough = valid_autocorrelation & (self.production_steps >= 100.0*autocorrelation_time)

        return SamplerResult(
            parameter_names=parameter_names,
            samples=samples,
            log_probability=log_probability,
            chain=chain,
            log_probability_chain=log_probability_chain,
            steps=steps,
            acceptance_fraction=acceptance_fraction,
            autocorrelation_time=autocorrelation_time,
            effective_sample_size=effective_sample_size,
            chain_long_enough=chain_long_enough,
            converged=bool(np.all(chain_long_enough)),
            metadata={
                "sampler":"emcee",
                "n_walkers":self.n_walkers,
                "move_weights":{
                    "stretch":self.move_weights.stretch,
                    "de":self.move_weights.de,
                    "de_snooker":self.move_weights.de_snooker,
                },
                "n_processes":self.n_processes,
                "burn_in":self.burn_in,
                "production_steps":self.production_steps,
                "thin":self.thin,
                "convergence_criterion":"production_steps >= 100*tau",
            },
        )

    def _log_probability(
        self,
        theta: dict[str,float],
        posterior: PosteriorCalculator,
    ) -> float:
        """单进程模式下计算对数后验"""

        procedure_file_label = f"{self.study_name}_{uuid4().hex}"

        return posterior(theta,procedure_file_label)
