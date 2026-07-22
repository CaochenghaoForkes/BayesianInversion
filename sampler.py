import numpy as np
from multiprocessing import Pool
from uuid import uuid4

from config import InputCard

class Sampler:
    """采样器"""

    def __init__(self,sampler_config:InputCard.Sampler):
        # 读入配置
        self.sampler_config = sampler_config
        self.sampler_label = self.sampler_config.sampler_label
        self.sampler_type = self.sampler_config.type
        self.n_walkers = self.sampler_config.n_walkers
        self.n_processes = self.sampler_config.n_processes
        self.n_steps = self.sampler_config.n_steps
        self.burn_in = self.sampler_config.burn_in
        self.thin = self.sampler_config.thin
        self.initial_spread = self.sampler_config.initial_spread
        # 初始化
        self.sampler_dict = {
            "emcee":self._emcee,
        }

    def sample(self,parameter_config_list:list[InputCard.Parameter],log_posterior_calculator,random_seed:int) -> dict[str,np.ndarray]:
        """采样"""

        sampler = self.sampler_dict[self.sampler_type]

        return sampler(parameter_config_list,log_posterior_calculator,random_seed)

    def _emcee(self,parameter_config_list:list[InputCard.Parameter],log_posterior_calculator,random_seed:int) -> dict[str,np.ndarray]:
        """emcee集成采样器"""

        import emcee

        np.random.seed(random_seed)
        parameter_names = [config.name for config in parameter_config_list]
        initial = np.asarray([config.initial for config in parameter_config_list],dtype=float)
        initial_spread = np.asarray([self.initial_spread[name] for name in parameter_names],dtype=float)
        n_dimension = len(parameter_names)
        initial_state = initial + initial_spread * np.random.randn(self.n_walkers,n_dimension)

        with Pool(self.n_processes) as pool:
            sampler = emcee.EnsembleSampler(self.n_walkers,n_dimension,self._log_probability,args=[log_posterior_calculator],parameter_names=parameter_names,pool=pool)
            # burn-in只用于让walker进入高概率区域，不进入正式采样统计
            state = sampler.run_mcmc(initial_state,self.burn_in,progress=True,progress_kwargs={"desc":"Burn-in"})
            sampler.reset()
            # reset后chain、接受率和自相关时间都只对应正式采样阶段，n_steps仍表示总步数
            sampler.run_mcmc(state,self.n_steps-self.burn_in,progress=True,progress_kwargs={"desc":"Sampling"})

        chain = sampler.get_chain(thin=self.thin,flat=False)
        log_probability_chain = sampler.get_log_prob(thin=self.thin,flat=False)
        samples = chain.reshape(-1,n_dimension)
        log_probability = log_probability_chain.reshape(-1)
        step = np.arange(self.burn_in + self.thin,self.n_steps + 1,self.thin)
        acceptance_fraction = sampler.acceptance_fraction
        autocorrelation_time = sampler.get_autocorr_time(tol=0)

        return {
            "samples":samples,
            "log_probability":log_probability,
            "chain":chain,
            "log_probability_chain":log_probability_chain,
            "step":step,
            "acceptance_fraction":acceptance_fraction,
            "autocorrelation_time":autocorrelation_time,
        }

    def _log_probability(self,theta:dict[str,float],log_posterior_calculator) -> float:
        """计算对数后验"""

        procedure_file_label = f"{self.sampler_label}_{uuid4().hex}"

        return log_posterior_calculator(theta,procedure_file_label)
