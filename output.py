import csv
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


class Output:
    """输出"""

    def __init__(self,output_dir:str | Path):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_DIR / self.output_dir
        self.output_dir.mkdir(parents=True,exist_ok=True)

    def save(self,parameter_names:list[str],sampler_result:dict[str,np.ndarray]):
        """保存采样结果与采样诊断信息"""

        self._save_samples(parameter_names,sampler_result["samples"],sampler_result["log_probability"])
        self._save_chain(parameter_names,sampler_result["chain"],sampler_result["log_probability_chain"],sampler_result["step"])
        self._save_acceptance_fraction(sampler_result["acceptance_fraction"])
        self._save_autocorrelation_time(parameter_names,sampler_result["autocorrelation_time"])

    def _save_samples(self,parameter_names:list[str],samples:np.ndarray,log_probability:np.ndarray):
        """保存展平后的后验样本"""

        output_data = np.column_stack((samples,log_probability))
        header = ",".join(parameter_names + ["log_probability"])
        np.savetxt(self.output_dir / "samples.csv",output_data,delimiter=",",header=header,comments="")

    def _save_chain(self,parameter_names:list[str],chain:np.ndarray,log_probability_chain:np.ndarray,step:np.ndarray):
        """保存各步、各walker的未展平采样链"""

        n_saved_steps,n_walkers,_ = chain.shape
        step_column = np.repeat(step,n_walkers)
        walker_column = np.tile(np.arange(n_walkers),n_saved_steps)
        output_data = np.column_stack((step_column,walker_column,chain.reshape(-1,chain.shape[-1]),log_probability_chain.reshape(-1)))
        header = ",".join(["step","walker"] + parameter_names + ["log_probability"])
        fmt = ["%d","%d"] + ["%.18e"] * (len(parameter_names) + 1)
        np.savetxt(self.output_dir / "chain.csv",output_data,delimiter=",",header=header,comments="",fmt=fmt)

    def _save_acceptance_fraction(self,acceptance_fraction:np.ndarray):
        """保存各walker的接受率"""

        output_data = np.column_stack((np.arange(len(acceptance_fraction)),acceptance_fraction))
        np.savetxt(self.output_dir / "acceptance_fraction.csv",output_data,delimiter=",",header="walker,acceptance_fraction",comments="",fmt=["%d","%.18e"])

    def _save_autocorrelation_time(self,parameter_names:list[str],autocorrelation_time:np.ndarray):
        """保存各参数的积分自相关时间"""

        with (self.output_dir / "autocorrelation_time.csv").open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["parameter","autocorrelation_time"])
            writer.writerows(zip(parameter_names,autocorrelation_time))
