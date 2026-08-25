"""Write Bayesian-inversion samples and diagnostics to result files."""

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from config import InputCard
from sampler import SamplerResult


class Output:
    """将贝叶斯反演结果写入文本与压缩数值文件"""

    MIN_COLUMN_WIDTH = 20
    COLUMN_SEPARATOR = "\t"

    def __init__(self,input_card: InputCard):
        self.study_name = input_card.study.name
        self.parameter_names = [
            parameter.name
            for parameter in input_card.parameters
        ]
        self.seed = input_card.random_seed
        self.output_directory = Path(input_card.output.directory)
        self.file_stem = (
            self.study_name.strip()
            .replace("/","_")
            .replace("\\","_")
        )
        if not self.file_stem:
            self.file_stem = "inversion"

    @staticmethod
    def _format_number(value) -> str:
        """将单个数值格式化为科学计数法"""

        return f"{float(value):.8e}"

    @staticmethod
    def _column_width(*values) -> int:
        """根据标题与内容计算列宽"""

        return max(
            Output.MIN_COLUMN_WIDTH,
            *(len(str(value)) for value in values),
        )

    @staticmethod
    def _format_row(values,column_widths: list[int]) -> str:
        """按固定列宽左对齐并使用制表符分隔"""

        if len(values) != len(column_widths):
            raise ValueError("表格数据列数与列宽数量不一致")

        cells = [
            str(value).ljust(width)
            for value,width in zip(
                values,
                column_widths,
                strict=True,
            )
        ]

        return Output.COLUMN_SEPARATOR.join(cells).rstrip()

    @staticmethod
    def _section(title: str) -> str:
        """生成分节标题"""

        return f"--------{title}----------"

    @staticmethod
    def _write_lines(path: Path,lines: Iterable[str]) -> None:
        """逐行写入文本"""

        with path.open("w",encoding="utf-8") as file:
            for line in lines:
                file.write(line)
                file.write("\n")

    @staticmethod
    def _metadata_value(value: Any) -> str:
        """将元数据转换为可读文本"""

        if isinstance(value,(float,np.floating)):
            return Output._format_number(value)
        if isinstance(value,(bool,np.bool_)):
            return str(bool(value))
        return str(value)

    @staticmethod
    def _npz_value(value: Any) -> np.ndarray:
        """将元数据转换为不含object的NumPy数组"""

        if isinstance(value,(str,bool,int,float,np.generic)):
            return np.asarray(value)
        return np.asarray(str(value))

    def write(self,result: SamplerResult) -> dict[str,Path]:
        """无需输出开关，写入全部采样结果与诊断信息"""

        self._validate_result(result)
        self.output_directory.mkdir(parents=True,exist_ok=True)

        paths = {
            "summary": self._write_summary(result),
            "samples": self._save_samples(
                result.parameter_names,
                result.samples,
                result.log_probability,
            ),
            "chain": self._save_chain(
                result.parameter_names,
                result.chain,
                result.log_probability_chain,
                result.steps,
            ),
            "acceptance_fraction": self._save_acceptance_fraction(
                result.acceptance_fraction
            ),
            "sampling_diagnostics": self._save_sampling_diagnostics(result),
            "sampler_summary": self._save_sampler_summary(result),
            "compressed_result": self._write_compressed_result(result),
        }

        return paths

    def _validate_result(self,result: SamplerResult) -> None:
        """检查采样结果各数组的形状与对应关系"""

        if result.parameter_names != self.parameter_names:
            raise ValueError("采样结果的参数名称或顺序与输入卡不一致")

        dimension = len(self.parameter_names)
        if result.chain.ndim != 3:
            raise ValueError("采样链必须为 step×walker×parameter 三维数组")

        saved_steps,walkers,chain_dimension = result.chain.shape
        if chain_dimension != dimension:
            raise ValueError("采样链的参数维度与参数数量不一致")
        if result.log_probability_chain.shape != (saved_steps,walkers):
            raise ValueError("对数后验链形状与采样链不一致")
        if result.samples.shape != (saved_steps*walkers,dimension):
            raise ValueError("展平后验样本形状与采样链不一致")
        if result.log_probability.shape != (saved_steps*walkers,):
            raise ValueError("展平对数后验形状与后验样本不一致")
        if result.steps.shape != (saved_steps,):
            raise ValueError("采样步编号数量与保存步数不一致")
        if result.acceptance_fraction.shape != (walkers,):
            raise ValueError("接受率数量与walker数量不一致")

        diagnostic_shape = (dimension,)
        diagnostics = {
            "autocorrelation_time": result.autocorrelation_time,
            "effective_sample_size": result.effective_sample_size,
            "chain_long_enough": result.chain_long_enough,
        }
        for name,values in diagnostics.items():
            if values.shape != diagnostic_shape:
                raise ValueError(
                    f"{name}形状应为 {diagnostic_shape}"
                )

        valid_acceptance = (
            np.isfinite(result.acceptance_fraction)
            & (result.acceptance_fraction >= 0.0)
            & (result.acceptance_fraction <= 1.0)
        )
        if not np.all(valid_acceptance):
            raise ValueError("接受率必须为 [0,1] 内的有限数")

        expected_converged = bool(np.all(result.chain_long_enough))
        if result.converged != expected_converged:
            raise ValueError("converged与chain_long_enough的结果不一致")

        metadata_seed = result.metadata.get("random_seed")
        if metadata_seed is not None and metadata_seed != self.seed:
            raise ValueError("采样结果与输入卡的random_seed不一致")

    def _summary_lines(self,result: SamplerResult) -> Iterable[str]:
        """生成研究概要、后验摘要与采样诊断"""

        yield self._section(self.study_name)
        yield ""
        yield "[Sampler]"

        sampler_items = {
            "study_name": self.study_name,
            "random_seed": self.seed,
            "converged": result.converged,
            "saved_steps": result.chain.shape[0],
            "saved_samples": result.samples.shape[0],
            **result.metadata,
        }
        item_widths = [
            self._column_width("item",*sampler_items),
            self._column_width(
                "value",
                *(
                    self._metadata_value(value)
                    for value in sampler_items.values()
                ),
            ),
        ]
        yield self._format_row(["item","value"],item_widths)
        for name,value in sampler_items.items():
            yield self._format_row([
                name,
                self._metadata_value(value),
            ],item_widths)

        yield ""
        yield "[Posterior Summary]"
        posterior_headers = [
            "parameter","mean","standard_deviation",
            "median","quantile_2.5%","quantile_97.5%",
        ]
        posterior_widths = [
            self._column_width("parameter",*self.parameter_names),
            *(self._column_width(header) for header in posterior_headers[1:]),
        ]
        yield self._format_row(posterior_headers,posterior_widths)

        means = np.mean(result.samples,axis=0)
        standard_deviations = np.std(result.samples,axis=0,ddof=1)
        medians = np.median(result.samples,axis=0)
        lower,upper = np.quantile(result.samples,[0.025,0.975],axis=0)
        for index,name in enumerate(self.parameter_names):
            yield self._format_row([
                name,
                self._format_number(means[index]),
                self._format_number(standard_deviations[index]),
                self._format_number(medians[index]),
                self._format_number(lower[index]),
                self._format_number(upper[index]),
            ],posterior_widths)

        yield ""
        yield "[Maximum Log Probability Sample]"
        map_widths = [
            self._column_width(
                "parameter",
                *self.parameter_names,
                "log_probability",
            ),
            self._column_width("value"),
        ]
        yield self._format_row(["parameter","value"],map_widths)

        finite_log_probability = np.isfinite(result.log_probability)
        if np.any(finite_log_probability):
            finite_indices = np.flatnonzero(finite_log_probability)
            local_index = np.argmax(
                result.log_probability[finite_log_probability]
            )
            map_index = finite_indices[local_index]
            map_values = result.samples[map_index]
            map_log_probability = result.log_probability[map_index]
        else:
            map_values = np.full(len(self.parameter_names),np.nan)
            map_log_probability = np.nan

        for name,value in zip(
            self.parameter_names,
            map_values,
            strict=True,
        ):
            yield self._format_row([
                name,
                self._format_number(value),
            ],map_widths)
        yield self._format_row([
            "log_probability",
            self._format_number(map_log_probability),
        ],map_widths)

        yield ""
        yield "[Parameter Diagnostics]"
        diagnostic_headers = [
            "parameter","autocorrelation_time",
            "effective_sample_size","chain_long_enough",
        ]
        diagnostic_widths = [
            self._column_width("parameter",*self.parameter_names),
            *(self._column_width(header) for header in diagnostic_headers[1:]),
        ]
        yield self._format_row(diagnostic_headers,diagnostic_widths)
        for values in zip(
            self.parameter_names,
            result.autocorrelation_time,
            result.effective_sample_size,
            result.chain_long_enough,
            strict=True,
        ):
            name,tau,effective_size,long_enough = values
            yield self._format_row([
                name,
                self._format_number(tau),
                self._format_number(effective_size),
                str(bool(long_enough)),
            ],diagnostic_widths)

        yield ""
        yield "[Walker Acceptance Fraction]"
        acceptance_widths = [
            self._column_width("walker"),
            self._column_width("acceptance_fraction"),
        ]
        yield self._format_row([
            "walker","acceptance_fraction",
        ],acceptance_widths)
        for walker,value in enumerate(result.acceptance_fraction):
            yield self._format_row([
                walker,
                self._format_number(value),
            ],acceptance_widths)

    def _write_summary(self,result: SamplerResult) -> Path:
        """写入贝叶斯反演汇总文件"""

        path = self.output_directory / f"{self.file_stem}_{self.seed}.txt"
        self._write_lines(path,self._summary_lines(result))
        return path

    def _save_samples(
        self,
        parameter_names: list[str],
        samples: np.ndarray,
        log_probability: np.ndarray,
    ) -> Path:
        """保存展平后的后验样本"""

        path = self.output_directory / "samples.csv"
        with path.open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                *parameter_names,
                "log_probability",
            ])
            for values,log_probability_value in zip(
                samples,
                log_probability,
                strict=True,
            ):
                writer.writerow([
                    *(
                        f"{float(value):.18e}"
                        for value in values
                    ),
                    f"{float(log_probability_value):.18e}",
                ])

        return path

    def _save_chain(
        self,
        parameter_names: list[str],
        chain: np.ndarray,
        log_probability_chain: np.ndarray,
        steps: np.ndarray,
    ) -> Path:
        """保存各步、各walker的未展平采样链"""

        path = self.output_directory / "chain.csv"
        with path.open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "step","walker",*parameter_names,"log_probability",
            ])

            for step,step_values,step_log_probability in zip(
                steps,
                chain,
                log_probability_chain,
                strict=True,
            ):
                for walker,(values,log_probability) in enumerate(zip(
                    step_values,
                    step_log_probability,
                    strict=True,
                )):
                    writer.writerow([
                        int(step),
                        walker,
                        *(
                            f"{float(value):.18e}"
                            for value in values
                        ),
                        f"{float(log_probability):.18e}",
                    ])

        return path

    def _save_acceptance_fraction(
        self,
        acceptance_fraction: np.ndarray,
    ) -> Path:
        """保存各walker的接受率"""

        path = self.output_directory / "acceptance_fraction.csv"
        output_data = np.column_stack((
            np.arange(len(acceptance_fraction)),
            acceptance_fraction,
        ))
        np.savetxt(
            path,
            output_data,
            delimiter=",",
            header="walker,acceptance_fraction",
            comments="",
            fmt=["%d","%.18e"],
        )
        return path

    def _save_sampling_diagnostics(
        self,
        result: SamplerResult,
    ) -> Path:
        """保存各参数的采样诊断信息"""

        path = self.output_directory / "sampling_diagnostics.csv"
        with path.open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "parameter",
                "autocorrelation_time",
                "effective_sample_size",
                "chain_long_enough",
            ])
            writer.writerows(zip(
                result.parameter_names,
                result.autocorrelation_time,
                result.effective_sample_size,
                result.chain_long_enough,
                strict=True,
            ))

        return path

    def _save_sampler_summary(
        self,
        result: SamplerResult,
    ) -> Path:
        """保存采样器配置与总体收敛判据"""

        path = self.output_directory / "sampler_summary.csv"
        with path.open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["item","value"])
            writer.writerow(["converged",result.converged])
            writer.writerows(result.metadata.items())

        return path

    def _write_compressed_result(
        self,
        result: SamplerResult,
    ) -> Path:
        """将完整采样结果无损压缩写入NPZ文件"""

        path = (
            self.output_directory
            / f"sampler_result_{self.seed}.npz"
        )
        arrays = {
            "parameter_names": np.asarray(result.parameter_names),
            "samples": result.samples,
            "log_probability": result.log_probability,
            "chain": result.chain,
            "log_probability_chain": result.log_probability_chain,
            "steps": result.steps,
            "acceptance_fraction": result.acceptance_fraction,
            "autocorrelation_time": result.autocorrelation_time,
            "effective_sample_size": result.effective_sample_size,
            "chain_long_enough": result.chain_long_enough,
            "converged": np.asarray(result.converged),
        }
        for name,value in result.metadata.items():
            arrays[f"metadata__{name}"] = self._npz_value(value)

        np.savez_compressed(path,**arrays)
        return path
