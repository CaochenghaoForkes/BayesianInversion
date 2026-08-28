"""Load, resolve defaults for, and validate Bayesian inversion configurations."""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class InputCard:
    """输入配置"""

    @dataclass
    class Workflow:
        """程序运行方式"""

        mode: str = "ask"  # [ask, run, postprocess]

    @dataclass
    class Study:
        """研究任务元数据"""

        name: str = "test"
        description: str = ""

    @dataclass
    class Parameter:
        """反演参数配置"""

        name: str
        initial: float

    @dataclass
    class Prior:
        """先验配置"""

        @dataclass
        class JointPrior:
            """联合先验配置"""

            type: str = "independent"  # [independent]

        @dataclass
        class SingleParameterPrior:
            """全部反演参数单参数先验配置"""

            @dataclass
            class Truncnorm:
                """反演参数截断正态先验配置"""

                mean: float | None = None
                sigma: float | None = None

            lower: float
            upper: float
            type: str = "uniform"  # [uniform, truncnorm]
            truncnorm: Truncnorm = field(default_factory=Truncnorm)

        joint_prior: JointPrior = field(default_factory=JointPrior)
        single_parameter_prior: dict[str,SingleParameterPrior] = field(default_factory=dict)

    @dataclass
    class Data:
        """观测数据配置"""

        @dataclass
        class Source:
            """观测数据来源"""

            path: str | None
            component_order: list[str]
            type: str = "csv"  # [csv]

        @dataclass
        class Dataset:
            """单个命名观测数据配置"""

            name: str
            source: "InputCard.Data.Source"

        datasets: list[Dataset] = field(default_factory=list)

    @dataclass
    class Likelihood:
        """似然配置"""

        @dataclass
        class JointLikelihood:
            """联合似然配置"""

            @dataclass
            class Distribution:
                """联合分布配置"""

                @dataclass
                class StudentT:
                    """多元 Student-t 分布配置"""

                    degrees_of_freedom: float = 5.0

                type: str = "normal"  # [normal, student_t]
                student_t: StudentT = field(default_factory=StudentT)

            @dataclass
            class Correlation:
                """联合相关性配置"""

                @dataclass
                class CompleteMatrix:
                    """完整联合相关矩阵配置"""

                    source: str | None = None

                @dataclass
                class CrossCorrelation:
                    """两组数据间的交叉相关性配置"""

                    @dataclass
                    class CompleteMatrix:
                        """完整交叉相关矩阵配置"""

                        source: str | None = None

                    @dataclass
                    class Exponential:
                        """指数交叉相关模型配置"""

                        length_scales: list[float] = field(
                            default_factory=list
                        )
                        correlation_value: float | None = None

                    data_i: str
                    data_j: str
                    type: str = "independent"  # [independent, complete_matrix, exponential]
                    complete_matrix: CompleteMatrix = field(
                        default_factory=CompleteMatrix
                    )
                    exponential: Exponential = field(
                        default_factory=Exponential
                    )

                type: str = "block_matrix"  # [complete_matrix, block_matrix]
                complete_matrix: CompleteMatrix = field(
                    default_factory=CompleteMatrix
                )
                block_matrix: dict[str, CrossCorrelation] = field(
                    default_factory=dict
                )

            data_order: list[str] = field(default_factory=list)
            distribution: Distribution = field(default_factory=Distribution)
            correlation: Correlation = field(default_factory=Correlation)

        @dataclass
        class SingleDataLikelihood:
            """单观测序列的似然配置"""

            @dataclass
            class PointCorrelation:
                """单组数据观测点间的相关性配置"""

                @dataclass
                class CompleteMatrix:
                    """完整相关矩阵配置"""

                    source: str | None = None

                @dataclass
                class Exponential:
                    """指数相关模型配置"""

                    length_scales: dict[str,float] = field(
                        default_factory=dict
                    )

                type: str = "independent"  # [independent, complete_matrix, exponential]
                complete_matrix: CompleteMatrix = field(default_factory=CompleteMatrix)
                exponential: Exponential = field(
                    default_factory=Exponential
                )

            @dataclass
            class Sigma:
                """模型误差配置"""

                @dataclass
                class Relative:
                    """物理线性空间中的相对误差配置"""

                    ratio: float | None = 0.1

                @dataclass
                class Absolute:
                    """物理线性空间中的绝对误差配置"""

                    value: float | None = None

                type: str = "relative"  # [relative, absolute]
                absolute: Absolute = field(default_factory=Absolute)
                relative: Relative = field(default_factory=Relative)

            dataset: str
            value_component: str
            coordinate_components: list[str] = field(default_factory=list)
            space: str = "linear"  # [linear, log10]
            point_correlation: PointCorrelation = field(
                default_factory=PointCorrelation
            )
            sigma_exp: Sigma = field(default_factory=Sigma)
            sigma_model: Sigma = field(default_factory=Sigma)

        joint_likelihood: JointLikelihood = field(default_factory=JointLikelihood)
        single_data_likelihood: list[SingleDataLikelihood] = field(default_factory=list)

    @dataclass 
    class Model:
        """正演模型配置"""

        @dataclass
        class Mars:
            """一维扩散释放程序MARS配置"""

            @dataclass
            class Dataset:
                """一个MARS算例产生的命名数据集"""

                component_map: dict[str,str] = field(default_factory=dict)

            @dataclass
            class Case:
                """一个MARS XML算例及其输出数据集"""

                datasets: dict[str,"InputCard.Model.Mars.Dataset"] = field(
                    default_factory=dict
                )

            work_dir: str | None = None
            case_map: dict[str,Case] = field(default_factory=dict)

        type: str = "mars"  # [mars]
        mars: Mars = field(default_factory=Mars)

    @dataclass
    class Sampler:
        """采样器配置"""

        @dataclass
        class Emcee:
            """emcee采样器专用配置"""

            @dataclass
            class MoveWeights:
                """emcee各提议方法的相对选择权重"""

                stretch: float
                de: float
                de_snooker: float

            n_walkers: int
            move_weights: MoveWeights

        n_processes: int
        production_steps: int
        burn_in: int
        thin: int
        initial_spread: dict[str,float]
        type: str = "emcee"  # [emcee]
        emcee: Emcee | None = None

    @dataclass
    class Output:
        """输出配置"""

        directory: str = "results/test"

    @dataclass
    class Postprocess:
        """后处理配置"""

        separate_figures: bool = False

    schema_version: str  # [1.0]
    study: Study
    random_seed: int
    parameters: list[Parameter]
    prior: Prior
    data: Data
    likelihood: Likelihood
    model: Model
    sampler: Sampler
    output: Output
    workflow: Workflow = field(default_factory=Workflow)
    postprocess: Postprocess = field(default_factory=Postprocess)


class Examine:
    """检查已解析的输入配置"""

    @staticmethod
    def examine(input_card: InputCard) -> None:
        """统一检查入口"""

        if input_card.schema_version != "1.0":
            raise ValueError(
                f"不支持的 schema_version: {input_card.schema_version}"
            )
        if not input_card.study.name.strip():
            raise ValueError("study.name 不能为空")
        if input_card.random_seed < 0:
            raise ValueError("random_seed 不能为负数")

        Examine._workflow(input_card.workflow)
        Examine._parameters(input_card)
        Examine._prior(input_card)
        Examine._data_and_likelihood(input_card)
        Examine._model(input_card)
        Examine._sampler(input_card)
        Examine._postprocess(input_card)

    @staticmethod
    def _workflow(config: InputCard.Workflow) -> None:
        """检查程序运行方式"""

        if config.mode not in {"ask","run","postprocess"}:
            raise ValueError(
                "workflow.mode 只支持 ask、run 或 postprocess"
            )

    @staticmethod
    def _parameters(input_card: InputCard) -> None:
        """检查参数名称及其对应配置"""

        names = [parameter.name for parameter in input_card.parameters]
        if not names:
            raise ValueError("至少需要一个反演参数")
        if len(names) != len(set(names)):
            raise ValueError("反演参数名称不能重复")

        expected = set(names)
        if set(input_card.prior.single_parameter_prior) != expected:
            raise ValueError(
                "prior.single_parameter_prior 与 parameters 的参数名称不一致"
            )
        if set(input_card.sampler.initial_spread) != expected:
            raise ValueError(
                "sampler.initial_spread 与 parameters 的参数名称不一致"
            )

    @staticmethod
    def _prior(input_card: InputCard) -> None:
        """检查当前启用的先验配置"""

        if input_card.prior.joint_prior.type != "independent":
            raise NotImplementedError("当前仅实现 independent 联合先验")

        parameters = {
            parameter.name: parameter
            for parameter in input_card.parameters
        }
        for name,prior in input_card.prior.single_parameter_prior.items():
            if prior.type not in {"uniform", "truncnorm"}:
                raise NotImplementedError(f"未实现的先验类型: {prior.type}")
            if prior.lower >= prior.upper:
                raise ValueError(f"参数 {name} 的 lower 必须小于 upper")
            if not prior.lower <= parameters[name].initial <= prior.upper:
                raise ValueError(f"参数 {name} 的初始值不在先验范围内")
            if prior.type != "truncnorm":
                continue
            if prior.truncnorm.mean is None:
                raise ValueError(f"参数 {name} 的 truncnorm.mean 不能为 null")
            if prior.truncnorm.sigma is None or prior.truncnorm.sigma <= 0.0:
                raise ValueError(f"参数 {name} 的 truncnorm.sigma 必须大于 0")

    @staticmethod
    def _data_and_likelihood(input_card: InputCard) -> None:
        """检查观测数据与似然名称的对应关系"""

        data = input_card.data
        data_names = [dataset.name for dataset in data.datasets]
        if not data_names or len(data_names) != len(set(data_names)):
            raise ValueError("data.datasets 的名称必须非空且不能重复")
        for dataset in data.datasets:
            source = dataset.source
            if source.type != "csv":
                raise NotImplementedError(f"未实现的数据来源: {source.type}")
            if source.path is None or not Path(source.path).is_file():
                raise FileNotFoundError(
                    f"观测数据 {dataset.name} 的文件不存在: {source.path}"
                )
            components = source.component_order
            if not components:
                raise ValueError(
                    f"观测数据 {dataset.name} 的 component_order 不能为空"
                )
            if len(components) != len(set(components)):
                raise ValueError(
                    f"观测数据 {dataset.name} 的 component_order 不能重复"
                )

        likelihood = input_card.likelihood
        order = likelihood.joint_likelihood.data_order
        configs = likelihood.single_data_likelihood
        config_names = [config.dataset for config in configs]
        if len(config_names) != len(set(config_names)):
            raise ValueError("single_data_likelihood 中的 dataset 不能重复")
        if set(order) != set(config_names):
            raise ValueError(
                "joint_likelihood.data_order 与 single_data_likelihood 名称不一致"
            )

        for config in configs:
            if config.dataset not in data_names:
                raise ValueError(
                    f"观测数据 {config.dataset} 不存在于 data.datasets"
                )

            dataset = next(
                item
                for item in data.datasets
                if item.name == config.dataset
            )
            components = set(dataset.source.component_order)
            if config.value_component not in components:
                raise ValueError(
                    f"观测数据 {config.dataset} 不包含数值分量 "
                    f"{config.value_component}"
                )
            if len(config.coordinate_components) != len(
                set(config.coordinate_components)
            ):
                raise ValueError(
                    f"观测数据 {config.dataset} 的坐标分量不能重复"
                )
            missing_coordinates = (
                set(config.coordinate_components)
                - components
            )
            if missing_coordinates:
                raise ValueError(
                    f"观测数据 {config.dataset} 缺少坐标分量 "
                    f"{sorted(missing_coordinates)}"
                )
            Examine._single_likelihood(config)

        joint = likelihood.joint_likelihood
        distribution_type = joint.distribution.type
        if distribution_type not in {"normal", "student_t"}:
            raise NotImplementedError(
                f"当前未实现联合分布: {distribution_type}"
            )
        if distribution_type == "student_t":
            degrees_of_freedom = (
                joint.distribution.student_t.degrees_of_freedom
            )
            if (
                not isinstance(degrees_of_freedom,(int,float))
                or not math.isfinite(degrees_of_freedom)
                or degrees_of_freedom <= 2.0
            ):
                raise ValueError(
                    "student_t.degrees_of_freedom 必须大于 2，"
                    "以保证 sigma 表示真实标准差"
                )
        if joint.correlation.type == "complete_matrix":
            Examine._required_file(
                joint.correlation.complete_matrix.source,
                "joint_likelihood.correlation.complete_matrix.source",
            )
        elif joint.correlation.type == "block_matrix":
            Examine._cross_correlations(
                joint.correlation.block_matrix,
                order,
                configs,
            )
        else:
            raise NotImplementedError(
                f"未实现的联合相关结构: {joint.correlation.type}"
            )

    @staticmethod
    def examine_observed_data(
        input_card: InputCard,
        values: dict[str,Any],
    ) -> None:
        """检查已加载观测数据的坐标对应关系"""

        for config in input_card.likelihood.single_data_likelihood:
            if not config.coordinate_components:
                continue

            dataset_values = values[config.dataset]
            if dataset_values.ndim != 2:
                raise ValueError(
                    f"观测数据 {config.dataset} 必须为二维的点×分量数组"
                )

    @staticmethod
    def _single_likelihood(
        config: InputCard.Likelihood.SingleDataLikelihood,
    ) -> None:
        """检查单组数据当前启用的似然配置"""

        if config.space not in {"linear", "log10"}:
            raise NotImplementedError(f"未实现的似然空间: {config.space}")

        point = config.point_correlation
        if point.type == "complete_matrix":
            Examine._required_file(
                point.complete_matrix.source,
                f"{config.dataset}.point_correlation.complete_matrix.source",
            )
        elif point.type == "exponential":
            if not config.coordinate_components:
                raise ValueError(
                    f"数据 {config.dataset} 的指数相关需要 coordinate_components"
                )
            length_scales = point.exponential.length_scales
            if set(length_scales) != set(config.coordinate_components):
                raise ValueError(
                    f"数据 {config.dataset} 的 length_scales 必须与坐标分量一致"
                )
            if any(value <= 0.0 for value in length_scales.values()):
                raise ValueError(
                    f"数据 {config.dataset} 的 length_scales 必须全部大于 0"
                )
        elif point.type != "independent":
            raise NotImplementedError(f"未实现的点相关类型: {point.type}")

        for label,sigma in (
            ("sigma_exp", config.sigma_exp),
            ("sigma_model", config.sigma_model),
        ):
            if sigma.type == "absolute":
                if sigma.absolute.value is None or sigma.absolute.value <= 0.0:
                    raise ValueError(f"{config.dataset}.{label}.absolute.value 必须大于 0")
            elif sigma.type == "relative":
                if sigma.relative.ratio is None or sigma.relative.ratio <= 0.0:
                    raise ValueError(f"{config.dataset}.{label}.relative.ratio 必须大于 0")
            else:
                raise NotImplementedError(f"未实现的误差类型: {sigma.type}")

    @staticmethod
    def _cross_correlations(
        block_matrix: dict[str,InputCard.Likelihood.JointLikelihood.Correlation.CrossCorrelation],
        data_order: list[str],
        single_data_configs: list[InputCard.Likelihood.SingleDataLikelihood],
    ) -> None:
        """检查当前启用的交叉相关配置"""

        data_names = set(data_order)
        config_map = {
            config.dataset: config
            for config in single_data_configs
        }
        pairs = set()
        for config in block_matrix.values():
            if config.data_i not in data_names or config.data_j not in data_names:
                raise ValueError("交叉相关配置引用了 data_order 之外的数据")
            if config.data_i == config.data_j:
                raise ValueError("交叉相关的 data_i 和 data_j 不能相同")
            pair = frozenset((config.data_i,config.data_j))
            if pair in pairs:
                raise ValueError("同一数据对的交叉相关不能重复配置")
            pairs.add(pair)

            if config.type == "complete_matrix":
                Examine._required_file(
                    config.complete_matrix.source,
                    f"{config.data_i}_{config.data_j}.complete_matrix.source",
                )
            elif config.type == "exponential":
                correlation_value = config.exponential.correlation_value
                if correlation_value is None:
                    raise ValueError("指数交叉相关必须提供 correlation_value")
                if not -1.0 <= correlation_value <= 1.0:
                    raise ValueError("correlation_value 必须位于 [-1,1]")

                coordinates_i = config_map[config.data_i].coordinate_components
                coordinates_j = config_map[config.data_j].coordinate_components
                if not coordinates_i or not coordinates_j:
                    raise ValueError("指数交叉相关需要两组数据都配置坐标分量")
                if len(coordinates_i) != len(coordinates_j):
                    raise ValueError(
                        f"数据 {config.data_i} 与 {config.data_j} "
                        "的坐标维数不一致"
                    )

                length_scales = config.exponential.length_scales
                if len(length_scales) != len(coordinates_i):
                    raise ValueError(
                        f"数据 {config.data_i} 与 {config.data_j} "
                        "的 length_scales 数量必须等于坐标维数"
                    )
                if any(value <= 0.0 for value in length_scales):
                    raise ValueError("指数交叉相关的 length_scales 必须全部大于 0")
            elif config.type != "independent":
                raise NotImplementedError(f"未实现的交叉相关类型: {config.type}")

        expected_pair_count = len(data_order)*(len(data_order)-1)//2
        if len(pairs) != expected_pair_count:
            raise ValueError("使用 block_matrix 时必须配置所有数据对的交叉相关")

    @staticmethod
    def _model(input_card: InputCard) -> None:
        """检查当前启用的正演模型"""

        if input_card.model.type != "mars":
            raise NotImplementedError(f"未实现的模型类型: {input_card.model.type}")
        mars = input_card.model.mars
        if mars.work_dir is None:
            raise ValueError("model.mars.work_dir 不能为 null")
        if not mars.case_map:
            raise ValueError("model.mars.case_map 不能为空")

        work_dir = Path(mars.work_dir)
        source_term_dir = (
            work_dir
            if "SourceTermAnalysis" in work_dir.parts
            else work_dir / "SourceTermAnalysis"
        )
        if not source_term_dir.is_dir():
            raise FileNotFoundError(
                f"MARS 工作目录不存在: {source_term_dir}"
            )
        for case_path in mars.case_map:
            resolved_case = Path(case_path)
            if not resolved_case.is_absolute():
                resolved_case = source_term_dir / resolved_case
            if not resolved_case.is_file():
                raise FileNotFoundError(f"MARS 输入卡不存在: {resolved_case}")

        mapped_data = set()
        observed_components = {
            dataset.name: set(dataset.source.component_order)
            for dataset in input_card.data.datasets
        }
        available_raw_outputs = {"time", "release_rate"}

        for case in mars.case_map.values():
            if not case.datasets:
                raise ValueError("MARS case_map 中的 datasets 不能为空")

            for dataset_name,dataset in case.datasets.items():
                if dataset_name not in observed_components:
                    raise ValueError(
                        f"MARS映射引用了不存在的观测数据集: {dataset_name}"
                    )
                if not dataset.component_map:
                    raise ValueError(
                        f"MARS数据集 {dataset_name} 的 component_map 不能为空"
                    )
                mapped_components = set(dataset.component_map)
                if mapped_components != observed_components[dataset_name]:
                    raise ValueError(
                        f"MARS数据集 {dataset_name} 的分量映射与观测数据不一致"
                    )
                unknown_outputs = (
                    set(dataset.component_map.values())
                    - available_raw_outputs
                )
                if unknown_outputs:
                    raise ValueError(
                        "MARS component_map 包含未知原始输出: "
                        f"{sorted(unknown_outputs)}"
                    )
                mapped_data.add(dataset_name)

        if len(mapped_data) != sum(
            len(case.datasets)
            for case in mars.case_map.values()
        ):
            raise ValueError("同一个MARS预测数据集不能由多个算例重复生成")

        required_data = {
            config.dataset
            for config in input_card.likelihood.single_data_likelihood
        }
        missing = required_data - mapped_data
        if missing:
            raise ValueError(f"model.mars.case_map 缺少预测数据映射: {sorted(missing)}")

    @staticmethod
    def _sampler(input_card: InputCard) -> None:
        """检查当前启用的采样器"""

        sampler = input_card.sampler
        if sampler.type != "emcee":
            raise NotImplementedError(f"未实现的采样器: {sampler.type}")
        if sampler.emcee is None:
            raise ValueError("sampler.emcee 不能为空")
        if sampler.emcee.n_walkers < 2*len(input_card.parameters):
            raise ValueError("emcee n_walkers 至少应为参数数量的 2 倍")

        move_weights = sampler.emcee.move_weights
        weights = (
            move_weights.stretch,
            move_weights.de,
            move_weights.de_snooker,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("emcee move_weights 必须为非负有限数值")
        if sum(weights) <= 0.0:
            raise ValueError("emcee move_weights 至少应有一个正权重")
        if sampler.n_processes < 1:
            raise ValueError("n_processes 必须大于或等于 1")
        if sampler.burn_in < 0:
            raise ValueError("burn_in 必须大于或等于 0")
        if sampler.production_steps < 1:
            raise ValueError("production_steps 必须大于 0")
        if sampler.thin < 1:
            raise ValueError("thin 必须大于或等于 1")
        if sampler.thin > sampler.production_steps:
            raise ValueError("thin 不能大于 production_steps")
        if any(
            not math.isfinite(spread) or spread <= 0.0
            for spread in sampler.initial_spread.values()
        ):
            raise ValueError("initial_spread 中的所有数值必须大于 0")

    @staticmethod
    def _postprocess(input_card: InputCard) -> None:
        """检查后处理配置"""

        if not isinstance(input_card.postprocess.separate_figures,bool):
            raise TypeError("postprocess.separate_figures 必须为布尔值")

    @staticmethod
    def _required_file(path: str | None,label: str) -> None:
        """检查当前启用的文件配置"""

        if path is None:
            raise ValueError(f"{label} 不能为 null")
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} 指向的文件不存在: {path}")


class JsonReader:
    """将 JSON 数据转换成输入配置"""

    @classmethod
    def load(cls, path: str | Path) -> InputCard:
        """读取、解析并检查 JSON 输入卡"""

        card_path = Path(path).expanduser().resolve()
        data = json.loads(card_path.read_text(encoding="utf-8"))
        base_directory = card_path.parent

        study_data = data.get("study", {})
        if "case_name" in data and "name" not in study_data:
            study_data["name"] = data["case_name"]
        study = InputCard.Study(**study_data)

        random_seed = data.get("random_seed")
        if random_seed is None:
            random_seed = secrets.randbits(32)

        input_card = InputCard(
            schema_version=data.get("schema_version", "1.0"),
            workflow=InputCard.Workflow(**data.get("workflow", {})),
            study=study,
            random_seed=random_seed,
            parameters=cls.read_parameters(data.get("parameters", [])),
            prior=cls.read_prior(data.get("prior", {})),
            data=cls.read_data(data.get("data", {}),base_directory),
            likelihood=cls.read_likelihood(
                data.get("likelihood", {}),base_directory
            ),
            model=cls.read_model(data.get("model", {}),base_directory),
            sampler=cls.read_sampler(data.get("sampler", {})),
            output=cls.read_output(
                data.get("output", {}),base_directory,study.name
            ),
            postprocess=cls.read_postprocess(
                data.get("postprocess", {})
            ),
        )
        Examine.examine(input_card)

        return input_card

    @staticmethod
    def read_parameters(data: list[dict[str, Any]]) -> list[InputCard.Parameter]:
        """读取反演参数配置"""

        return [InputCard.Parameter(**item) for item in data]

    @staticmethod
    def read_prior(data: dict[str, Any]) -> InputCard.Prior:
        """读取先验配置"""

        joint_prior = InputCard.Prior.JointPrior(
            **data.get("joint_prior", {})
        )
        single_parameter_prior = {}
        for name,item in data.get("single_parameter_prior", {}).items():
            prior_type = item.get("type", "uniform")
            if prior_type == "normal":
                prior_type = "truncnorm"
            truncnorm_data = item.get("truncnorm", item.get("normal", {}))
            truncnorm = InputCard.Prior.SingleParameterPrior.Truncnorm(
                **truncnorm_data
            )
            single_parameter_prior[name] = InputCard.Prior.SingleParameterPrior(
                lower=item["lower"],
                upper=item["upper"],
                type=prior_type,
                truncnorm=truncnorm,
            )

        return InputCard.Prior(
            joint_prior=joint_prior,
            single_parameter_prior=single_parameter_prior,
        )

    @staticmethod
    def read_data(
        data: dict[str,Any],
        base_directory: Path,
    ) -> InputCard.Data:
        """读取观测数据配置"""

        datasets = []
        for item in data.get("datasets", []):
            source_data = item.get("source", {})
            source = InputCard.Data.Source(
                path=JsonReader.resolve_path(
                    source_data.get("path"),base_directory
                ),
                component_order=source_data.get("component_order", []),
                type=source_data.get("type", "csv"),
            )
            datasets.append(
                InputCard.Data.Dataset(name=item["name"],source=source)
            )

        return InputCard.Data(datasets=datasets)

    @staticmethod
    def read_sigma(data: dict[str, Any]) -> InputCard.Likelihood.SingleDataLikelihood.Sigma:
        """读取误差配置"""

        sigma_type = InputCard.Likelihood.SingleDataLikelihood.Sigma
        return sigma_type(
            type=data.get("type", "relative"),
            absolute=sigma_type.Absolute(**data.get("absolute", {})),
            relative=sigma_type.Relative(**data.get("relative", {})),
        )

    @staticmethod
    def read_joint_likelihood(
        data: dict[str,Any],
        base_directory: Path,
    ) -> InputCard.Likelihood.JointLikelihood:
        """读取联合似然配置"""

        joint_type = InputCard.Likelihood.JointLikelihood

        distribution_data = data.get("distribution", {})
        distribution = joint_type.Distribution(
            type=distribution_data.get("type", "normal"),
            student_t=joint_type.Distribution.StudentT(
                **distribution_data.get("student_t", {})
            ),
        )

        correlation_data = data.get("correlation", {})
        complete_data = correlation_data.get("complete_matrix", {})
        complete_source = complete_data.get(
            "source",correlation_data.get("complete_matrix_source")
        )
        complete_matrix = joint_type.Correlation.CompleteMatrix(
            source=JsonReader.resolve_path(complete_source,base_directory)
        )

        cross_type = joint_type.Correlation.CrossCorrelation
        block_matrix = {}
        for name,item in correlation_data.get("block_matrix", {}).items():
            cross_complete_data = item.get("complete_matrix", {})
            cross_source = cross_complete_data.get(
                "source",item.get("complete_matrix_source")
            )
            exponential_data = item.get("exponential", {}).copy()
            if "correlation_value" not in exponential_data:
                exponential_data["correlation_value"] = item.get(
                    "correlation_value"
                )
            if "length_scales" not in exponential_data:
                legacy_length_scale = exponential_data.pop(
                    "length_scale",None
                )
                if legacy_length_scale is not None:
                    exponential_data["length_scales"] = [
                        legacy_length_scale
                    ]

            block_matrix[name] = cross_type(
                data_i=item["data_i"],
                data_j=item["data_j"],
                type=item.get("type", "independent"),
                complete_matrix=cross_type.CompleteMatrix(
                    source=JsonReader.resolve_path(
                        cross_source,base_directory
                    )
                ),
                exponential=cross_type.Exponential(**exponential_data),
            )

        correlation = joint_type.Correlation(
            type=correlation_data.get("type", "block_matrix"),
            complete_matrix=complete_matrix,
            block_matrix=block_matrix,
        )

        return joint_type(
            data_order=data.get("data_order", []),
            distribution=distribution,
            correlation=correlation,
        )

    @staticmethod
    def read_point_correlation(
        data: dict[str,Any],
        base_directory: Path,
        coordinate_components: list[str],
    ) -> InputCard.Likelihood.SingleDataLikelihood.PointCorrelation:
        """读取单组数据观测点间的相关性配置"""

        complete_data = data.get("complete_matrix", {})
        point_type = InputCard.Likelihood.SingleDataLikelihood.PointCorrelation
        exponential_data = data.get("exponential", {}).copy()
        if "length_scales" not in exponential_data:
            legacy_length_scale = exponential_data.pop(
                "length_scale",None
            )
            if legacy_length_scale is not None:
                exponential_data["length_scales"] = {
                    name: legacy_length_scale
                    for name in coordinate_components
                }

        return point_type(
            type=data.get("type", "independent"),
            complete_matrix=point_type.CompleteMatrix(
                source=JsonReader.resolve_path(
                    complete_data.get("source"),base_directory
                )
            ),
            exponential=point_type.Exponential(
                **exponential_data
            ),
        )

    @staticmethod
    def read_likelihood(
        data: dict[str,Any],
        base_directory: Path,
    ) -> InputCard.Likelihood:
        """读取似然配置"""

        joint_likelihood = JsonReader.read_joint_likelihood(
            data.get("joint_likelihood", {}),base_directory
        )
        single_data_likelihood = []
        for item in data.get("single_data_likelihood", []):
            coordinate_components = item.get("coordinate_components")
            if coordinate_components is None:
                legacy_coordinate = item.get("coordinate")
                coordinate_components = (
                    []
                    if legacy_coordinate is None
                    else [legacy_coordinate]
                )

            single_data_likelihood.append(
                InputCard.Likelihood.SingleDataLikelihood(
                    dataset=item.get("dataset",item.get("name")),
                    value_component=item.get(
                        "value_component",item.get("name")
                    ),
                    coordinate_components=coordinate_components,
                    space=item.get("space", "linear"),
                    point_correlation=JsonReader.read_point_correlation(
                        item.get("point_correlation", {}),
                        base_directory,
                        coordinate_components,
                    ),
                    sigma_exp=JsonReader.read_sigma(item.get("sigma_exp", {})),
                    sigma_model=JsonReader.read_sigma(item.get("sigma_model", {})),
                )
            )

        return InputCard.Likelihood(
            joint_likelihood=joint_likelihood,
            single_data_likelihood=single_data_likelihood,
        )

    @staticmethod
    def read_model(
        data: dict[str,Any],
        base_directory: Path,
    ) -> InputCard.Model:
        """读取正演模型配置"""

        mars_data = data.get("mars", {})
        case_map = {}
        for case_path,case_data in mars_data.get("case_map", {}).items():
            datasets = {
                name: InputCard.Model.Mars.Dataset(
                    component_map=dataset.get("component_map", {})
                )
                for name,dataset in case_data.get("datasets", {}).items()
            }
            case_map[case_path] = InputCard.Model.Mars.Case(
                datasets=datasets
            )

        return InputCard.Model(
            type=data.get("type", "mars"),
            mars=InputCard.Model.Mars(
                work_dir=JsonReader.resolve_path(
                    mars_data.get("work_dir"),base_directory
                ),
                case_map=case_map,
            ),
        )

    @staticmethod
    def read_sampler(
        data: dict[str,Any],
    ) -> InputCard.Sampler:
        """读取采样器配置"""

        burn_in = data.get("burn_in", 0)
        if "production_steps" in data:
            production_steps = data["production_steps"]
        elif "n_steps" in data:
            production_steps = data["n_steps"] - burn_in
        else:
            raise KeyError("sampler.production_steps")

        emcee_data = data.get("emcee", {})
        move_data = emcee_data.get("move_weights", {})
        n_walkers = emcee_data.get("n_walkers", data.get("n_walkers"))
        if n_walkers is None:
            raise KeyError("sampler.emcee.n_walkers")

        return InputCard.Sampler(
            n_processes=data["n_processes"],
            production_steps=production_steps,
            burn_in=burn_in,
            thin=data.get("thin", 1),
            initial_spread=data["initial_spread"],
            type=data.get("type", "emcee"),
            emcee=InputCard.Sampler.Emcee(
                n_walkers=n_walkers,
                move_weights=InputCard.Sampler.Emcee.MoveWeights(
                    stretch=move_data.get("stretch", 1.0),
                    de=move_data.get("de", 0.0),
                    de_snooker=move_data.get("de_snooker", 0.0),
                ),
            ),
        )

    @staticmethod
    def read_output(
        data: dict[str,Any],
        base_directory: Path,
        study_name: str,
    ) -> InputCard.Output:
        """读取输出配置"""

        directory = data.get("directory", data.get("output_dir"))
        if directory is None:
            directory = f"results/{study_name}"

        return InputCard.Output(
            directory=JsonReader.resolve_path(directory,base_directory)
        )

    @staticmethod
    def read_postprocess(
        data: dict[str,Any],
    ) -> InputCard.Postprocess:
        """读取后处理配置"""

        return InputCard.Postprocess(
            separate_figures=data.get("separate_figures",False)
        )

    @staticmethod
    def resolve_path(path: str | None,base_directory: Path) -> str | None:
        """将用户路径相对于 JSON 输入卡所在目录解析"""

        if path is None:
            return None

        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = base_directory / resolved

        return str(resolved.resolve())
