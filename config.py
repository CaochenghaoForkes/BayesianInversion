"""读取 BayesianInversion 的 JSON 输入卡"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class InputCard:
    """输入配置"""

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

            type: str = "independent"

        @dataclass
        class SingleParameterPrior:
            """全部反演参数单参数先验配置"""

            @dataclass
            class Normal:
                """反演参数单参数正态先验配置"""

                mean: float | None = None
                sigma: float | None = None

            lower: float
            upper: float
            type: str = "uniform"
            normal: Normal = field(default_factory=Normal)

        joint_prior: JointPrior = field(default_factory=JointPrior)
        single_parameter_prior: dict[str,SingleParameterPrior] = field(default_factory=dict)

    @dataclass
    class Data:
        """观测数据配置"""

        name: list[str]
        value: list[list[float]] | None
        data_source: str = "json"

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

                type: str = "normal"
                student_t: StudentT = field(default_factory=StudentT)

            @dataclass
            class Correlation:
                """联合相关性配置"""

                @dataclass
                class CrossCorrelation:
                    """两组数据间的交叉相关性配置"""

                    @dataclass
                    class Exponential:
                        """指数交叉相关模型配置"""

                        length_scale: float | None = None

                    data_i: str
                    data_j: str
                    type: str = "independent"
                    complete_matrix_source: str | None = None
                    correlation_value: float | None = None
                    exponential: Exponential = field(
                        default_factory=Exponential
                    )

                type: str = "complete_matrix"
                complete_matrix_source: str | None = None
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

                    length_scale: float | None = None

                type: str = "independent"
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

                type: str = "relative"
                absolute: Absolute = field(default_factory=Absolute)
                relative: Relative = field(default_factory=Relative)

            name: str
            coordinate: str | None
            space: str = "linear"
            epsilon: float = 1e-300
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

            work_dir: str | None = None
            case_map: dict[str, dict[str, str]] | None = None

        type: str = "mars"
        mars: Mars = field(default_factory=Mars)

    @dataclass
    class Sampler:
        """采样器配置"""

        sampler_label: str
        type: str
        n_walkers: int
        n_processes: int
        n_steps: int
        burn_in: int
        thin: int
        initial_spread: dict[str,float]

    @dataclass
    class Output:
        """输出配置"""

        output_dir: str

    case_name: str
    random_seed: int
    parameters: list[Parameter]
    prior: Prior
    data: Data
    likelihood: Likelihood
    model: Model
    sampler: Sampler
    output: Output

class JsonReader:
    """将 JSON 数据转换成输入配置"""

    @classmethod
    def load(cls, path: str | Path) -> InputCard:
        """读取 JSON 输入卡，并转换成结构化 InputCard"""

        card_path = Path(path)
        data = json.loads(card_path.read_text(encoding="utf-8"))

        return InputCard(
            case_name=data["case_name"],
            random_seed=data["random_seed"],
            parameters=cls.read_parameters(data["parameters"]),
            prior=cls.read_prior(data["prior"]),
            data=cls.read_data(data["data"]),
            likelihood=cls.read_likelihood(data["likelihood"]),
            model=cls.read_model(data["model"]),
            sampler=cls.read_sampler(data["sampler"]),
            output=cls.read_output(data["output"]),
        )

    @staticmethod
    def read_parameters(data: list[dict[str, Any]]) -> list[InputCard.Parameter]:
        """读取反演参数配置"""

        return [InputCard.Parameter(**item) for item in data]

    @staticmethod
    def read_prior(data: dict[str, Any]) -> InputCard.Prior:
        """读取先验配置"""

        joint_prior = InputCard.Prior.JointPrior(**data["joint_prior"])
        single_parameter_prior = {}
        for name,item in data["single_parameter_prior"].items():
            normal = InputCard.Prior.SingleParameterPrior.Normal(**item["normal"])
            single_parameter_prior[name] = InputCard.Prior.SingleParameterPrior(
                lower=item["lower"],
                upper=item["upper"],
                type=item["type"],
                normal=normal,
            )

        return InputCard.Prior(
            joint_prior=joint_prior,
            single_parameter_prior=single_parameter_prior,
        )

    @staticmethod
    def read_data(data: dict[str, Any]) -> InputCard.Data:
        """读取观测数据配置"""

        return InputCard.Data(**data)

    @staticmethod
    def read_sigma(data: dict[str, Any]) -> InputCard.Likelihood.SingleDataLikelihood.Sigma:
        """读取误差配置"""

        sigma_type = InputCard.Likelihood.SingleDataLikelihood.Sigma
        return sigma_type(
            type=data["type"],
            absolute=sigma_type.Absolute(**data["absolute"]),
            relative=sigma_type.Relative(**data["relative"]),
        )

    @staticmethod
    def read_joint_likelihood(data: dict[str, Any]) -> InputCard.Likelihood.JointLikelihood:
        """读取联合似然配置"""

        joint_type = InputCard.Likelihood.JointLikelihood

        distribution_data = data["distribution"]
        distribution = joint_type.Distribution(
            type=distribution_data["type"],
            student_t=joint_type.Distribution.StudentT(
                **distribution_data["student_t"]
            ),
        )

        correlation_data = data["correlation"]
        cross_type = joint_type.Correlation.CrossCorrelation
        block_matrix = {}
        for name,item in correlation_data["block_matrix"].items():
            block_matrix[name] = cross_type(
                data_i=item["data_i"],
                data_j=item["data_j"],
                type=item["type"],
                complete_matrix_source=item["complete_matrix_source"],
                correlation_value=item["correlation_value"],
                exponential=cross_type.Exponential(
                    **item["exponential"]
                ),
            )

        correlation = joint_type.Correlation(
            type=correlation_data["type"],
            complete_matrix_source=correlation_data["complete_matrix_source"],
            block_matrix=block_matrix,
        )

        return joint_type(
            data_order=data["data_order"],
            distribution=distribution,
            correlation=correlation,
        )

    @staticmethod
    def read_point_correlation(
        data: dict[str, Any]
    ) -> InputCard.Likelihood.SingleDataLikelihood.PointCorrelation:
        """读取单组数据观测点间的相关性配置"""

        point_type = InputCard.Likelihood.SingleDataLikelihood.PointCorrelation
        return point_type(
            type=data["type"],
            complete_matrix=point_type.CompleteMatrix(**data["complete_matrix"]),
            exponential=point_type.Exponential(
                **data["exponential"]
            ),
        )

    @staticmethod
    def read_likelihood(data: dict[str, Any]) -> InputCard.Likelihood:
        """读取似然配置"""

        joint_likelihood = JsonReader.read_joint_likelihood(
            data["joint_likelihood"]
        )
        single_data_likelihood = []
        for item in data["single_data_likelihood"]:
            single_data_likelihood.append(
                InputCard.Likelihood.SingleDataLikelihood(
                    name=item["name"],
                    coordinate=item["coordinate"],
                    space=item["space"],
                    epsilon=item["epsilon"],
                    point_correlation=JsonReader.read_point_correlation(
                        item["point_correlation"]
                    ),
                    sigma_exp=JsonReader.read_sigma(item["sigma_exp"]),
                    sigma_model=JsonReader.read_sigma(item["sigma_model"]),
                )
            )

        return InputCard.Likelihood(
            joint_likelihood=joint_likelihood,
            single_data_likelihood=single_data_likelihood,
        )

    @staticmethod
    def read_model(data: dict[str, Any]) -> InputCard.Model:
        """读取正演模型配置"""

        return InputCard.Model(
            type=data["type"],
            mars=InputCard.Model.Mars(**data["mars"]),
        )

    @staticmethod
    def read_sampler(data: dict[str, Any]) -> InputCard.Sampler:
        """读取采样器配置"""

        return InputCard.Sampler(**data)

    @staticmethod
    def read_output(data: dict[str, Any]) -> InputCard.Output:
        """读取输出配置"""

        return InputCard.Output(**data)
