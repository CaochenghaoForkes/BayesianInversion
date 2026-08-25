from functools import partial

import numpy as np
from scipy.stats import truncnorm

from config import InputCard

class Prior:
    """先验"""

    class SingleParameterPrior:
        """单参数的先验"""

        def __init__(
            self,
            prior_config: dict[str,InputCard.Prior.SingleParameterPrior],
        ):
            self.prior_builder_dict = {
                "uniform":self._build_uniform,
                "truncnorm":self._build_truncnorm,
            }
            self.parameter_prior_dict = {name:self._build_prior(config) for name,config in prior_config.items()}

        def _build_prior(
            self,
            config: InputCard.Prior.SingleParameterPrior,
        ):
            """根据配置构造单参数先验计算函数"""

            if config.type not in self.prior_builder_dict:
                raise NotImplementedError(f"未实现的先验类型: {config.type}")

            builder = self.prior_builder_dict[config.type]
            return builder(config)

        def log_single_parameter_prior(
            self,
            name: str,
            theta: float,
        ) -> float:
            """计算一个参数的 log 先验"""

            if not np.isfinite(theta):
                return -np.inf

            calculator = self.parameter_prior_dict[name]
            return float(calculator(theta))

        def _build_uniform(
            self,
            config: InputCard.Prior.SingleParameterPrior,
        ):
            """构造均匀先验计算函数"""

            return partial(
                self._uniform,
                lower=config.lower,
                upper=config.upper,
            )

        def _build_truncnorm(
            self,
            config: InputCard.Prior.SingleParameterPrior,
        ):
            """构造截断正态先验计算函数"""

            mean = config.truncnorm.mean
            sigma = config.truncnorm.sigma
            lower_standard = (config.lower-mean)/sigma
            upper_standard = (config.upper-mean)/sigma

            return partial(
                self._truncnorm,
                lower_standard=lower_standard,
                upper_standard=upper_standard,
                mean=mean,
                sigma=sigma,
            )

        @staticmethod
        def _uniform(
            theta: float,
            *,
            lower: float,
            upper: float,
        ) -> float:
            """均匀分布"""

            if theta < lower or theta > upper:
                return -np.inf

            return -np.log(upper-lower)

        @staticmethod
        def _truncnorm(
            theta: float,
            *,
            lower_standard: float,
            upper_standard: float,
            mean: float,
            sigma: float,
        ) -> float:
            """截断正态分布"""

            return truncnorm.logpdf(
                theta,
                lower_standard,
                upper_standard,
                loc=mean,
                scale=sigma,
            )

    class JointPrior:
        """联合先验"""

        def __init__(
            self,
            joint_prior_config: InputCard.Prior.JointPrior,
            parameter_names: list[str],
            single_parameter_prior: "Prior.SingleParameterPrior",
        ):
            self.joint_prior_config = joint_prior_config
            self.parameter_names = parameter_names
            self.single_parameter_prior = single_parameter_prior
            self.prior_dict = {
                "independent":self._independent,
            }

        def log_joint_prior(
            self,
            theta: dict[str,float],
        ) -> float:
            """计算联合先验"""

            prior_type = self.joint_prior_config.type
            if prior_type not in self.prior_dict:
                raise NotImplementedError(f"未实现的联合先验: {prior_type}")

            calculator = self.prior_dict[prior_type]
            return calculator(theta)

        def _independent(
            self,
            theta: dict[str,float],
        ) -> float:
            """参数间相互独立计算联合先验"""

            log_prior = 0.0
            for name in self.parameter_names:
                value = self.single_parameter_prior.log_single_parameter_prior(name,theta[name])
                if not np.isfinite(value):
                    return -np.inf

                log_prior += value

            return log_prior

    def __init__(
        self,
        prior_config: InputCard.Prior,
        parameter_names: list[str],
    ):
        # 读入配置
        self.prior_config = prior_config
        self.single_parameter_prior_config = self.prior_config.single_parameter_prior        
        self.joint_prior_config = self.prior_config.joint_prior
        # 初始化子类对象
        self.single_parameter_prior_calculator = self.SingleParameterPrior(
            self.single_parameter_prior_config
        )
        self.joint_prior_calculator = self.JointPrior(
            self.joint_prior_config,
            parameter_names,
            self.single_parameter_prior_calculator,
        )

    def log_prior(self,theta:dict[str,float]) -> float:
        """计算先验"""

        return self.joint_prior_calculator.log_joint_prior(theta)
