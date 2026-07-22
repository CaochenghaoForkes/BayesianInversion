import numpy as np
from scipy.stats import truncnorm

from config import InputCard

class Prior:
    """先验"""

    class SingleParameterPrior:
        """单参数的先验"""

        def __init__(
            self
        ):
            self.prior_dict = {
                'uniform': self._uniform,
                'normal': self._normal
            }            

        def log_single_parameter_prior(self,prior_type_i:str,theta_i:float,mean_i:float,sigma_i:float,lower_i:float,upper_i:float) -> float:
            """计算第i个变量的 log 先验"""

            if theta_i < lower_i or theta_i > upper_i:
                return -np.inf

            if prior_type_i not in self.prior_dict:
                raise ValueError(f"未知先验类型: {prior_type_i}")

            prior_calculator = self.prior_dict[prior_type_i]
            return prior_calculator(theta_i,mean_i,sigma_i,lower_i,upper_i)

        @staticmethod
        def _uniform(theta_i:float,mean_i:float,sigma_i:float,lower_i:float,upper_i:float) -> float:
            """均匀分布"""

            log_prior = -np.log(upper_i - lower_i)

            return log_prior

        @staticmethod
        def _normal(theta_i:float,mean_i:float,sigma_i:float,lower_i:float,upper_i:float) -> float:
            """截断正态分布"""

            lower_standard = (lower_i - mean_i) / sigma_i
            upper_standard = (upper_i - mean_i) / sigma_i
            log_prior = truncnorm.logpdf(theta_i,lower_standard,upper_standard,loc=mean_i,scale=sigma_i)

            return log_prior

    class JointPrior:
        """联合先验"""

        def __init__(
            self
        ):
            self.prior_dict = {
                'independent': self._independent
            }

        def log_joint_prior(self,log_prior_list,joint_prior_type):
            """计算联合先验"""

            prior_calculator = self.prior_dict[joint_prior_type]

            return prior_calculator(log_prior_list)

        @staticmethod
        def _independent(log_prior_list:list[float]) -> float:
            """参数间相互独立计算联合先验"""

            log_prior = sum(log_prior_list)

            return log_prior

    def __init__(
        self,
        prior_config: InputCard.Prior
    ):
        # 读入配置
        self.prior_config = prior_config
        self.single_parameter_prior_config = self.prior_config.single_parameter_prior        
        self.joint_prior_config = self.prior_config.joint_prior
        # 初始化变量
        self.log_prior_dict = {name:None for name in self.single_parameter_prior_config}
        # 初始化子类对象
        self.single_parameter_prior_calculator = self.SingleParameterPrior()
        self.joint_prior_calculator = self.JointPrior()

    def log_prior(self,theta:dict[str,float]) -> float:
        """计算先验"""

        log_prior_list = []

        for name_i,prior_config in self.single_parameter_prior_config.items():
            theta_i = theta[name_i]
            lower_i = prior_config.lower
            upper_i = prior_config.upper
            prior_type_i = prior_config.type
            mean_i = prior_config.normal.mean
            sigma_i = prior_config.normal.sigma

            log_prior_i = self.single_parameter_prior_calculator.log_single_parameter_prior(prior_type_i,theta_i,mean_i,sigma_i,lower_i,upper_i)
            log_prior_list.append(log_prior_i)
            self.log_prior_dict[name_i] = log_prior_i

        joint_prior_type = self.joint_prior_config.type
        joint_prior = self.joint_prior_calculator.log_joint_prior(log_prior_list,joint_prior_type)

        return joint_prior
