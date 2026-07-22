import numpy as np
from scipy.stats import multivariate_normal

from config import InputCard
from data import Data

class Likelihood:
    """似然"""

    class SingleDataLikelihood:
        """单组观测数据进入联合似然前的计算结果"""
        """ -> name residual sigma point_correlation"""

        class Sigma:
            """计算实验与模型在似然空间中的总标准差"""

            def __init__(self):
                # 初始化
                self.sigma_dict = {
                    'absolute':self._absolute_sigma,
                    'relative':self._relative_sigma
                }  
        
            def sigma(self,single_data_likelihood:InputCard.Likelihood.SingleDataLikelihood,y_pred:np.ndarray,y_obs:np.ndarray) -> np.ndarray:
                """计算观测值的实验/模型/总标准差"""

                # 读取配置 
                space = single_data_likelihood.space
                sigma_exp_config = single_data_likelihood.sigma_exp
                sigma_model_config = single_data_likelihood.sigma_model
                epsilon = single_data_likelihood.epsilon
                # 计算单独因素的标准差
                sigma_exp_linear = self._calculate_single_reason_sigma(sigma_exp_config.type,y_obs,sigma_exp_config.absolute.value,sigma_exp_config.relative.ratio)
                sigma_model_linear = self._calculate_single_reason_sigma(sigma_model_config.type,y_pred,sigma_model_config.absolute.value,sigma_model_config.relative.ratio)
                # 转换到似然空间
                sigma_exp_target = self._sigma_space_convert(space,sigma_exp_linear,y_obs,epsilon)
                sigma_model_target = self._sigma_space_convert(space,sigma_model_linear,y_pred,epsilon)
                # 计算总标准差
                sigma_total_target = self._calculate_sigma_total(sigma_exp_target,sigma_model_target)
                
                return sigma_total_target

            def _calculate_single_reason_sigma(self,sigma_type:str,y:np.ndarray,absolute_value:float,relative_ratio:float) -> np.ndarray:
                """计算单个因素的标准差"""

                sigma_calculator = self.sigma_dict[sigma_type]
                sigma_linear = sigma_calculator(absolute_value,relative_ratio,y)

                return sigma_linear

            @staticmethod
            def _absolute_sigma(absolute_value:float,relative_ratio:float,y:np.ndarray) -> np.ndarray:
                """计算绝对标准差"""

                sigma_linear = np.ones_like(y,dtype=float) * absolute_value

                return sigma_linear

            @staticmethod
            def _relative_sigma(absolute_value:float,relative_ratio:float,y:np.ndarray) -> np.ndarray:
                """计算物理线性空间中的相对标准差"""

                sigma_linear = np.abs(np.asarray(y,dtype=float)) * relative_ratio

                return sigma_linear

            @staticmethod
            def _calculate_sigma_total(sigma_exp:np.ndarray,sigma_model:np.ndarray) -> np.ndarray:
                """计算总标准差"""

                sigma_total = np.sqrt(sigma_exp**2 + sigma_model**2)

                return sigma_total
            
            @staticmethod
            def _sigma_space_convert(space:str,sigma_linear:np.ndarray,y:np.ndarray,epsilon:float) -> np.ndarray:
                """转换为似然空间的标准差"""

                if space == 'linear':
                    sigma_target = sigma_linear
                elif space == 'log10':
                    y_safe = np.maximum(np.abs(np.asarray(y, dtype=float)),epsilon)
                    sigma_target = sigma_linear / np.asarray(y_safe) / np.log(10)

                return sigma_target

        class PointCorrelation:
            """曲线内部观测点的相关性矩阵"""

            def __init__(self):
                # 初始化
                self.point_correlation_dict = {
                    "independent":self._independent,
                    "complete_matrix":self._complete_matrix,
                    "exponential":self._exponential
                }

            def point_correlation(self,point_correlation_old,data_obs_dict:dict,single_data_likelihood:InputCard.Likelihood.SingleDataLikelihood,data_order:list[str]):
                """计算单组实验点内部的相关性矩阵"""

                name = single_data_likelihood.name
                y_obs = data_obs_dict[name]
                n_points = len(y_obs)
                correlation_type = single_data_likelihood.point_correlation.type
                index = data_order.index(name)

                if point_correlation_old is not None and correlation_type in ["independent","complete_matrix","exponential"]:
                    correlation_updated = False
                    return point_correlation_old,correlation_updated
                
                source = single_data_likelihood.point_correlation.complete_matrix.source
                coordinate_name = single_data_likelihood.coordinate
                coordinate = data_obs_dict[coordinate_name] if correlation_type in ['exponential'] else None
                length_scale = single_data_likelihood.point_correlation.exponential.length_scale
                point_correlation_calculator = self.point_correlation_dict[correlation_type]
                correlation_matrix = point_correlation_calculator(n_points,source,coordinate,length_scale)

                correlation_updated = True

                return (index,index,correlation_matrix),correlation_updated

            @staticmethod
            def _independent(n_points,source,coordinate,length_scale):
                """独立观测点"""

                point_correlation = np.eye(n_points)

                return point_correlation

            @staticmethod
            def _complete_matrix(n_points,source,coordinate,length_scale):
                """从csv读取完整的内部观测点相关性矩阵"""

                point_correlation = np.loadtxt(source,delimiter=",")

                return point_correlation

            @staticmethod
            def _exponential(n_points,source,coordinate,length_scale):
                """指数相关项"""

                n = n_points
                point_correlation = np.zeros((n,n),dtype=float)
                for i in range(n):
                    for j in range(n):
                        point_correlation[i][j] = np.exp(-np.abs(coordinate[i]-coordinate[j])/length_scale)

                return point_correlation

        def __init__(
            self,
            single_data_likelihood_list: list[InputCard.Likelihood.SingleDataLikelihood],
            data_obs_dict: dict,
            data_order: list[str],
            calculate_point_correlation: bool
        ):
            # 读入配置
            self.data_obs_dict = data_obs_dict
            self.data_order = data_order
            self.calculate_point_correlation:bool = calculate_point_correlation
            # 初始化子类对象
            self.sigma = self.Sigma()
            self.point_correlation = self.PointCorrelation()
            # 初始化变量
            self.single_data_result_dict = {
                config.name: {
                    "sigma": None,
                    "point_correlation": None,
                    "residual": None,
                    "single_data_likelihood": config
                }
                for config in single_data_likelihood_list
            }
            
        def single_data_likelihood(self,data_pred_dict:dict):
            """更新单组数据的内部似然"""

            correlation_updated = False

            for y_name,result in self.single_data_result_dict.items():

                single_config = result["single_data_likelihood"]
                y_obs = np.asarray(self.data_obs_dict[y_name],dtype=float)
                y_pred = self._interpolate_prediction(single_config,y_name,data_pred_dict,self.data_obs_dict)

                result["sigma"] = self.sigma.sigma(single_config,y_pred,y_obs)
                result["residual"] = self._residual(single_config,y_pred,y_obs)
                if self.calculate_point_correlation:
                    result["point_correlation"],point_correlation_updated = self.point_correlation.point_correlation(result["point_correlation"],self.data_obs_dict,single_config,self.data_order)
                    correlation_updated = correlation_updated or point_correlation_updated
                else:
                    result["point_correlation"] = None

            # 按照data_order拼接残差与标准差
            residual_list = []
            sigma_list = []
            for name in self.data_order:
                result = self.single_data_result_dict[name]
                residual_list.append(result["residual"])
                sigma_list.append(result["sigma"])
            residual = np.concatenate(residual_list)
            sigma = np.concatenate(sigma_list)

            return self.single_data_result_dict,residual,sigma,correlation_updated

        @staticmethod
        def _residual(
            single_data_likelihood: InputCard.Likelihood.SingleDataLikelihood,
            y_pred: np.ndarray,
            y_obs: np.ndarray,
        ) -> np.ndarray:
            """计算似然空间的残差"""

            space = single_data_likelihood.space
            epsilon = single_data_likelihood.epsilon
            y_pred = np.asarray(y_pred, dtype=float)
            y_obs = np.asarray(y_obs, dtype=float)

            if space == "linear":
                residual = y_pred - y_obs
            elif space == "log10":
                y_pred_safe = np.maximum(y_pred, epsilon)
                y_obs_safe = np.maximum(y_obs, epsilon)
                residual = np.log10(y_pred_safe) - np.log10(y_obs_safe)

            return residual

        @staticmethod
        def _interpolate_prediction(single_config,y_name,data_pred_dict,data_obs_dict):
            """将预测数据插值到观测点位置"""

            y_pred_raw = np.asarray(data_pred_dict[y_name],dtype=float)
            coordinate_name = single_config.coordinate

            if coordinate_name is None:
                return y_pred_raw

            coordinate_pred = data_pred_dict[coordinate_name]
            coordinate_obs = data_obs_dict[coordinate_name]

            return np.interp(coordinate_obs,coordinate_pred,y_pred_raw)
        
    class CorrelationBuilder:
        """构造联合相关矩阵"""

        class CrossCorrelation:
            """构造两组数据之间的交叉相关矩阵"""

            def __init__(self,block_matrix:dict,data_order:list[str],single_data_likelihood_list:list[InputCard.Likelihood.SingleDataLikelihood]):
                # 读入配置
                self.block_matrix = block_matrix
                self.data_index_dict = {name:index for index,name in enumerate(data_order)}
                # 初始化变量
                self.cross_correlation_dict = {
                    "independent":self._independent,
                    "complete_matrix":self._complete_matrix,
                    "exponential":self._exponential,
                }
                self.single_config_dict = {config.name:config for config in single_data_likelihood_list}
                self.cross_correlation_list = None

            def cross_correlation(self,data_obs_dict:dict):
                """构造所有数据组之间的交叉相关矩阵"""

                if self.cross_correlation_list is not None:
                    return self.cross_correlation_list,False

                self.cross_correlation_list = []

                for config in self.block_matrix.values():
                    result = self._cross_correlation_yi_yj(config,data_obs_dict)
                    self.cross_correlation_list.extend(result)

                return self.cross_correlation_list,True

            def _cross_correlation_yi_yj(self,block_matrix_i_j:InputCard.Likelihood.JointLikelihood.Correlation.CrossCorrelation,data_obs_dict:dict) -> tuple[tuple[int,int,np.ndarray],tuple[int,int,np.ndarray]]:
                """构造指定两组数据之间的交叉相关矩阵"""

                name_i = block_matrix_i_j.data_i
                name_j = block_matrix_i_j.data_j
                loc_i = self.data_index_dict[name_i]
                loc_j = self.data_index_dict[name_j]
                cross_correlation = self._calculate_cross_correlation(name_i,name_j,block_matrix_i_j,data_obs_dict)

                return (loc_i,loc_j,cross_correlation),(loc_j,loc_i,cross_correlation.T)

            def _calculate_cross_correlation(self,name_i:str,name_j:str,cross_correlation_config:InputCard.Likelihood.JointLikelihood.Correlation.CrossCorrelation,data_obs_dict:dict) -> np.ndarray:
                """读取配置并计算两组数据之间的交叉相关矩阵"""

                cross_correlation_type = cross_correlation_config.type
                complete_matrix_source = cross_correlation_config.complete_matrix_source
                correlation_coef_i_j = cross_correlation_config.correlation_value
                exponential_length_scale = cross_correlation_config.exponential.length_scale

                if cross_correlation_type in ["exponential"]:
                    config_i = self.single_config_dict[name_i]
                    config_j = self.single_config_dict[name_j]
                    coordinate_i_name = config_i.coordinate
                    coordinate_j_name = config_j.coordinate
                    coordinate_i = data_obs_dict[coordinate_i_name]
                    coordinate_j = data_obs_dict[coordinate_j_name]
                else:
                    coordinate_i = None
                    coordinate_j = None

                n_i = len(data_obs_dict[name_i])
                n_j = len(data_obs_dict[name_j])

                cross_correlation_calculator = self.cross_correlation_dict[cross_correlation_type]
                cross_correlation = cross_correlation_calculator(n_i,n_j,complete_matrix_source,correlation_coef_i_j,coordinate_i,coordinate_j,exponential_length_scale)

                return cross_correlation

            @staticmethod
            def _independent(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scale) -> np.ndarray:
                """两组曲线相互独立"""

                cross_correlation = np.zeros((n_i,n_j),dtype=float)

                return cross_correlation

            @staticmethod
            def _complete_matrix(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scale) -> np.ndarray:
                """从csv读取完整的曲线间观测点相关性矩阵"""

                cross_correlation = np.loadtxt(source,delimiter=",")

                return cross_correlation

            @staticmethod
            def _exponential(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scale) -> np.ndarray:
                """指数曲线模型"""

                coordinate_i = np.asarray(coordinate_i, dtype=float)
                coordinate_j = np.asarray(coordinate_j, dtype=float)

                coordinate_distance = np.abs(coordinate_i[:, None] - coordinate_j[None, :])

                return correlation_coef_i_j * np.exp(-coordinate_distance / length_scale)

        def __init__(
            self,
            joint_likelihood:InputCard.Likelihood.JointLikelihood,
            single_data_likelihood_list:list[InputCard.Likelihood.SingleDataLikelihood]
        ):
            # 读入配置
            correlation_config = joint_likelihood.correlation
            self.data_order = joint_likelihood.data_order
            self.correlation_type = correlation_config.type
            self.complete_matrix_source = (correlation_config.complete_matrix_source)
            self.block_matrix = correlation_config.block_matrix
            # 初始化子类
            self.cross_correlation = self.CrossCorrelation(self.block_matrix,self.data_order,single_data_likelihood_list)
            self.correlation_builder_dict = {
                "complete_matrix": self._complete_matrix,
                "block_matrix": self._assemble_block_matrix,
            }
            self.correlation_matrix = None

        def assemble_complete_correlation_matrix(self,point_correlation_list,cross_correlation_list,correlation_updated:bool=False):
            """组合完整的相关性矩阵"""

            if self.correlation_matrix is not None and not correlation_updated:
                return self.correlation_matrix
            
            correlation_builder = self.correlation_builder_dict[self.correlation_type]
            self.correlation_matrix = correlation_builder(self.complete_matrix_source,point_correlation_list,cross_correlation_list,self.data_order)

            return self.correlation_matrix

        @staticmethod
        def _complete_matrix(source,point_correlation_list,cross_correlation_list,data_order) -> np.ndarray:
            """直接导入完整的相关性矩阵"""

            correlation_matrix = np.loadtxt(source,delimiter=",")

            return correlation_matrix

        @staticmethod
        def _assemble_block_matrix(source,point_correlation_list,cross_correlation_list,data_order) -> np.ndarray:
            """组装相关性矩阵快"""

            n_data = len(data_order)
            block_list = [[None for j in range(n_data)] for i in range(n_data)]

            correlation_list = (point_correlation_list + cross_correlation_list)

            for loc_i,loc_j,correlation_matrix in correlation_list:
                block_list[loc_i][loc_j] = correlation_matrix

            correlation_matrix = np.block(block_list)

            return correlation_matrix

    class Distribution: 
        """根据联合分布计算对数似然"""

        def __init__(self,distribution_config:InputCard.Likelihood.JointLikelihood.Distribution):
            self.distribution_config = distribution_config
            self.distribution_type = self.distribution_config.type
            self.distribution_dict = {
                "normal":self._normal,
            }

        def log_likelihood(self,residual,sigma,correlation_matrix):
            """计算联合对数似然"""

            distribution_calculator = self.distribution_dict[self.distribution_type]

            return distribution_calculator(residual,sigma,correlation_matrix)

        @staticmethod
        def _normal(residual,sigma,correlation_matrix):
            """多元正态分布"""

            covariance_matrix = sigma[:,None] * correlation_matrix * sigma[None,:]
            # 假设模型误差和实验误差整体上没有系统性偏差
            mean = np.zeros(len(residual))

            return multivariate_normal.logpdf(residual,mean=mean,cov=covariance_matrix)

    def __init__(
        self,
        likelihood_config: InputCard.Likelihood,
        data: Data
    ):
        # 读入配置
        self.likelihood_config = likelihood_config
        self.single_data_likelihood_config_list = self.likelihood_config.single_data_likelihood
        self.joint_likelihood_config = self.likelihood_config.joint_likelihood
        self.data_obs_dict = data.data_dict

        # 初始化变量
        calculate_point_correlation = self.joint_likelihood_config.correlation.type == "block_matrix"
                
        # 初始化子类
        self.single_data_likelihood = self.SingleDataLikelihood(
            self.single_data_likelihood_config_list,
            self.data_obs_dict,
            self.joint_likelihood_config.data_order,
            calculate_point_correlation
        )
        self.correlation_builder = self.CorrelationBuilder(
            self.joint_likelihood_config,
            self.single_data_likelihood_config_list
        )
        self.distribution = self.Distribution(
            self.joint_likelihood_config.distribution
        )

    def log_likelihood(self,data_pred_dict:dict):
        """参数后验"""

        # 单曲线内部参数的残差 标准差 相关性矩阵
        single_data_result_dict,residual,sigma,point_correlation_updated = self.single_data_likelihood.single_data_likelihood(data_pred_dict)

        if self.correlation_builder.correlation_type == "block_matrix":
            # 提取单曲线内部参数的相关性矩阵列表
            point_correlation_list = []
            for value in single_data_result_dict.values():
                point_correlation = value["point_correlation"]
                point_correlation_list.append(point_correlation)
            # 提取曲线间参数的相关性交叉矩阵列表
            cross_correlation_list,cross_correlation_updated = self.correlation_builder.cross_correlation.cross_correlation(self.data_obs_dict)
        else:
            point_correlation_list = []
            cross_correlation_list = []
            cross_correlation_updated = False

        correlation_updated = point_correlation_updated or cross_correlation_updated
        # 构造完整的相关性矩阵
        correlation_matrix = self.correlation_builder.assemble_complete_correlation_matrix(point_correlation_list,cross_correlation_list,correlation_updated)
        # 计算似然
        log_likelihood = self.distribution.log_likelihood(residual,sigma,correlation_matrix)

        return log_likelihood

    
        
