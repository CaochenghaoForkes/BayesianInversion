from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.special import gammaln
from scipy.spatial import QhullError

from config import InputCard
from data import ObservedData
from model import ModelResult


FLOAT_TINY = np.finfo(np.float64).tiny


@dataclass
class SingleLikelihoodResult:
    """一个数据集进入联合似然前的计算结果"""

    dataset: str
    residual: np.ndarray
    sigma: np.ndarray
    point_correlation: tuple[int,int,np.ndarray] | None

class Likelihood:
    """似然"""

    @staticmethod
    def _coordinate_matrix(
        data: ObservedData | ModelResult,
        dataset: str,
        coordinate_components: list[str],
    ) -> np.ndarray:
        """提取点×坐标分量矩阵"""

        point_count = data.values[dataset].shape[0]
        if not coordinate_components:
            return np.empty((point_count,0),dtype=float)
        columns = [data.component(dataset,component) for component in coordinate_components]

        return np.column_stack(columns)

    class SingleDataLikelihood:
        """计算各数据集进入联合似然前的结果"""

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
                # 计算单独因素的标准差
                sigma_exp_linear = self._calculate_single_reason_sigma(sigma_exp_config.type,y_obs,sigma_exp_config.absolute.value,sigma_exp_config.relative.ratio)
                sigma_model_linear = self._calculate_single_reason_sigma(sigma_model_config.type,y_pred,sigma_model_config.absolute.value,sigma_model_config.relative.ratio)
                # 转换到似然空间
                sigma_exp_target = self._sigma_space_convert(space,sigma_exp_linear,y_obs)
                sigma_model_target = self._sigma_space_convert(space,sigma_model_linear,y_pred)
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
            def _sigma_space_convert(space:str,sigma_linear:np.ndarray,y:np.ndarray) -> np.ndarray:
                """转换为似然空间的标准差"""

                if space == 'linear':
                    sigma_target = sigma_linear
                elif space == 'log10':
                    y_safe = np.maximum(np.abs(np.asarray(y,dtype=float)),FLOAT_TINY)
                    sigma_target = sigma_linear / np.asarray(y_safe) / np.log(10)

                return sigma_target

        class PointCorrelation:
            """曲线内部观测点的相关性矩阵"""

            def __init__(self):
                self.point_correlation_dict = {
                    "independent":self._independent,
                    "complete_matrix":self._complete_matrix,
                    "exponential":self._exponential
                }
                self.cache = {}

            def point_correlation(
                self,
                data: ObservedData,
                config: InputCard.Likelihood.SingleDataLikelihood,
                data_order: list[str],
            ):
                """计算单组实验点内部的相关性矩阵"""

                dataset = config.dataset
                if dataset in self.cache:
                    return self.cache[dataset],False

                y_obs = data.component(dataset,config.value_component)
                n_points = len(y_obs)
                index = data_order.index(dataset)
                correlation_config = config.point_correlation
                correlation_type = correlation_config.type
                source = correlation_config.complete_matrix.source
                length_scale_map = correlation_config.exponential.length_scales
                length_scales = np.asarray([length_scale_map[name] for name in config.coordinate_components],dtype=float)
                coordinate = Likelihood._coordinate_matrix(data,dataset,config.coordinate_components) if correlation_type == "exponential" else None
                point_correlation_calculator = self.point_correlation_dict[correlation_type]
                correlation_matrix = point_correlation_calculator(
                    n_points,source,coordinate,length_scales
                )
                result = (index,index,correlation_matrix)
                self.cache[dataset] = result

                return result,True

            @staticmethod
            def _independent(n_points,source,coordinate,length_scales):
                """独立观测点"""

                point_correlation = np.eye(n_points)

                return point_correlation

            @staticmethod
            def _complete_matrix(n_points,source,coordinate,length_scales):
                """从csv读取完整的内部观测点相关性矩阵"""

                point_correlation = np.loadtxt(source,delimiter=",")

                return point_correlation

            @staticmethod
            def _exponential(n_points,source,coordinate,length_scales):
                """指数相关项"""

                coordinate = np.asarray(coordinate,dtype=float)
                scale = np.asarray(length_scales,dtype=float)
                scaled_difference = (coordinate[:,None] - coordinate[None,:]) / scale
                coordinate_distance = np.linalg.norm(scaled_difference,axis=-1)

                return np.exp(-coordinate_distance)

        def __init__(
            self,
            single_data_likelihood_list: list[InputCard.Likelihood.SingleDataLikelihood],
            data: ObservedData,
            data_order: list[str],
            calculate_point_correlation: bool
        ):
            self.configs = single_data_likelihood_list
            self.data = data
            self.data_order = data_order
            self.calculate_point_correlation = calculate_point_correlation
            self.sigma = self.Sigma()
            self.point_correlation = self.PointCorrelation()
            
        def evaluate(
            self,
            model_result: ModelResult,
        ):
            """计算每个数据集进入联合似然前的结果"""

            results = {}
            correlation_updated = False

            for config in self.configs:
                dataset = config.dataset
                y_obs = self.data.component(dataset,config.value_component)
                y_pred = self._interpolate_prediction(config,self.data,model_result)

                sigma = self.sigma.sigma(config,y_pred,y_obs)
                residual = self._residual(config,y_pred,y_obs)
                point_correlation = None

                if self.calculate_point_correlation:
                    point_correlation,point_correlation_updated = self.point_correlation.point_correlation(self.data,config,self.data_order)
                    correlation_updated = correlation_updated or point_correlation_updated

                results[dataset] = SingleLikelihoodResult(
                    dataset=dataset,
                    residual=residual,
                    sigma=sigma,
                    point_correlation=point_correlation,
                )

            # 按照data_order拼接残差与标准差
            residual = np.concatenate([results[name].residual for name in self.data_order])
            sigma = np.concatenate([results[name].sigma for name in self.data_order])

            return results,residual,sigma,correlation_updated

        @staticmethod
        def _residual(
            single_data_likelihood: InputCard.Likelihood.SingleDataLikelihood,
            y_pred: np.ndarray,
            y_obs: np.ndarray,
        ) -> np.ndarray:
            """计算似然空间的残差"""

            space = single_data_likelihood.space
            y_pred = np.asarray(y_pred, dtype=float)
            y_obs = np.asarray(y_obs, dtype=float)

            if space == "linear":
                residual = y_pred - y_obs
            elif space == "log10":
                y_pred_safe = np.maximum(y_pred,FLOAT_TINY)
                y_obs_safe = np.maximum(y_obs,FLOAT_TINY)
                residual = np.log10(y_pred_safe) - np.log10(y_obs_safe)

            return residual

        @staticmethod
        def _interpolate_prediction(
            config: InputCard.Likelihood.SingleDataLikelihood,
            data: ObservedData,
            model_result: ModelResult,
        ):
            """将预测数据插值到观测点位置"""

            dataset = config.dataset
            y_pred_raw = model_result.component(dataset,config.value_component)
            coordinate_components = config.coordinate_components

            if not coordinate_components:
                return y_pred_raw

            coordinate_pred = Likelihood._coordinate_matrix(model_result,dataset,coordinate_components)
            coordinate_obs = Likelihood._coordinate_matrix(data,dataset,coordinate_components)

            if len(coordinate_components) == 1:
                return Likelihood.SingleDataLikelihood._interpolate_1d(coordinate_pred[:,0],coordinate_obs[:,0],y_pred_raw)

            same_coordinates = coordinate_pred.shape == coordinate_obs.shape and np.allclose(coordinate_pred,coordinate_obs)
            if same_coordinates:
                return y_pred_raw

            return Likelihood.SingleDataLikelihood._interpolate_nd(coordinate_pred,coordinate_obs,y_pred_raw)

        @staticmethod
        def _interpolate_1d(
            coordinate_pred: np.ndarray,
            coordinate_obs: np.ndarray,
            y_pred: np.ndarray,
        ) -> np.ndarray:
            """将一维预测结果插值到观测坐标"""

            order = np.argsort(coordinate_pred)

            return np.interp(coordinate_obs,coordinate_pred[order],y_pred[order])

        @staticmethod
        def _interpolate_nd(
            coordinate_pred: np.ndarray,
            coordinate_obs: np.ndarray,
            y_pred: np.ndarray,
        ) -> np.ndarray:
            """将多维规则网格或散点预测结果插值到观测坐标"""

            lower = np.min(coordinate_pred,axis=0)
            spans = np.ptp(coordinate_pred,axis=0)
            varying = spans > 0.0
            if not np.allclose(coordinate_obs[:,~varying],lower[~varying]):
                raise ValueError("观测坐标超出模型固定坐标分量的位置")

            coordinate_pred = coordinate_pred[:,varying]
            coordinate_obs = coordinate_obs[:,varying]
            spans = spans[varying]

            if coordinate_pred.shape[1] == 0:
                if not np.allclose(y_pred,y_pred[0]):
                    raise ValueError("相同预测坐标对应了不同模型输出")
                return np.full(coordinate_obs.shape[0],y_pred[0])

            coordinate_pred = (coordinate_pred-lower[varying])/spans
            coordinate_obs = (coordinate_obs-lower[varying])/spans

            if coordinate_pred.shape[1] == 1:
                return Likelihood.SingleDataLikelihood._interpolate_1d(coordinate_pred[:,0],coordinate_obs[:,0],y_pred)

            regular_grid = Likelihood.SingleDataLikelihood._regular_grid(coordinate_pred,y_pred)
            if regular_grid is not None:
                axes,grid_values = regular_grid
                interpolator = RegularGridInterpolator(
                    axes,
                    grid_values,
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interpolated = interpolator(coordinate_obs)
            else:
                try:
                    interpolator = LinearNDInterpolator(
                        coordinate_pred,
                        y_pred,
                        fill_value=np.nan,
                    )
                    interpolated = interpolator(coordinate_obs)
                except QhullError as error:
                    raise ValueError(
                        "模型坐标无法构造多维线性插值"
                    ) from error

            if not np.all(np.isfinite(interpolated)):
                raise ValueError("观测坐标超出模型多维插值范围")

            return np.asarray(interpolated,dtype=float)

        @staticmethod
        def _regular_grid(
            coordinates: np.ndarray,
            values: np.ndarray,
        ) -> tuple[list[np.ndarray],np.ndarray] | None:
            """识别展平的规则张量网格并恢复网格数值"""

            axes = [
                np.unique(coordinates[:,axis])
                for axis in range(coordinates.shape[1])
            ]
            grid_shape = tuple(len(axis) for axis in axes)
            if np.prod(grid_shape) != len(coordinates):
                return None

            grid_indices = tuple(
                np.searchsorted(axis,coordinates[:,column])
                for column,axis in enumerate(axes)
            )
            flat_indices = np.ravel_multi_index(
                grid_indices,
                grid_shape,
            )
            if len(np.unique(flat_indices)) != len(coordinates):
                return None

            grid_values = np.empty(grid_shape,dtype=float)
            grid_values[grid_indices] = values

            return axes,grid_values

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
                self.single_config_dict = {config.dataset: config for config in single_data_likelihood_list}
                self.cross_correlation_list = None

            def cross_correlation(self,data: ObservedData):
                """构造所有数据组之间的交叉相关矩阵"""

                if self.cross_correlation_list is not None:
                    return self.cross_correlation_list,False

                self.cross_correlation_list = []

                for config in self.block_matrix.values():
                    result = self._cross_correlation_yi_yj(config,data)
                    self.cross_correlation_list.extend(result)

                return self.cross_correlation_list,True

            def _cross_correlation_yi_yj(
                self,
                config: InputCard.Likelihood.JointLikelihood.Correlation.CrossCorrelation,
                data: ObservedData,
            ) -> tuple[tuple[int,int,np.ndarray],tuple[int,int,np.ndarray]]:
                """构造指定两组数据之间的交叉相关矩阵"""

                name_i = config.data_i
                name_j = config.data_j
                loc_i = self.data_index_dict[name_i]
                loc_j = self.data_index_dict[name_j]
                cross_correlation = self._calculate_cross_correlation(name_i,name_j,config,data)

                return (loc_i,loc_j,cross_correlation),(loc_j,loc_i,cross_correlation.T)

            def _calculate_cross_correlation(
                self,
                name_i: str,
                name_j: str,
                cross_correlation_config: InputCard.Likelihood.JointLikelihood.Correlation.CrossCorrelation,
                data: ObservedData,
            ) -> np.ndarray:
                """读取配置并计算两组数据之间的交叉相关矩阵"""

                cross_correlation_type = cross_correlation_config.type
                complete_matrix_source = cross_correlation_config.complete_matrix.source
                correlation_coef_i_j = cross_correlation_config.exponential.correlation_value
                exponential_length_scales = cross_correlation_config.exponential.length_scales

                if cross_correlation_type in ["exponential"]:
                    config_i = self.single_config_dict[name_i]
                    config_j = self.single_config_dict[name_j]
                    coordinate_i = Likelihood._coordinate_matrix(
                        data,
                        name_i,
                        config_i.coordinate_components,
                    )
                    coordinate_j = Likelihood._coordinate_matrix(
                        data,
                        name_j,
                        config_j.coordinate_components,
                    )
                else:
                    coordinate_i = None
                    coordinate_j = None

                config_i = self.single_config_dict[name_i]
                config_j = self.single_config_dict[name_j]
                n_i = len(data.component(name_i,config_i.value_component))
                n_j = len(data.component(name_j,config_j.value_component))

                cross_correlation_calculator = self.cross_correlation_dict[cross_correlation_type]
                cross_correlation = cross_correlation_calculator(
                    n_i,n_j,complete_matrix_source,correlation_coef_i_j,
                    coordinate_i,coordinate_j,exponential_length_scales,
                )

                return cross_correlation

            @staticmethod
            def _independent(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scales) -> np.ndarray:
                """两组曲线相互独立"""

                cross_correlation = np.zeros((n_i,n_j),dtype=float)

                return cross_correlation

            @staticmethod
            def _complete_matrix(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scales) -> np.ndarray:
                """从csv读取完整的曲线间观测点相关性矩阵"""

                cross_correlation = np.loadtxt(source,delimiter=",")

                return cross_correlation

            @staticmethod
            def _exponential(n_i,n_j,source,correlation_coef_i_j,coordinate_i,coordinate_j,length_scales) -> np.ndarray:
                """多维坐标下的指数交叉相关模型"""

                coordinate_i = np.asarray(coordinate_i,dtype=float)
                coordinate_j = np.asarray(coordinate_j,dtype=float)
                scale = np.asarray(length_scales,dtype=float)

                scaled_difference = (coordinate_i[:,None,:] - coordinate_j[None,:,:]) / scale
                coordinate_distance = np.linalg.norm(scaled_difference,axis=-1)

                return correlation_coef_i_j * np.exp(-coordinate_distance)

        def __init__(
            self,
            joint_likelihood:InputCard.Likelihood.JointLikelihood,
            single_data_likelihood_list:list[InputCard.Likelihood.SingleDataLikelihood]
        ):
            # 读入配置
            correlation_config = joint_likelihood.correlation
            self.data_order = joint_likelihood.data_order
            self.correlation_type = correlation_config.type
            self.complete_matrix_source = correlation_config.complete_matrix.source
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
            self.degrees_of_freedom = (
                self.distribution_config.student_t.degrees_of_freedom
            )
            self.distribution_dict = {
                "normal":self._normal,
                "student_t":self._student_t,
            }
            self.factor_cache = {}

        def log_likelihood(
            self,
            residual: np.ndarray,
            sigma: np.ndarray,
            correlation_matrix: np.ndarray,
            cache_key: str,
        ) -> float:
            """计算联合对数似然"""

            residual = np.asarray(residual,dtype=float)
            sigma = np.asarray(sigma,dtype=float)
            correlation = np.asarray(correlation_matrix,dtype=float)
            if residual.ndim != 1 or sigma.ndim != 1:
                raise ValueError("残差与标准差必须是一维数组")
            if residual.shape != sigma.shape:
                raise ValueError("残差与标准差的形状不一致")
            if correlation.shape != (residual.size,residual.size):
                raise ValueError("相关矩阵形状与残差长度不一致")
            if not np.all(np.isfinite(residual)):
                return -np.inf
            if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
                return -np.inf

            cholesky,log_determinant = self._correlation_factor(correlation,cache_key)
            standardized_residual = residual / sigma
            whitened_residual = solve_triangular(cholesky,standardized_residual,lower=True,check_finite=False)
            dimension = residual.size
            quadratic_form = float(whitened_residual@whitened_residual)
            covariance_log_determinant = log_determinant + 2.0*np.sum(np.log(sigma))

            distribution_calculator = self.distribution_dict.get(self.distribution_type)
            if distribution_calculator is None:
                raise NotImplementedError(
                    f"未实现的联合分布: {self.distribution_type}"
                )

            return float(
                distribution_calculator(
                    dimension,
                    quadratic_form,
                    covariance_log_determinant,
                )
            )

        @staticmethod
        def _normal(
            dimension: int,
            quadratic_form: float,
            covariance_log_determinant: float,
        ) -> float:
            """多元正态分布"""

            return -0.5*(
                dimension*np.log(2.0*np.pi)
                + covariance_log_determinant
                + quadratic_form
            )

        def _student_t(
            self,
            dimension: int,
            quadratic_form: float,
            covariance_log_determinant: float,
        ) -> float:
            """sigma 表示真实标准差的多元 Student-t 分布"""

            degrees_of_freedom = self.degrees_of_freedom
            scale_ratio = (degrees_of_freedom-2.0) / degrees_of_freedom
            scale_log_determinant = covariance_log_determinant + dimension * np.log(scale_ratio)
            scale_quadratic_form = quadratic_form / scale_ratio

            return (
                gammaln((degrees_of_freedom+dimension)/2.0)
                - gammaln(degrees_of_freedom/2.0)
                - 0.5*dimension*np.log(degrees_of_freedom*np.pi)
                - 0.5*scale_log_determinant
                - 0.5*(degrees_of_freedom+dimension)
                * np.log1p(scale_quadratic_form/degrees_of_freedom)
            )

        def _correlation_factor(
            self,
            correlation_matrix: np.ndarray,
            cache_key: str,
        ) -> tuple[np.ndarray,float]:
            """获取并缓存相关矩阵的Cholesky分解和对数行列式"""

            if cache_key in self.factor_cache:
                return self.factor_cache[cache_key]

            correlation = np.asarray(correlation_matrix,dtype=float)
            if not np.all(np.isfinite(correlation)):
                raise ValueError("相关矩阵包含非有限数值")
            if not np.allclose(correlation,correlation.T):
                raise ValueError("相关矩阵必须对称")
            if not np.allclose(np.diag(correlation),1.0):
                raise ValueError("相关矩阵的对角元必须为1")

            try:
                cholesky = np.linalg.cholesky(correlation)
            except np.linalg.LinAlgError as error:
                raise ValueError("相关矩阵必须为正定矩阵") from error
            log_determinant = float(2.0*np.sum(np.log(np.diag(cholesky))))
            factor = (cholesky,log_determinant)
            self.factor_cache[cache_key] = factor

            return factor

    def __init__(
        self,
        likelihood_config: InputCard.Likelihood,
        data: ObservedData
    ):
        # 读入配置
        self.likelihood_config = likelihood_config
        self.single_data_likelihood_config_list = self.likelihood_config.single_data_likelihood
        self.joint_likelihood_config = self.likelihood_config.joint_likelihood
        self.data = data
        # 初始化变量
        calculate_point_correlation = self.joint_likelihood_config.correlation.type == "block_matrix"
        # 初始化子类
        self.single_data_evaluator = self.SingleDataLikelihood(
            self.single_data_likelihood_config_list,
            self.data,
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

    def _independent_block_log_likelihood(
        self,
        single_data_results: dict[str,SingleLikelihoodResult],
    ) -> float | None:
        # 数据组之间完全独立：
        # 分别计算每组似然，避免构造完整大矩阵
        independent_blocks = (
            self.correlation_builder.correlation_type == "block_matrix"
            and all(
                config.type == "independent"
                for config in self.correlation_builder.block_matrix.values()
            )
        )
        if independent_blocks:
            log_likelihood = 0.0

            for name in self.joint_likelihood_config.data_order:
                result = single_data_results[name]
                if result.point_correlation is None:
                    raise RuntimeError(
                        f"数据集 {name} 缺少点相关矩阵"
                    )
                _,_,correlation = result.point_correlation

                log_likelihood += self.distribution.log_likelihood(
                    result.residual,
                    result.sigma,
                    correlation,
                    cache_key=f"dataset:{name}",
                )

            return log_likelihood

        return None

    def log_likelihood(self,model_result: ModelResult):
        """计算给定模型结果的联合对数似然"""

        # 单曲线内部参数的残差 标准差 相关性矩阵
        single_data_results,residual,sigma,point_correlation_updated = self.single_data_evaluator.evaluate(model_result)

        # 简化计算
        log_likelihood = self._independent_block_log_likelihood(single_data_results)
        if log_likelihood is not None:
            return log_likelihood

        # 曲线间存在相关性
        if self.correlation_builder.correlation_type == "block_matrix":
            # 提取单曲线内部参数的相关性矩阵列表
            point_correlation_list = []
            for result in single_data_results.values():
                if result.point_correlation is None:
                    raise RuntimeError(
                        f"数据集 {result.dataset} 缺少点相关矩阵"
                    )
                point_correlation_list.append(result.point_correlation)
            # 提取曲线间参数的相关性交叉矩阵列表
            cross_correlation_list,cross_correlation_updated = self.correlation_builder.cross_correlation.cross_correlation(self.data)
        else:
            point_correlation_list = []
            cross_correlation_list = []
            cross_correlation_updated = False

        correlation_updated = point_correlation_updated or cross_correlation_updated
        # 构造完整的相关性矩阵
        correlation_matrix = self.correlation_builder.assemble_complete_correlation_matrix(point_correlation_list,cross_correlation_list,correlation_updated)
        # 计算似然
        log_likelihood = self.distribution.log_likelihood(
            residual,
            sigma,
            correlation_matrix,
            cache_key="joint",
        )

        return log_likelihood

    
        
