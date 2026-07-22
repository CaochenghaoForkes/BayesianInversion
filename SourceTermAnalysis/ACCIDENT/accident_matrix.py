from ACCIDENT.accident_xml_reader import AccidentXMLReader
from ACCIDENT.accident_solver import AccidentSolver
from TOOL.cosmos_general_function import Registry
from typing import List
import numpy as np
import sys
from scipy.linalg import block_diag

class TimeSeries:
    '''
        时间差值表
    '''
    def __init__(self, times, values: List[float],typ):
        
        self.typ = typ
        assert len(times) == len(values) and len(times) >= 1
        # 保证有序
        pairs = sorted(zip(times, values), key=lambda x: x[0])
        self._times, self._values = zip(*pairs)
        if isinstance(times[0],float):
            self.categoty = 'at_time'
        elif isinstance(times[0],list):
            self.categoty = 'at_step'
            self._times_total = [x for sublist in self._times for x in sublist]

    def at(self,time,step):

        if self.categoty == 'at_time':
            return self._at_time(time)
        elif self.categoty == 'at_step':
            return self._at_step(step)

    def _at_step(self, step: int):

        if step in self._times_total:
            for idx,_time in enumerate(self._times):
                if step in _time:
                    return self._values[idx]
        else:
            return 0.0

    def _at_time(self, t: float, mode: str = "linear") -> float:
        
        # 确定某时刻的值
        ts, vs = self._times, self._values
        # 左右边界
        if t <= ts[0]: return vs[0]
        if t >= ts[-1]: return vs[-1]
        # 二分
        import bisect
        i = bisect.bisect_right(ts, t) - 1  # ts[i] <= t < ts[i+1]
        if mode == "step":
            return vs[i]
        # 线性
        t0, t1 = ts[i], ts[i+1]
        v0, v1 = vs[i], vs[i+1]
        w = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return v0 + w * (v1 - v0)
    
class AccidentMatrix:
    '''
        事故矩阵
    '''
    class Transfer:
        '''
            转移信息
        '''
        def __init__(self,reader:AccidentXMLReader):
            
            # 爆发模式
            self.burst_process_model = reader.burst_process_model
            
            # Cv转移信息
            self.transfer_info = reader.transfer_info
            self.n_nuclide = reader.n_nuclide
            self.n_cv = reader.n_cv
            
            # 预处理为时间序列对象
            self._init_transfer_info()

        def _init_transfer_info(self):
            '''
                预处理每个Cv的每个核素的转移信息
            '''
            self.transfer_time_series = [[[] for __ in range(self.n_nuclide)] for _ in range(self.n_cv)]
            self.pure_coef_matrix = [[[] for __ in range(self.n_nuclide)] for _ in range(self.n_cv)]
            for cv_idx in range(self.n_cv):
                for nuclide_idx in range(self.n_nuclide):
                    for transfer_idx,transfer_info in enumerate(self.transfer_info[cv_idx][nuclide_idx]):
                        label = transfer_info['label']
                        typ = transfer_info['type']
                        value = transfer_info['value']
                        time_or_step = transfer_info['time_or_step']
                        self.transfer_time_series[cv_idx][nuclide_idx].append(TimeSeries(time_or_step,value,typ))
                        self.pure_coef_matrix[cv_idx][nuclide_idx].append(transfer_info['pur_coef'])

            # 累计量矩阵
            self.transfer_rate_detailed_cumulant_matrix = np.array([[np.zeros(len(self.transfer_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv-1)],dtype=object)
            self.transfer_rate_pured_detailed_cumulant_matrix = np.array([[np.zeros(len(self.transfer_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv-1)],dtype=object)

        def update_transfer_diag(self,time,step):
            '''
                转移对角阵
            '''
            # 转移系数矩阵
            self.transfer_rate_matrix = np.zeros((self.n_cv-1,self.n_nuclide))
            self.transfer_rate_detailed_matrix = [[np.zeros(len(self.transfer_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv-1)]
            self.transfer_rate_pured_matrix = np.zeros((self.n_cv-1,self.n_nuclide))
            self.transfer_rate_pured_detailed_matrix = [[np.zeros(len(self.transfer_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv-1)]
            for cv_idx in range(self.n_cv-1):
                for nuclide_idx in range(self.n_nuclide):
                    for transfer_idx,transfer_time_series in enumerate(self.transfer_time_series[cv_idx][nuclide_idx]):
                        if transfer_time_series.typ == 'rate':
                            transfer_rate = transfer_time_series.at(time,step)
                            pur_coef = self.pure_coef_matrix[cv_idx][nuclide_idx][transfer_idx]
                            self.transfer_rate_matrix[cv_idx][nuclide_idx] += transfer_rate
                            self.transfer_rate_pured_matrix[cv_idx][nuclide_idx] += transfer_rate * (1-pur_coef)
                            self.transfer_rate_detailed_matrix[cv_idx][nuclide_idx][transfer_idx] = transfer_rate
                            self.transfer_rate_pured_detailed_matrix[cv_idx][nuclide_idx][transfer_idx] = transfer_rate * (1-pur_coef)
                        elif self.burst_process_model == 'StepBurst' and transfer_time_series.typ == 'burst':
                            transfer_rate = transfer_time_series.at(time,step)
                            pur_coef = self.pure_coef_matrix[cv_idx][nuclide_idx][transfer_idx]
                            self.transfer_rate_matrix[cv_idx][nuclide_idx] += transfer_rate
                            self.transfer_rate_pured_matrix[cv_idx][nuclide_idx] += transfer_rate * (1-pur_coef)
                            self.transfer_rate_detailed_matrix[cv_idx][nuclide_idx][transfer_idx] = transfer_rate 
                            self.transfer_rate_pured_detailed_matrix[cv_idx][nuclide_idx][transfer_idx] = transfer_rate * (1-pur_coef)
            # 构建转移对角阵
            self.transfer_diag_list = []
            self.transfer_pured_diag_list = []
            self.transfer_rate_vector = []
            self.transfer_rate_pured_vector = []
            for cv_idx in range(self.n_cv): # 最后一个控制体不可能有下游了
                if cv_idx != self.n_cv - 1:
                    self.transfer_diag_list.append(np.diag(self.transfer_rate_matrix[cv_idx]))
                    self.transfer_pured_diag_list.append(np.diag(self.transfer_rate_pured_matrix[cv_idx]))
                    self.transfer_rate_vector += list(self.transfer_rate_matrix[cv_idx])
                    self.transfer_rate_pured_vector += list(self.transfer_rate_pured_matrix[cv_idx])
                else:
                    self.transfer_rate_vector += [0.0]*self.n_nuclide
                    self.transfer_rate_pured_vector += [0.0]*self.n_nuclide
            self.transfer_rate_vector = np.array(self.transfer_rate_vector)
            self.transfer_rate_pured_vector = np.array(self.transfer_rate_pured_vector)

    class Remove:
        '''
            消失信息
        '''
        def __init__(self,reader:AccidentXMLReader):
            
            # 爆发模式
            self.burst_process_model = reader.burst_process_model
            
            # Cv消失信息
            self.remove_info = reader.remove_info
            self.n_nuclide = reader.n_nuclide
            self.n_cv = reader.n_cv
            
            # 预处理为时间序列对象
            self._init_remove_info()

        def _init_remove_info(self):
            '''
                预处理每个Cv的每个核素的消失信息
            '''
            self.remove_time_series = [[[] for __ in range(self.n_nuclide)] for _ in range(self.n_cv)]
            for cv_idx in range(self.n_cv):
                for nuclide_idx in range(self.n_nuclide):
                    for remove_idx,remove_info in enumerate(self.remove_info[cv_idx][nuclide_idx]):
                        label = remove_info['label']
                        typ = remove_info['type']
                        value = remove_info['value']
                        time_or_step = remove_info['time_or_step']
                        self.remove_time_series[cv_idx][nuclide_idx].append(TimeSeries(time_or_step,value,typ))
                    
            # 累计量矩阵
            self.remove_rate_detailed_cumulant_matrix = np.array([[np.zeros(len(self.remove_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv)],dtype=object)
          
        def update_remove_vector(self,time,step):
            '''
                消失向量
            '''
            # 消失系数矩阵
            self.remove_rate_matrix = np.zeros((self.n_cv,self.n_nuclide))
            self.remove_rate_detailed_matrix = [[np.zeros(len(self.remove_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv)]
            for cv_idx in range(self.n_cv):
                for nuclide_idx in range(self.n_nuclide):
                    for remove_idx,remove_time_series in enumerate(self.remove_time_series[cv_idx][nuclide_idx]):
                        if remove_time_series.typ == 'rate':
                            remove_rate = remove_time_series.at(time,step)
                            self.remove_rate_matrix[cv_idx][nuclide_idx] += remove_rate
                            self.remove_rate_detailed_matrix[cv_idx][nuclide_idx][remove_idx] = remove_rate
                        elif self.burst_process_model == 'StepBurst' and remove_time_series.typ == 'burst':
                            remove_rate = remove_time_series.at(time,step)
                            self.remove_rate_matrix[cv_idx][nuclide_idx] += remove_rate
                            self.remove_rate_detailed_matrix[cv_idx][nuclide_idx][remove_idx] = remove_rate

            # 构建消失向量  
            self.remove_vector = self.remove_rate_matrix.flatten()

    class Release:
        '''
            释放信息
        '''
        def __init__(self,reader:AccidentXMLReader):
            
            # 爆发模式
            self.burst_process_model = reader.burst_process_model
            
            # 伪核素残差
            self.pseudo_nuclide_residual = reader.solver_dict['pseudo_nuclide_residual']
            
            # Cv释放信息
            self.release_info = reader.release_info
            self.n_nuclide = reader.n_nuclide
            self.n_cv = reader.n_cv
            
            # 预处理为时间序列对象
            self._init_release_info()

        def _init_release_info(self):
            '''
                预处理每个Cv的每个核素的释放信息
            '''
            self.release_time_series = [[[] for __ in range(self.n_nuclide)] for _ in range(self.n_cv)]
            for cv_idx in range(self.n_cv):
                for nuclide_idx in range(self.n_nuclide):
                    for release_idx,release_info in enumerate(self.release_info[cv_idx][nuclide_idx]):
                        label = release_info['label']
                        typ = release_info['type']
                        value = release_info['value']
                        time_or_step = release_info['time_or_step']
                        self.release_time_series[cv_idx][nuclide_idx].append(TimeSeries(time_or_step,value,typ))

            # 累计量矩阵
            self.release_rate_detailed_cumulant_matrix = np.array([[np.zeros(len(self.release_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv)],dtype=object)

        def update_release_vector(self,time,step,time_step):
            '''
                释放向量
            '''
            # 释放系数矩阵
            self.release_rate_matrix = np.zeros((self.n_cv,self.n_nuclide))
            self.release_rate_detailed_matrix = [[np.zeros(len(self.release_time_series[cv_idx][nuclide_idx]),dtype=float) for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv)]
            for cv_idx in range(self.n_cv):
                for nuclide_idx in range(self.n_nuclide):
                    for release_idx,release_time_series in enumerate(self.release_time_series[cv_idx][nuclide_idx]):
                        if release_time_series.typ == 'rate':
                            release_rate = release_time_series.at(time,step)
                            self.release_rate_matrix[cv_idx][nuclide_idx] += release_rate
                            self.release_rate_detailed_matrix[cv_idx][nuclide_idx][release_idx] = release_rate
                        elif self.burst_process_model == 'StepBurst' and release_time_series.typ == 'burst':
                            release_rate = release_time_series.at(time,step)
                            self.release_rate_matrix[cv_idx][nuclide_idx] += release_rate
                            self.release_rate_detailed_matrix[cv_idx][nuclide_idx][release_idx] = release_rate
            
            # 构建释放向量
            self.release_vector = self.release_rate_matrix.flatten()
            if np.sum(self.release_vector) == 0.0:
                self.pesudo_nuclide_inventory = 0.0
                self.pesudo_decay_transfer_vector = np.zeros_like(self.release_vector)
                self.pesudo_decay_constant = 0.0
            else:
                self.pesudo_nuclide_inventory = np.sum(self.release_vector)*time_step / self.pseudo_nuclide_residual
                self.pesudo_decay_transfer_vector = self.release_vector / self.pesudo_nuclide_inventory
                self.pesudo_decay_constant = np.sum(self.pesudo_decay_transfer_vector)

    def __init__(self,reader:AccidentXMLReader):
        
        # 爆发模式
        self.burst_process_model = reader.burst_process_model
        
        # 基本信息
        self.n_nuclide = reader.n_nuclide
        self.n_cv = reader.n_cv

        # 衰变矩阵
        self.decay_matrix = reader.decay_matirx
        self.decay_detailed_cumulant_matrix = np.array([[0.0 for nuclide_idx in range(self.n_nuclide)] for cv_idx in range(self.n_cv)],dtype=object)
        self.decay_constant_list = -np.diag(self.decay_matrix)

        # 转移
        self.transfer = self.Transfer(reader)

        # 消失
        self.remove = self.Remove(reader)

        # 释放
        self.release = self.Release(reader)

        # 求解器
        self.register = Registry()
        self.matrix_solver = reader.solver_dict['matrix_solver']
        matrix_solver = AccidentSolver(reader,self.register)

        # 初始核素盘存量
        self.inventory_matrix = reader.initial_inventory_matrix
        self.inventory_list = self.inventory_matrix.flatten()

    def generate_completed_accident_matrix(self):
        '''
            生成完整的事故矩阵 (dN/dt = AN + s)
        '''
        # 衰变块
        accident_matrix = block_diag(*([self.decay_matrix]*self.n_cv))
        
        # 转移块
        for cv_idx in range(self.n_cv-1):
            transfer_matrix = self.transfer.transfer_diag_list[cv_idx]
            transfer_pured_matrix = self.transfer.transfer_pured_diag_list[cv_idx]
            add_row_0_idx = (cv_idx+1) * self.n_nuclide
            add_row_1_idx = (cv_idx+2) * self.n_nuclide
            add_col_0_idx = (cv_idx) * self.n_nuclide
            add_col_1_idx = (cv_idx+1) * self.n_nuclide
            minus_row_0_idx = (cv_idx) * self.n_nuclide
            minus_row_1_idx = (cv_idx+1) * self.n_nuclide
            minus_col_0_idx = (cv_idx) * self.n_nuclide
            minus_col_1_idx = (cv_idx+1) * self.n_nuclide
            accident_matrix[add_row_0_idx:add_row_1_idx,add_col_0_idx:add_col_1_idx] += transfer_pured_matrix
            accident_matrix[minus_row_0_idx:minus_row_1_idx,minus_col_0_idx:minus_col_1_idx] -= transfer_matrix
        
        # 去除块
        removal_diag = np.diag(self.remove.remove_vector)
        accident_matrix -= removal_diag

        # 转移矩阵
        self.A = accident_matrix

        # 释放块
        if self.release.pesudo_nuclide_inventory > 0.0:
            accident_matrix = np.hstack([accident_matrix,self.release.pesudo_decay_transfer_vector.reshape(-1,1)])
            bottom_row = np.zeros(self.n_cv*self.n_nuclide+1)
            bottom_row[-1] = -self.release.pesudo_decay_constant
            accident_matrix = np.vstack([accident_matrix,bottom_row])
        
        self.accident_matrix = accident_matrix

    def _burst_update_inventory(self,time,step):
        '''
            爆发模式下更新核素盘存量
        '''
        # 爆发转移量
        transfer_elevate_amount = np.zeros((self.n_cv,self.n_nuclide))
        transfer_decline_amount = np.zeros((self.n_cv,self.n_nuclide))
        for cv_idx in range(self.n_cv-1): # 最后一个控制体不可能有下游了
            for nuclide_idx in range(self.n_nuclide):
                for transfer_index,transfer_time_series in enumerate(self.transfer.transfer_time_series[cv_idx][nuclide_idx]):
                    if transfer_time_series.typ == 'burst':
                        pur_coef = self.transfer.pure_coef_matrix[cv_idx][nuclide_idx][transfer_index]
                        transfer_amount = self.inventory_matrix[cv_idx][nuclide_idx] * transfer_time_series.at(time,step)
                        transfer_decline_amount[cv_idx][nuclide_idx] += transfer_amount
                        transfer_elevate_amount[cv_idx+1][nuclide_idx] += transfer_amount*(1-pur_coef)
                        self.transfer.transfer_rate_detailed_cumulant_matrix[cv_idx][nuclide_idx][transfer_index] += transfer_amount
                        self.transfer.transfer_rate_pured_detailed_cumulant_matrix[cv_idx][nuclide_idx][transfer_index] += transfer_amount*(1-pur_coef)
        
        # 爆发消失量
        remove_decline_amount = np.zeros((self.n_cv,self.n_nuclide))
        for cv_idx in range(self.n_cv):
            for nuclide_idx in range(self.n_nuclide):
                for remove_index,remove_time_series in enumerate(self.remove.remove_time_series[cv_idx][nuclide_idx]):
                    if remove_time_series.typ == 'burst':
                        remove_amount = self.inventory_matrix[cv_idx][nuclide_idx] * remove_time_series.at(time,step)
                        remove_decline_amount[cv_idx][nuclide_idx] += remove_amount
                        self.remove.remove_rate_detailed_cumulant_matrix[cv_idx][nuclide_idx][remove_index] += remove_amount
        # 爆发释放量
        reelase_elevate_amount = np.zeros((self.n_cv,self.n_nuclide))
        for cv_idx in range(self.n_cv):
            for nuclide_idx in range(self.n_nuclide):
                for release_index,release_time_series in enumerate(self.release.release_time_series[cv_idx][nuclide_idx]):
                    if release_time_series.typ == 'burst':
                        release_amount = release_time_series.at(time,step)
                        reelase_elevate_amount[cv_idx][nuclide_idx] += release_amount
                        self.release.release_rate_detailed_cumulant_matrix[cv_idx][nuclide_idx][release_index] += release_amount
        # 更新盘存量
        self.inventory_matrix += transfer_elevate_amount
        self.inventory_matrix -= transfer_decline_amount
        self.inventory_matrix -= remove_decline_amount
        self.inventory_matrix += reelase_elevate_amount

        self.inventory_list = self.inventory_matrix.flatten()

    def update_inventory(self,time,step,time_step):
        '''
            更新核素盘存量
        '''
        if self.burst_process_model == 'InstantBurst':
            self._burst_update_inventory(time,step)
        if self.release.pesudo_nuclide_inventory > 0.0:
            self.inventory_old = np.append(self.inventory_list,self.release.pesudo_nuclide_inventory)
            self.inventory_new = self.register.get('matrix_solver',self.matrix_solver)(self.inventory_old,self.accident_matrix,time_step)
            self.inventory_new = self.inventory_new[0:-1]
            self.inventory_old = self.inventory_old[0:-1]
            self.inventory_list = self.inventory_new
        else:
            self.inventory_old = self.inventory_list
            self.inventory_new = self.register.get('matrix_solver',self.matrix_solver)(self.inventory_old,self.accident_matrix,time_step)
            self.inventory_list = self.inventory_new
        
        self.inventory_matrix = self.inventory_list.reshape(self.n_cv,self.n_nuclide)

    def _examin_zero_col_idx(self,zero_col_idx,vec,label):
        eps = 0.0
        for i in zero_col_idx:
            if i >= len(vec):
                continue
            if abs(vec[i]) > eps:
                raise ValueError(
                    f"列 {i} 被判定为全零列，但 {label}[{i}] = {vec[i]} 非零，数据或逻辑可能有问题"
                ) 
                sys.exit()

    def calculate_integral(self,time_step):
        '''
            计算积分值
        '''
        # 产生量
        release_integral = self.release.release_vector * time_step

        # 去掉全0列,解积分矩阵
        vector = self.inventory_new - self.inventory_old - release_integral
        A = self.A.copy()
        n = A.shape[0]
        zero_col_mask = np.all(A == 0, axis=0)      
        zero_col_idx = np.where(zero_col_mask)[0]   
        process_idx = [i for i in range(n) if not zero_col_mask[i]]
        A_reduced = A[np.ix_(process_idx, process_idx)]
        vector_reduced = vector[process_idx]
        x_reduced = np.linalg.solve(A_reduced, vector_reduced)
        inventory_integral = np.zeros_like(vector)
        inventory_integral[process_idx] = x_reduced

        # 转移从上游的去除量
        flat_transfer_rate_detailed = [lst for cv_block in self.transfer.transfer_rate_detailed_matrix for lst in cv_block]
        transfer_integral = inventory_integral * self.transfer.transfer_rate_vector
        transfer_integral_detailed = [transfer_integral[i]*flat_transfer_rate_detailed[i]/self.transfer.transfer_rate_vector[i] if self.transfer.transfer_rate_vector[i] != 0 else np.zeros_like(flat_transfer_rate_detailed[i]) for i in range((self.n_cv-1)*self.n_nuclide)]
        nrow = self.n_cv - 1
        ncol = self.n_nuclide
        transfer_integral_detailed_matrix = np.array([[transfer_integral_detailed[cv_idx * ncol + nuclide_idx] for nuclide_idx in range(ncol)] for cv_idx in range(nrow)],dtype=object)
        self.transfer.transfer_rate_detailed_cumulant_matrix += transfer_integral_detailed_matrix
        self._examin_zero_col_idx(zero_col_idx,self.transfer.transfer_rate_vector,'transfer_rate_vector')

        # 经过净化后转移到下游的量
        flat_transfer_rate_pured_detailed = [lst for cv_block in self.transfer.transfer_rate_pured_detailed_matrix for lst in cv_block]
        transfer_pured_integral = inventory_integral * self.transfer.transfer_rate_pured_vector
        transfer_pured_integral_detailed = [transfer_pured_integral[i]*flat_transfer_rate_pured_detailed[i]/self.transfer.transfer_rate_pured_vector[i] if self.transfer.transfer_rate_pured_vector[i] != 0 else np.zeros_like(flat_transfer_rate_pured_detailed[i]) for i in range((self.n_cv-1)*self.n_nuclide)]
        nrow = self.n_cv - 1
        ncol = self.n_nuclide
        transfer_pured_integral_detailed_matrix = np.array([[transfer_pured_integral_detailed[cv_idx * ncol + nuclide_idx] for nuclide_idx in range(ncol)] for cv_idx in range(nrow)],dtype=object)
        self.transfer.transfer_rate_pured_detailed_cumulant_matrix += transfer_pured_integral_detailed_matrix
        self._examin_zero_col_idx(zero_col_idx,self.transfer.transfer_rate_pured_vector,'transfer_rate_pured_vector')

        # 消失量
        flat_remove_rate_detailed = [lst for cv_block in self.remove.remove_rate_detailed_matrix for lst in cv_block]
        remove_integral = inventory_integral * self.remove.remove_vector
        remove_integral_detailed = [remove_integral[i]*flat_remove_rate_detailed[i]/self.remove.remove_vector[i] if self.remove.remove_vector[i] != 0 else np.zeros_like(flat_remove_rate_detailed[i]) for i in range(self.n_cv*self.n_nuclide)]
        nrow = self.n_cv
        ncol = self.n_nuclide
        remove_integral_detailed_matrix = np.array([[remove_integral_detailed[cv_idx * ncol + nuclide_idx] for nuclide_idx in range(ncol)] for cv_idx in range(nrow)],dtype=object)
        self.remove.remove_rate_detailed_cumulant_matrix += remove_integral_detailed_matrix
        self._examin_zero_col_idx(zero_col_idx,self.remove.remove_vector,'remove_vector')

        # 衰变量
        for cv_index in range(self.n_cv):
            decay_integral = inventory_integral[cv_index*self.n_nuclide:(cv_index+1)*self.n_nuclide]*self.decay_constant_list
            self.decay_detailed_cumulant_matrix[cv_index] += decay_integral
        self._examin_zero_col_idx(zero_col_idx,list(self.decay_constant_list)*(self.n_cv*self.n_nuclide),'decay_vector')

        # 各途径产生量
        flat_release_rate_detailed = [lst for cv_block in self.release.release_rate_detailed_matrix for lst in cv_block]
        release_integral_detailed = [release_integral[i]*flat_release_rate_detailed[i]/self.release.release_vector[i] if self.release.release_vector[i] != 0 else np.zeros_like(flat_release_rate_detailed[i]) for i in range(self.n_cv*self.n_nuclide)]
        nrow = self.n_cv
        ncol = self.n_nuclide
        release_integral_detailed_matrix = np.array([[release_integral_detailed[cv_idx * ncol + nuclide_idx] for nuclide_idx in range(ncol)] for cv_idx in range(nrow)],dtype=object)
        self.release.release_rate_detailed_cumulant_matrix += release_integral_detailed_matrix

        # 爆发产生量在_burst_update_inventory已然更新