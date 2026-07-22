from TOOL.cosmos_xml_reader import CosmosXMLReader,Preprocessor
from TOOL.cosmos_general_function import round_sig
import xml.etree.ElementTree as ET
import numpy as np
import copy
import re

TIME = '[s]'
ACTIVITY = '[atom]'

class AccidentXMLReader(CosmosXMLReader):
    '''
        读取ACCIDENT XML文件,提取相关信息
    '''
    class Preprocessor(Preprocessor):
        '''
            ACCIDENT的数据预处理对象
        '''
        @classmethod
        def generate_time_sequence_raw(cls,time_spans,time_steps):
            '''
                生成模拟的原始时间序列
            '''
            time = 0.0
            finish_time = 0.0
            time_list = [0.0]
            time_step_list = []
            for phase,time_span in enumerate(time_spans):
                time_step = round_sig(time_steps[phase])
                finish_time = round_sig(finish_time+time_span)
                while time < finish_time:
                    time = round_sig(time+time_step)            
                    time_list.append(time)
                    time_step_list.append(time_step)

            return time_list,time_step_list

        @classmethod
        def _insert_burst_time_point(cls,burst_time_list,time_step_raw_list,time_raw_list,tol=1e-12):
            '''
                插入爆发时间点
            '''
            time_list = list(time_raw_list)
            time_step_list = list(time_step_raw_list)

            burst_times = sorted(burst_time_list)

            def insert_single_burst(t_burst):
                # 精度格式化
                t_burst = round_sig(t_burst)

                # 超出时间范围
                if t_burst < time_list[0] - tol or t_burst > time_list[-1] + tol:
                    return
                # 已经存在此爆发时刻
                for t in time_list:
                    if abs(t - t_burst) <= tol:
                        return
                # 寻找爆发时刻在哪个时间步
                idx = None
                for i in range(len(time_list) - 1):
                    t_left = time_list[i]
                    t_right = time_list[i + 1]
                    if t_left < t_burst < t_right:
                        idx = i
                        break

                # 拆分时间步
                t_left = time_list[idx]
                t_right = time_list[idx + 1]
                dt1 = round_sig(t_burst - t_left)      
                dt2 = round_sig(t_right - t_burst)     

                time_list.insert(idx + 1, t_burst)
                time_step_list[idx] = dt1
                time_step_list.insert(idx + 1, dt2)

            for t_burst in burst_times:
                insert_single_burst(t_burst)

            return time_list,time_step_list

        @classmethod    
        def _step_function_expansion(cls,f_raw,x_raw):
            '''
                阶梯函数展开
            '''        
            x_list = [0.0]
            f_list = [0.0]     

            for (a, b), v in zip(x_raw, f_raw):
                x_list.extend([a, a, b, b])
                f_list.extend([0.0, v, v, 0.0])
            x_list.append(1e40)
            f_list.append(0.0)
            return f_list,x_list

        @classmethod
        def _calculate_time_span_average_remove_rate(cls,value_raw,time_span):
            '''
                一段时间的平均去除份额
                N = N0*exp(-\lambda*t)
                N0*(1-value) = N0*exp(-\lambda*t)
                \lambda = -ln(1-valve)/t
            '''
            if isinstance(value_raw,(list,np.ndarray)):
                value = [- np.log(1-v) / (t[1]-t[0]) for v,t in zip(value_raw,time_span)]
            elif isinstance(value_raw,float):
                value = - np.log(1-value_raw) / (time_span[1]-time_span[0])
            
            return value

        @classmethod
        def _calculate_time_span_average_release_rate(cls,value_raw,time_span):
            '''
                一段时间的平均去除份额
                N = N0 + R*t
                R = (N-N0)/t
            '''
            if isinstance(value_raw,(list,np.ndarray)):
                value = [v / (t[1]-t[0]) for v,t in zip(value_raw,time_span)]
            elif isinstance(value_raw,float):
                value = value_raw / (time_span[1]-time_span[0])

            return value

        @classmethod
        def _convert_interpolation_table_to_interpolation_table(cls,value_raw_list,time_raw_list):
            '''
                转换插值表为插值表
            '''
            value_list = value_raw_list
            time_list = time_raw_list

            return value_list,time_list
        
        @classmethod
        def _convert_constant_rate_interval_to_interpolation_table(cls,value_raw_list,time_raw_list):
            '''
                转换恒定速率区间为插值表
            '''
            value_list,time_list = cls._step_function_expansion(value_raw_list,time_raw_list)
    
            return value_list,time_list
        
        @classmethod
        def _convert_interval_remove_quantity_to_interpolation_table(cls,value_raw_list,time_raw_list):
            '''
                转换一段时间的去除总量为该区段的平均去除率,以插值表形式体现
            '''
            # 首先将区间去除量转化为此区间的平均去除速率
            value_raw_list_ = cls._calculate_time_span_average_remove_rate(value_raw_list,time_raw_list)
            # 再展开为阶梯函数
            value_list,time_list = cls._step_function_expansion(value_raw_list_,time_raw_list)

            return value_list,time_list

        @classmethod
        def _convert_interval_release_quantity_to_interpolation_table(cls,value_raw_list,time_raw_list):
            '''
                转换一段时间的释放总量为该区段的平均释放率,以插值表形式体现
            '''
            # 首先将区间释放量转化为此区间的平均释放速率
            value_raw_list_ = cls._calculate_time_span_average_release_rate(value_raw_list,time_raw_list)
            # 再展开为阶梯函数
            value_list,time_list = cls._step_function_expansion(value_raw_list_,time_raw_list)

            return value_list,time_list

        @classmethod
        def _convert_burst_remove_quantity(cls,value_raw_list,time_raw_list,burst_time_step_number,burst_process_model,simulation_time_list:list,simulation_time_step_list:list):
            '''
                处理爆发去除事件
            '''
            value_list = []
            step_list = []
            for v_raw,t_raw in zip(value_raw_list,time_raw_list):
                t_raw = round_sig(t_raw)
                index = simulation_time_list.index(t_raw)
                if burst_process_model == 'StepBurst':
                    t_burst_span = sum(simulation_time_step_list[index:index+burst_time_step_number])
                    v = cls._calculate_time_span_average_remove_rate(v_raw,(0.0,t_burst_span))
                    value_list.append(v)
                    step_list.append(list(range(index, index + burst_time_step_number)))
                elif burst_process_model == 'InstantBurst':
                    value_list.append(v_raw)
                    step_list.append([index])
            
            return value_list,step_list
        
        @classmethod
        def _convert_burst_release_quantity(cls,value_raw_list,time_raw_list,burst_time_step_number,burst_process_model,simulation_time_list:list,simulation_time_step_list:list):
            '''
                处理爆发释放事件
            '''
            value_list = []
            step_list = []
            for v_raw,t_raw in zip(value_raw_list,time_raw_list):
                t_raw = round_sig(t_raw)
                index = simulation_time_list.index(t_raw)
                if burst_process_model == 'StepBurst':
                    t_burst_span = sum(simulation_time_step_list[index:index+burst_time_step_number])
                    v = cls._calculate_time_span_average_release_rate(v_raw,(0.0,t_burst_span))
                    value_list.append(v)
                    step_list.append(list(range(index, index + burst_time_step_number)))
                elif burst_process_model == 'InstantBurst':
                    value_list.append(v_raw)
                    step_list.append([index])
            
            return value_list,step_list

        @classmethod
        def convert_release_or_remove_data(cls,data_type,value_raw_list,time_raw_list,burst_time_step_number,burst_process_model,simulation_time_list:list,simulation_time_step_list:list):
            '''
                转换释放/去除数据的接口
            '''
            convert_dict = {
                'interpolation_table':cls._convert_interpolation_table_to_interpolation_table,
                'constant_rate_interval':cls._convert_constant_rate_interval_to_interpolation_table,
                'interval_remove_quantity':cls._convert_interval_remove_quantity_to_interpolation_table,
                'interval_release_quantity':cls._convert_interval_release_quantity_to_interpolation_table,
                'burst_remove_quantity':cls._convert_burst_remove_quantity,
                'burst_release_quantity':cls._convert_burst_release_quantity
            }
            if 'burst' in data_type:
                value_list,step_list = convert_dict[data_type](value_raw_list,time_raw_list,burst_time_step_number,burst_process_model,simulation_time_list,simulation_time_step_list)
                return value_list,step_list,'burst'
            else:
                value_list,time_list = convert_dict[data_type](value_raw_list,time_raw_list)
                return value_list,time_list,'rate'
        
        @classmethod
        def _extract_time_span(cls,s):        
            '''
                提取时间段
            '''
            pairs = re.findall(r'\[([^\s\]]+)\s+([^\s\]]+)\]', s)
            result = [(float(a), float(b)) for a, b in pairs]
        
            return result  

    def __init__(self,xml_path,file_path_decay_constant_default = 'SourceTermAnalysis/TOOL/DecayLib.dat'):
        
        super().__init__()
        # 预处理类对象
        self.preprocessor = self.Preprocessor()
        # 输入文件路径
        self.xml_path = xml_path
        # 物性参数路径
        self.file_path_decay_constant_default = file_path_decay_constant_default
        # XML输入
        self.root = ET.parse(self.xml_path).getroot()
        # 独立进行Accident计算
        self.accident_independent_configuration = self.root.find('Accident-Accident')

        # 初始化
        # 模拟控制(时间，时间步,输出)
        self.control_dict = {
            'time':None,
            'time_step':None,
            'output_path':None,
            'write_interval':None,
            'integral_swtich':None
        }
        # 求解方式与模型(爆发释放处理模式,时间步模式爆发步数量,伪核素法残差)
        self.solver_dict = {
            'matrix_solver':None,
            'burst_process_model':None,
            'burst_time_step_number':None,
            'pseudo_nuclide_residual':None
        }
        # 核素数量
        self.n_nuclide = len(self.accident_independent_configuration.find('./Radioactivity/DecayConstant').get('required_nuclides').split())
        # 控制体信息(标签,转移率,消失率,释放率)
        self.n_cv = len(self.accident_independent_configuration.findall('Cv'))
        self.transfer_info = [[[] for _ in range(self.n_nuclide)] for _ in range(self.n_cv)]
        self.remove_info = [[[] for _ in range(self.n_nuclide)] for _ in range(self.n_cv)]
        self.release_info = [[[] for _ in range(self.n_nuclide)] for _ in range(self.n_cv)]
        self.initial_inventory_matrix = np.zeros((self.n_cv,self.n_nuclide))
        self.cv_labels = [None for index in range(self.n_cv)]

        # 默认值
        self.control_dict_default = {
            'time_unit':'[s]',
            'time_step_unit':'[s]',
            'write_interval':[0],
            'write_interval_unit':'[s]',
            'output_path':'OUTPUT/ACCIDENTOUTPUT',
            'integral_swtich':'off'
        }
        self.solver_dict_default = {
            'matrix_solver':'MMPA32',
            'burst_process_model':'StepBurst',
            'burst_time_step_number':2,
            'pseudo_nuclide_residual':1e-10
        }
        self.initial_inventory_default = 0.0
        self.inventory_unit_default = '[Bq]'
        self.cv_labels_default = [f'Cv{index}' for index in range(self.n_cv)]
        self.time_unit_default = '[s]'
        self.pur_coef_list_default = np.zeros(self.n_nuclide,dtype=float)

    def read_configuration(self):
        '''
            读取配置信息
        '''
        self._read_independent_configuration()

    def _read_independent_configuration(self):
        '''
            读取独立进行Accident计算的配置信息
        '''
        # 读取模拟时间/输出
        self._read_control()
        
        # 生成模拟时间序列
        self._generate_simulation_time_sequence()

        # 读取求解模型
        self._read_solver()

        # 读取放射性
        self._read_radioactivity()

        # 读取全部控制体
        self._read_cvs()
 
    def _read_control(self):
        '''
            读取模拟时间与输出
        '''
        control = self.accident_independent_configuration.find('Control')
        # 模拟时间
        parent = control
        element_name = 'RequiredSimulationTime'
        default_unit = self.control_dict_default['time_unit']  
        target_unit = TIME                                     
        data_type = 'float-array'
        default_value = False
        self.control_dict['time'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        # 模拟时间步
        parent = control
        element_name = 'RequiredTimeStep'
        default_unit = self.control_dict_default['time_step_unit']  
        target_unit = TIME                                     
        data_type = 'float-array'
        default_value = False
        self.control_dict['time_step'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        # 输出
        output = control.find('Output')
        # 输出路径
        parent = output
        element_name = 'OutputPath'
        option_name = 'value'
        default_value = self.control_dict_default['output_path']                                       
        self.control_dict['output_path'] = self._read_element_option(parent,element_name,option_name,default=default_value)

        # 输出间隔
        parent = output
        element_name = 'WriteInterval'
        default_unit = self.control_dict_default['write_interval_unit']  
        target_unit = TIME                                     
        data_type = 'float-array'
        default_value = self.control_dict_default['write_interval']  
        write_interval = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        self.control_dict['write_interval'] = [-1 for _ in self.control_dict['time']] if 0 in write_interval else write_interval

        # 输出积分值
        parent = output
        element_name = 'Integral'
        option_name = 'value'                                        
        default_value = self.control_dict_default['integral_swtich']  
        self.control_dict['integral_swtich'] = self._read_element_option(parent,element_name,option_name,default=default_value)
    
    def _read_solver(self):
        '''
            读取求解模型
        '''
        solver = self.accident_independent_configuration.find('Solver')
        # 矩阵求解器
        parent = solver
        element_name = 'MatrixSolver'                                   
        default_value = self.solver_dict_default['matrix_solver']
        self.solver_dict['matrix_solver'] = self._read_element_option(parent,element_name,'value',default=default_value)
        
        # 爆发过程处理模型
        parent = solver
        element_name = 'BurstProcessModel'                                   
        default_value = self.solver_dict_default['burst_process_model']
        self.solver_dict['burst_process_model'] = self._read_element_option(parent,element_name,'value',default=default_value)
        self.burst_process_model = self.solver_dict['burst_process_model']

        # 爆发释放模型为分解为多个时间步释放时所占据的时间步数量
        parent = solver
        element_name = 'BurstTimeStepNumber'                                   
        data_type = 'int'
        default_value = self.solver_dict_default['burst_time_step_number']
        self.solver_dict['burst_time_step_number'] = self._read_element_option(parent,element_name,'value',default=default_value,data_type=data_type)
        self.burst_time_step_number = self.solver_dict['burst_time_step_number']

        # 伪核素的残差
        parent = solver
        element_name = 'PseudoNuclideResidual'                                   
        data_type = 'float'
        default_value = self.solver_dict_default['pseudo_nuclide_residual']
        self.solver_dict['pseudo_nuclide_residual'] = self._read_element_option(parent,element_name,'value',default=default_value,data_type=data_type)

    def _read_radioactivity(self):
        '''
            读取放射性
        '''
        radioactivity = self.accident_independent_configuration.find('Radioactivity')
        # 核素列表
        self.nuclides = self._read_element_option(radioactivity,'DecayConstant','required_nuclides').split()
        
        # 衰变常数
        decay_constant_raw = self._read_element_option(radioactivity,'DecayConstant','decay_constant',default=None)
        if decay_constant_raw is not None:
            decay_constant = [float(x) for x in decay_constant_raw.split()]
        else:
            decay_constant = [self.preprocessor.extract_decay_constant(nuclide) for nuclide in self.nuclides]
        decay_constant_dict = {}
        for i,nuc in enumerate(self.nuclides):
            decay_constant_dict[nuc] = float(decay_constant[i])
        self.decay_constant_dict = decay_constant_dict

        # 读取衰变链
        chains = radioactivity.findall('DecayChain')
        decay_chain_raw = [self._read_decay_chain(chain) for chain in chains]   
        
        # 生成衰变矩阵
        self.decay_matirx = self.preprocessor._generate_decay_matrix(decay_chain_raw,decay_constant_dict,self.nuclides)

        # 生成完整衰变链
        self.decay_chain = self.preprocessor._generate_decay_chain(decay_chain_raw,decay_constant_dict)

    def _read_cvs(self):
        '''
            读取全部控制体
        '''
        cv_list = self.accident_independent_configuration.findall('Cv')
        for cv in cv_list:

            # 读取控制体的基本信息
            cv_idx = int(self._read_option(cv,'required_index'))
            cv_label = self._read_option(cv,'label',default=self.cv_labels_default[cv_idx])
            self.cv_labels[cv_idx] = cv_label

            # 读取控制体的转移途径
            transfers = cv.find('Transfers')
            if transfers is not None:
                transfer_list = transfers.findall('Transfer')
                for transfer_idx,transfer in enumerate(transfer_list):
                    transfer_label = self._read_option(transfer,'label',default=self.cv_labels_default[cv_idx]+f'_transfer{transfer_idx}')
                    value_list,time_or_step_list,typ = self._read_release_or_remove(transfer,'remove')
                    pur_coef_list = copy.deepcopy(self.pur_coef_list_default)
                    pur_coef_raw = [float(x) for x in self._read_option(transfer,'pur_coef',default='').split()]
                    pur_nuc_raw = self._read_option(transfer,'pur_nuc',default='').split()
                    for pur_nuc,pur_coef in zip(pur_nuc_raw,pur_coef_raw):
                        index = self.nuclides.index(pur_nuc)
                        pur_coef_list[index] = pur_coef
                    for nuclide_idx,pur_coef in enumerate(pur_coef_list):
                        transfer_rate = value_list
                        self.transfer_info[cv_idx][nuclide_idx].append({
                            'label':transfer_label+f'_{self.nuclides[nuclide_idx]}',
                            'type':typ,
                            'value':transfer_rate,
                            'time_or_step':time_or_step_list,
                            'pur_coef':pur_coef
                        })

            # 读取控制体的核素信息
            nucldies = cv.find('Nuclides')
            if nucldies is not None:
                nuclide_list = nucldies.findall('Nuclide')
                for nuclide in nuclide_list:
                    nuclide_name = self._read_option(nuclide,'value')
                    nuclide_idx = self.nuclides.index(nuclide_name)

                    # 初始库存
                    initial_inventory_raw = float(self._read_option(nuclide,'initial_inventory',default=self.initial_inventory_default))
                    initial_inventory_unit = self._read_option(nuclide,'inventory_unit',default=self.inventory_unit_default)
                    initial_inventory,_ = self.preprocessor.unit_convert(initial_inventory_raw,initial_inventory_unit,ACTIVITY,decay_constant=self.decay_constant_dict[nuclide_name])
                    self.initial_inventory_matrix[cv_idx][nuclide_idx] = initial_inventory

                    # 释放
                    release_list = nuclide.findall('Release')
                    for release_index,release in enumerate(release_list):
                        release_label = self._read_option(release,'label',default=self.cv_labels_default[cv_idx]+f'_{nuclide_name}_release{release_index}')
                        value_list,time_or_step_list,typ = self._read_release_or_remove(release,'release',decay_constant=self.decay_constant_dict[nuclide_name])
                        self.release_info[cv_idx][nuclide_idx].append({
                            'label':release_label,
                            'type':typ,
                            'value':value_list,
                            'time_or_step':time_or_step_list
                        })

                    # 去除
                    remove_list = nuclide.findall('Remove')
                    for remove_index,remove in enumerate(remove_list):
                        remove_label = self._read_option(remove,'label',default=self.cv_labels_default[cv_idx]+f'_{nuclide_name}_remove{remove_index}')
                        value_list,time_or_step_list,typ = self._read_release_or_remove(remove,'remove')
                        self.remove_info[cv_idx][nuclide_idx].append({
                            'label':remove_label,
                            'type':typ,
                            'value':value_list,
                            'time_or_step':time_or_step_list
                        })

    def _read_release_or_remove(self,element,release_or_remove,decay_constant=None):
        '''
            读取释放,消失,转移
        '''
        # 读取原始字符串
        unit = self._read_option(element,'unit',default=None)
        value_raw_list = [float(x) for x in self._read_option(element,'value').split()]
        time_str = self._read_option(element,'time')
        time_unit = self._read_option(element,'time_unit')
        # 将时间段展开,并转换时间单位
        time_raw_list = self.preprocessor._extract_time_span(time_str) if '[' in time_str else [float(x) for x in time_str.split()]
        time_raw_list,_ = self.preprocessor.unit_convert(time_raw_list,time_unit,TIME)
        # 判断类型
        if unit is None and '[' in time_str:
            data_type = 'interval_remove_quantity'
        elif unit is None and '[' not in time_str:
            data_type = 'burst_remove_quantity'
        elif '/' in unit and '[' not in time_str:
            data_type = 'interpolation_table' 
        elif '/' in unit and '[' in time_str:
            data_type = 'constant_rate_interval'
        elif '/' not in unit and '[' in time_str and release_or_remove == 'release':
            data_type = 'interval_release_quantity'
        elif '/' not in unit and '[' not in time_str:
            data_type = 'burst_release_quantity'
        # 转换数据单位
        if data_type in ['interpolation_table','constant_rate_interval']:
            value_raw_list,_ = self.preprocessor.unit_convert(value_raw_list,unit,ACTIVITY+'/'+TIME,decay_constant=decay_constant) if release_or_remove == 'release' else self.preprocessor.unit_convert(value_raw_list,unit,'/'+TIME)
        elif data_type in ['interval_release_quantity','burst_release_quantity']:
            value_raw_list,_ = self.preprocessor.unit_convert(value_raw_list,unit,ACTIVITY,decay_constant=decay_constant)
        # 转换数据
        value_list,time_or_step_list,typ = self.preprocessor.convert_release_or_remove_data(data_type,value_raw_list,time_raw_list,self.burst_time_step_number,self.burst_process_model,self.simulation_time_list,self.simulation_time_step_list)

        return np.array(value_list),time_or_step_list,typ
    
    def _generate_simulation_time_sequence(self):
        '''
            产生模拟时间序列(原始时间序列+爆发时间点)
        '''
        # 生成原始的时间与时间步列表
        time_raw_list,time_step_raw_list = self.preprocessor.generate_time_sequence_raw(self.control_dict['time'],self.control_dict['time_step'])
        
        def detect_burst_time_point(element):
            # 检测爆发时间点
            unit = self._read_option(element,'unit',default=None)
            time_str = self._read_option(element,'time')
            if '[' not in time_str and (unit is None or '/' not in unit):
                time_unit = self._read_option(element,'time_unit',default=self.time_unit_default)
                # 将时间段展开,并转换时间单位
                burst_times = [float(x) for x in time_str.split()]
                burst_times, _ = self.preprocessor.unit_convert(burst_times, time_unit, TIME)
                return list(burst_times)
            else:
                return []
            
        # 提取爆发时间点
        burst_time_list = []
        cv_list = self.accident_independent_configuration.findall('Cv')
        for cv in cv_list:
            # 爆发转移时刻
            transfers = cv.find('Transfers')
            if transfers is not None:
                transfer_list = transfers.findall('Transfer')
                for transfer in transfer_list:
                    burst_times = detect_burst_time_point(transfer)
                    burst_time_list += burst_times
            nucldies = cv.find('Nuclides')
            if nucldies is not None:
                nuclide_list = nucldies.findall('Nuclide')
                for nuclide in nuclide_list:
                    release_list = nuclide.findall('Release')
                    remove_list = nuclide.findall('Remove')
                    # 爆发释放时刻
                    for release in release_list:
                        burst_times = detect_burst_time_point(release)
                        burst_time_list += burst_times
                    for remove in remove_list:
                        burst_times = detect_burst_time_point(remove)
                        burst_time_list += burst_times

        self.simulation_time_list,self.simulation_time_step_list = self.preprocessor._insert_burst_time_point(burst_time_list,time_step_raw_list,time_raw_list)
        
