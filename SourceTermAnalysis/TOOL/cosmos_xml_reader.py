import xml.etree.ElementTree as ET
import numpy as np
import re
import sys

# 常数
NA = 6.02214076E+23
# 内部计算单位
TIME = '[s]'

# 量纲转化字典
conversion_dict = {
        'time': {
            '[a]': 365 * 24 * 3600,
            '[d]': 24 * 3600,
            '[h]': 3600,
            '[min]': 60,
            '[s]': 1,
        },
        'length': {
            '[m]': 1E+2,
            '[mm]': 1E-1,
            '[cm]': 1,
            '[nm]': 1E-7,
            '[um]': 1E-4,
            '[km]': 1E+5,
        },
        'volume': {
            '[m3]': 1,
            '[cm3]': 1E-6,
            '[mm3]': 1E-9,
            '[L]': 1E-3,
            '[um3]': 1E-18,
        },
        'temperature': {
            '[C]': 1,
            '[K]': 273.15,
        },
        'mass': {
                '[g]': 1,
                '[kg]': 1e3,
                '[t]': 1e6,
            },
        'moleculus':{
            '[n]':1,
            '[mol]':NA
        },
        'pressure':{
                '[Pa]': 1,
                '[kPa]': 1E+3,
                '[MPa]': 1E+6,
                '[atm]': 1.015E+5,
                '[bar]': 1e5
        },
        'activity':{
            '[atom]':1,
            '[Bq]':None
        },
        'power':{
            '[W]':1,
            '[kW]':1e3,
            '[MW]':1e6,
            '[GW]':1e9,
        }
    } 

class Preprocessor:
    '''
        预处理数据、进行单位转化、补全 
    '''
    def __init__(self):
        pass

    def _splite_unit(self,input_string):
        '''
            用正则表达式匹配 [a] [b] 等模式,识别量纲处于分子或是分母
        '''
        pattern = r'\[([^\]]+)\]'
        if '/' in input_string: 
            before_slash, after_slash = input_string.split('/', 1)
            before_elements = re.findall(pattern, before_slash)
            after_elements = re.findall(pattern, after_slash)
        else:
            before_elements = re.findall(pattern, input_string)
            after_elements = []
        for i,unit in enumerate(before_elements):
            before_elements[i] = '['+unit+']'
        for i,unit in enumerate(after_elements):
            after_elements[i] = '['+unit+']'
        
        return before_elements, after_elements

    def unit_convert(self,value_raw,unit_raw,unit_target,decay_constant=None):
        '''
            根据输入的单位和目标单位进行转换
            :value_raw,unit_raw:值和单位
            :target_unit: 目标单位
            :return: 转换后的值和目标单位
        '''
        if unit_raw == unit_target or value_raw is None:
            return value_raw, unit_target

        if decay_constant is not None:
            conversion_dict['activity']['[Bq]'] = 1 / decay_constant
            
        value_convert = np.array(value_raw) if isinstance(value_raw, list) else value_raw
        unit_target_numerator_list,unit_target_denominator_list = self._splite_unit(unit_target)
        unit_raw_numerator_list,unit_raw_denominator_list = self._splite_unit(unit_raw)

        # 分子量纲的转化
        for i,unit_target_numerator in enumerate(unit_target_numerator_list):
            if unit_target_numerator in conversion_dict['temperature'].keys():
                if len(unit_target_numerator_list) == 1 and len(unit_target_denominator_list) == 0:
                    value_convert = value_convert - 273.15 if unit_raw_numerator_list[0] == '[K]' else value_convert + 273.15
                    return value_convert,unit_target
                else:
                    continue
            for dimension in conversion_dict.keys():
                if unit_target_numerator in conversion_dict[dimension].keys():
                    value_base = value_convert * conversion_dict[dimension][unit_raw_numerator_list[i]]
                    value_convert = value_base / conversion_dict[dimension][unit_target_numerator]
        
        # 分母量纲的转化
        for i,unit_target_denominator in enumerate(unit_target_denominator_list):
            if unit_target_denominator in conversion_dict['temperature'].keys():
                continue
            for dimension in conversion_dict.keys():
                if unit_target_denominator in conversion_dict[dimension]:
                    value_base = value_convert / conversion_dict[dimension][unit_raw_denominator_list[i]]
                    value_convert = value_base * conversion_dict[dimension][unit_target_denominator]
        
        return value_convert,unit_target
    
    def extract_decay_constant(self,nuc,file_path='SourceTermAnalysis/TOOL/DecayLib.dat'):
        '''
            提取衰变常数 /[s]
        '''
        lamda = 0.0
        with open(file_path, "r") as file:
            lines = iter(file)
            for line in lines:
                if nuc in line:
                    key_line = line
                    lamda = float(key_line[24:40].strip())
                    break
        return lamda
    
    def _generate_decay_matrix(self,decay_chain_raw,decay_constant_dict,nuclides):
        '''
            生成衰变率矩阵
        '''
        n = len(nuclides)
        idx_of = {nuc: i for i, nuc in enumerate(nuclides)}
        decay_matrix = np.zeros((n, n), dtype=float)

        # 写入对角线
        for nuc, i in idx_of.items():
            decay_matrix[i, i] = -decay_constant_dict[nuc]

        # 非对角线
        for chain in decay_chain_raw:
            chain_nuclides = chain['nuclide']
            chain_branch    = chain['branch']

            for k, parent in enumerate(chain_nuclides[:-1]): 
                child = chain_nuclides[k + 1]
                branch_coeff = chain_branch[k]
                i_parent = idx_of[parent]
                i_child  = idx_of[child]
                lambda_parent = decay_constant_dict[parent]
                transfer_rate = lambda_parent * branch_coeff
                decay_matrix[i_child, i_parent] += transfer_rate

        return decay_matrix

    def _generate_decay_chain(self,decay_chain,decay_constant_dict):
        '''
            生成完整衰变链
        '''
        normalized_chains = []
        for ch in decay_chain:
            ch_copy = dict(ch)  
            params = {}
            
            if 'branch' in ch_copy and 'params' not in ch_copy:
                params['branch'] = list(ch_copy['branch'])
            if 'params' in ch_copy:
                params.update(ch_copy['params'])
            ch_copy['params'] = params
            normalized_chains.append(ch_copy)

        all_param_names = set()
        for ch in normalized_chains:
            all_param_names.update(ch['params'].keys())

        total_nuclides = {}
        for chain in normalized_chains:
            nuclides = chain['nuclide']
            for nuc in nuclides:
                if nuc not in total_nuclides:
                    total_nuclides[nuc] = {
                        'up': [], 'down': [], 'down_params': {p: [] for p in all_param_names}
                    }

        for ch in normalized_chains:
            nuclides = ch['nuclide']
            params = ch['params']
            n = len(nuclides)
            for i in range(n):
                nuc_up = nuclides[i - 1] if i != 0 else None
                nuc_down = nuclides[i + 1] if i != n - 1 else None
                nuc_loc = nuclides[i]

                if nuc_up is not None:
                    if nuc_up not in total_nuclides[nuc_loc]['up']:
                        total_nuclides[nuc_loc]['up'].append(nuc_up)

                if nuc_down is not None:
                    if nuc_down not in total_nuclides[nuc_loc]['down']:
                        total_nuclides[nuc_loc]['down'].append(nuc_down)
                    for p in all_param_names:
                        val_list = params.get(p, [])
                        val = None
                        if i < len(val_list):
                            val = val_list[i]
                        total_nuclides[nuc_loc]['down_params'][p].append(val)

        def __generate_decay_chain_rec(d, node, parent_params=None):
        
            chain = {'decay_constant': decay_constant_dict.get(node, None)}
            for p in all_param_names:
                chain[p] = parent_params.get(p) if parent_params is not None else None

            if not d[node]['down']:
                return chain

            down_list = d[node]['down']
            down_params = d[node].get('down_params', {p: [] for p in all_param_names})

            for i, child in enumerate(down_list):
                child_parent_params = {}
                for p in all_param_names:
                    plist = down_params.get(p, [])
                    child_parent_params[p] = plist[i] if i < len(plist) else None
                chain[child] = __generate_decay_chain_rec(d, child, parent_params=child_parent_params)

            return chain

        def __build_total_decay_dict(d):
            starting_nodes = [node for node, relations in d.items() if not relations['up']]
            total_dict = {}
            for node in starting_nodes:
                total_dict[node] = __generate_decay_chain_rec(d, node, parent_params=None)
            return total_dict

        completed_chain = __build_total_decay_dict(total_nuclides)
        
        return completed_chain

class CosmosXMLReader:
    '''
        读取COSMOS XML文件,提取相关信息
    '''
    def __init__(self):
        self.preprocessor = Preprocessor()
        
    def _read_data_from_xml(self,location,parent,element_or_label_name,type,element_label_name=None,default=False,examine_default=False):
        '''
            从element区读取或从label读取
        '''
        default_swtich = False
        if parent is None:
            default_swtich = True
        else:
            if location == 'label':
                data = parent.get(element_or_label_name)
                default_swtich = False if data not in ("", None) else True
            elif location == 'element_label':
                element = parent.find(element_or_label_name)
                if element is None:
                    default_swtich = True
                else:
                    data = element.get(element_label_name)
                    default_swtich = False if data not in ("", None) else True
        if default_swtich == True: 
            if default is False:
                print(f'{element_or_label_name} incomplete')
                sys.exit()
            else:
                if examine_default:
                    return default,True
                else:
                    return default
                
        data = data.strip()
        # 根据目标类型做转换
        if type == 'str':
            value = data
        elif type == 'str-array':
            value = data.split()
        elif type == 'float':
            value = float(data)
        elif type == 'int':
            value = int(float(data))
        elif type == 'float-array':
            value = [float(x) for x in data.split()]
        if examine_default:
            return value,False
        else:
            return value

    def _read_option(self,parent,option_name,default=False):
        '''
            读取字符串选项模型
        '''
        option = self._read_data_from_xml('label',parent,option_name,'str',default=default)
        
        return option
    
    def _read_element_option(self,parent,element_name,option_name,default=False,data_type='str'):
        '''
            读取元素内字符串选项模型
        '''
        option = self._read_data_from_xml('element_label',parent,element_name,'str',element_label_name=option_name,default=default)
        if isinstance(option,str) and data_type != 'str':
            if data_type == 'float':
                option = float(option)
            elif data_type == 'int':
                option = int(option)
            elif data_type == 'float-array':
                option = [float(x) for x in option.split()]
            elif data_type == 'int-array':
                option = [int(x) for x in option.split()]
            elif data_type == 'str-array':
                option = option.split()
        
        return option
    
    def _read_element_option_value_unit_pair(self,parent,element_name,default_unit,target_unit,data_type,default_value=False,default_time=False,decay_constant=None):
        '''
            读取值与单位的对,并进行单位转化
        '''
        if data_type == 'float' or data_type == 'float-array' or data_type == 'int':

            value_raw,is_default = self._read_data_from_xml('element_label',parent,element_name,data_type,'value',default=default_value,examine_default=True)
            unit_raw = self._read_data_from_xml('element_label',parent,element_name,'str','unit',default=default_unit)
            if is_default:
                unit_raw = default_unit
            value,unit = self.preprocessor.unit_convert(value_raw,unit_raw,target_unit,decay_constant)

            return value
               
        elif data_type == 'time_series':

            data_type = 'float-array'
            time_raw = self._read_data_from_xml('element_label',parent,element_name,data_type,'time',default=default_time)
            time_raw_unit = self._read_data_from_xml('element_label',parent,element_name,'str','time_unit',default='[s]') 
            time,time_unit = self.preprocessor.unit_convert(time_raw,time_raw_unit,TIME)
            value_raw,is_default = self._read_data_from_xml('element_label',parent,element_name,data_type,'value',default=default_value,examine_default=True)
            unit_raw = self._read_data_from_xml('element_label',parent,element_name,'str','unit',default=default_unit)
            if is_default:
                unit_raw = default_unit
            value,unit = self.preprocessor.unit_convert(value_raw,unit_raw,target_unit,decay_constant)

            return value,time
    
    def _read_option_value_unit_pair(self,parent,default_unit,target_unit,data_type,default_value=False,default_time=False,decay_constant=None):
        '''
            读取值与单位的对,并进行单位转化
        '''
        if data_type == 'float' or data_type == 'float-array' or data_type == 'int':

            value_raw,is_default = self._read_data_from_xml('label',parent,'value',data_type,default=default_value,examine_default=True)
            unit_raw = self._read_data_from_xml('label',parent,'unit','str',default=default_unit)
            if is_default:
                unit_raw = default_unit
            value,unit = self.preprocessor.unit_convert(value_raw,unit_raw,target_unit,decay_constant)

            return value
               
        elif data_type == 'time_series':

            data_type = 'float-array'
            time_raw = self._read_data_from_xml('label',parent,'time',data_type,default=default_time)
            time_raw_unit = self._read_data_from_xml('label',parent,'time_unit','str',default='[s]') 
            time,time_unit = self.preprocessor.unit_convert(time_raw,time_raw_unit,TIME)
            value_raw,is_default = self._read_data_from_xml('label',parent,'value',data_type,default=default_value,examine_default=True)
            unit_raw = self._read_data_from_xml('label',parent,'unit','str',default=default_unit)
            if is_default:
                unit_raw = default_unit
            value,unit = self.preprocessor.unit_convert(value_raw,unit_raw,target_unit,decay_constant)

            return value,time
    
    def _read_decay_chain(self,decay_chain):
        '''
            读取衰变链
        '''
        nuclide = self._read_option(decay_chain,'chain').split('->')
        branch = self._read_option(decay_chain,'branch').split()
        branch = [None if value == 'None' else float(value) for value in branch]
        chain_dict = {
            'nuclide':nuclide,
            'branch':branch
        }
        return chain_dict
    
         