from TOOL.cosmos_xml_reader import CosmosXMLReader,Preprocessor
from scipy.interpolate import RegularGridInterpolator
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sys
import random
import copy
import re

TIME = '[s]'
MASS = '[g]'
POWER = '[GW]'
LENGTH = '[m]'
TEMPERATURE = '[K]'
AREA = '[m2]'

class CoreXMLReader(CosmosXMLReader):
    '''
        读取CORE XML文件,提起相关信息
    '''
    class Preprocessor(Preprocessor):
        '''
            CORE的数据预处理对象
        '''
        def generate_grid_position(self,core_radius,core_height,channel_number,layer_number):
            '''
                根据堆芯半径,堆芯高度,通道数,几何层数确定各网格位置
            '''
            k = np.arange(channel_number + 1, dtype=float)
            r_edges = core_radius * np.sqrt(k / channel_number)
            r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
            dx = core_height / layer_number
            h_mids = (np.arange(layer_number,dtype=float) + 0.5) * dx 
            return r_mid,h_mids

        def calculate_2d_interpolation(self,x_raw,y_raw,matrix_raw,x,y,method="linear",fill_value=None, bounds_error=False,batch_swtich=False):
            '''
                根据网格点通过二维插值得到新的场
                matrix_raw[iy, ix] ↔ (x_raw[ix], y_raw[iy])
                matrix_new[iy, ix] ↔ (x[ix], y[iy])
            '''
            def interpolation_2d(x_raw_,y_raw_,matrix_raw_,x_,y_): 

                x_raw = np.asarray(x_raw_, dtype=float)
                y_raw = np.asarray(y_raw_, dtype=float)
                Z = np.asarray(matrix_raw_, dtype=float) 
                x = np.asarray(x_, dtype=float)
                y = np.asarray(y_, dtype=float)

                # 保证坐标单调递增
                if np.any(np.diff(x_raw) <= 0):
                    ix = np.argsort(x_raw)
                    x_raw = x_raw[ix]
                    Z = Z[:, ix]
                if np.any(np.diff(y_raw) <= 0):
                    iy = np.argsort(y_raw)
                    y_raw = y_raw[iy]
                    Z = Z[iy, :]

                # 构造插值器
                interp = RegularGridInterpolator((y_raw, x_raw),Z,method=method,bounds_error=bounds_error,fill_value=fill_value)

                # 生成新网格点并插值
                Xn, Yn = np.meshgrid(x, y, indexing="xy")
                pts = np.column_stack([Yn.ravel(), Xn.ravel()])
                Znew = interp(pts).reshape(len(y), len(x))     

                return Znew

            if not batch_swtich:
                Znew = interpolation_2d(x_raw,y_raw,matrix_raw,x,y)
                return Znew
            else:
                Znew_list = []
                for x_raw_,y_raw_,matrix_raw_ in zip(x_raw,y_raw,matrix_raw):
                    Znew = interpolation_2d(x_raw_,y_raw_,matrix_raw_,x,y)
                    Znew_list.append(Znew)
                return Znew_list
        
        def convert_pebble_velocity_to_grid_stay_time(self,pebble_velocity_matrix,core_height,layer_number):
            '''
                转换球流速分布为每个网格的滞留时间分布
            '''
            gird_height = core_height / layer_number
            grid_stay_time_matrix = gird_height / pebble_velocity_matrix

            return grid_stay_time_matrix

    def __init__(self,xml_path):

        super().__init__()
        # 预处理类对象
        self.preprocessor = self.Preprocessor()
        # 输入文件路径
        self.xml_path = xml_path
        # XML输入
        self.root = ET.parse(self.xml_path).getroot()
        # 进行Core计算的基本配置
        self.core_basic_configuration = self.root.find('Core')
        
        # 初始化
        self.x_list = None
        self.y_list = None
        # 堆芯属性
        self.core_properties_dict = {
            'target_burnup':None,
            'channel_number':None,
            'layer_number':None,
            'element_number':None,
            'core_radius':None,
            'core_height':None,
            'temperature_field_steady':None,
            'temperature_field_transient_list':None,
            'temperature_field_transient_time_list':None,
            'neutron_flux_field_steady':None,
            'power_field_transient_list':None,
            'power_field_transient_time_list':None,
            'neutron_flux_field_transient_list':None,
            'neutron_flux_field_transient_time_list':None,
            'pebble_velocity_field_steady':None,
            'pebble_velocity_field_transient_list':None,
            'pebble_velocity_field_transient_time_list':None,
            'grid_stay_time_field_steady':None,
            'grid_stay_time_field_transient_list':None,
            'grid_stay_time_field_transient_time_list':None,
            'maximum_batch':None,
            'discharge_time':None,
            'discharge_mesh':None
        }
        # 随机球
        self.tracer_list = []
        # 求解方式
        self.solver_dict = {
            'nuit_mode':None,
            'nuit_solver':None
        }
        # 模拟控制
        self.control_dict = {
            'accident_begin':None,
            'time':None,
            'time_step':None,
            'output_nuclide':None,
            'output_type':None,
            'output_path':None,
            'write_interval':None,
            'batch_processing':None,
            'save_intermediate_files':None,
            'plot_swtich':None,
            'tracer_history':None
        }
        # 燃料源项控制
        self.core_mars_dict = {
            'mars_input_path':None,
        }
       
        # 默认值
        self.core_properties_dict_default = {
            'target_burnup':88.027,
            'target_burnup_unit':'[GW][d]/[t]',
            'channel_number':10,
            'layer_number':20,
            'element_number':420000,
            'core_radius':150,
            'core_radius_unit':'[cm]',
            'core_height':1100,
            'core_height_unit':'[cm]',
            'temperature_field_unit':'[K]',
            'temperature_field_position_unit':'[cm]',
            'neutron_flux_field_unit':'1/[cm2][s]',
            'neutron_flux_field_position_unit':'[cm]',
            'power_field_unit':'[MW]',
            'power_field_position_unit':'[cm]',
            'pebble_velocity_field_unit':'[m]/[s]',
            'pebble_velocity_field_position_unit':'[cm]',
            'transient_field_time_unit':'[d]',
            'discharge_time':2,
            'discharge_time_unit':'[d]',
            'discharge_mesh':3
        }
        self.tracer_dict_default = {
                'tracer_number':100,
                'each_time_add_number':2,
                'initial_channel':'random',
                'initial_layer':0,
                'add_start':0.0,
                'add_start_unit':'[s]',
                'add_end':0.0,
                'add_end_unit':'[s]',
                'material':{
                    'U238':6.405,
                    'U235':0.595
                },
                'material_mass_unit':'[g]',
                'material_mass':0.0,
                'channel_range':'all'
            }
        self.solver_dict_defalt = {
            'nuit_mode':'constflux',
            'nuit_solver':'CRAM'
        }
        self.control_dict_default = {
            'accident_begin':1e30,
            'accident_begin_unit':'[d]',
            'time_unit':'[s]',
            'time_step_unit':'[s]',
            'output_nuclide':'Cs137',
            'output_type':'Inventory',
            'output_path':'OUTPUT/COREOUTPUT',
            'write_interval':[0],
            'write_interval_unit':'[s]',
            'batch_processing':1,
            'save_intermediate_files':'',
            'plot_swtich':'off',
            'tracer_history':'off'
        }
        self.core_mars_dict_default = {
            'core_time_step_divide_mars_time_step':30
        }
       
    def read_configuration(self):
        '''
            读取配置信息
        ''' 
        # 读取基本配置
        self._read_basic_configuration()

        # Core联合Mars的配置
        if 'ReleaseRate' in self.control_dict['output_type']:
            self.core_mars_configuration = self.root.find('Core-Mars')
            self._read_core_mars_configuration()

    def _read_basic_configuration(self):
        '''
            读取基本Accident计算的配置信息
        '''
        # 读取求解器
        self._read_solver()

        # 读取模拟时间/输出
        self._read_control()

        # 读取堆芯属性
        self._read_core_properties()

        # 读取示踪球列表
        self._read_tracers()

    def _read_core_properties(self):
        '''
            读取堆芯属性
        '''
        core_properties = self.core_basic_configuration.find('CoreProperties')
        # 目标燃耗
        parent = core_properties
        element_name = 'TargetBurnup'
        default_unit = self.core_properties_dict_default['target_burnup_unit']  
        target_unit = POWER + '[d]' + '/' + '[t]'                            
        data_type = 'float'
        default_value = self.core_properties_dict_default['target_burnup']  
        self.core_properties_dict['target_burnup'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        # 通道数
        parent = core_properties
        element_name = 'ChannelNumber'
        option_name = 'value'
        default_value = self.core_properties_dict_default['channel_number']                                       
        data_type = 'int'
        self.core_properties_dict['channel_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 几何层数
        parent = core_properties
        element_name = 'LayerNumber'
        option_name = 'value'
        default_value = self.core_properties_dict_default['layer_number']                                       
        data_type = 'int'
        self.core_properties_dict['layer_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 堆芯半径
        parent = core_properties
        element_name = 'CoreRadius'
        default_unit = self.core_properties_dict_default['core_radius_unit']  
        target_unit = LENGTH
        data_type = 'float'
        default_value = self.core_properties_dict_default['core_radius']  
        self.core_properties_dict['core_radius'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        # 燃料元件数
        parent = core_properties
        element_name = 'ElementNumber'
        option_name = 'value'
        default_value = self.core_properties_dict_default['element_number']                                       
        data_type = 'int'
        self.core_properties_dict['element_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 堆芯高度
        parent = core_properties
        element_name = 'CoreHeight'
        default_unit = self.core_properties_dict_default['core_height_unit']  
        target_unit = LENGTH
        data_type = 'float'
        default_value = self.core_properties_dict_default['core_height']  
        self.core_properties_dict['core_height'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        # 卸料时间
        parent = core_properties
        element_name = 'DischargeTime'
        default_unit = self.core_properties_dict_default['discharge_time_unit']  
        target_unit = TIME
        data_type = 'float'
        default_value = self.core_properties_dict_default['discharge_time']  
        self.core_properties_dict['discharge_time'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        
        # 卸料网格数
        parent = core_properties
        element_name = 'DischargeMesh'
        option_name = 'value'
        default_value = self.core_properties_dict_default['discharge_mesh']                                       
        data_type = 'int'
        self.core_properties_dict['discharge_mesh'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        
        # 生成网格点
        self.x_list,self.y_list = self.preprocessor.generate_grid_position(self.core_properties_dict['core_radius'],self.core_properties_dict['core_height'],self.core_properties_dict['channel_number'],self.core_properties_dict['layer_number'])

        if 'ReleaseRate' in self.control_dict['output_type']:
            # 温度场
            temperature_field = core_properties.find('RequiredTemperatureField')
            # 温度场单位
            parent = temperature_field
            option_name = 'unit'
            default = self.core_properties_dict_default['temperature_field_unit']
            temperature_field_unit = self._read_option(parent,option_name,default)
            # 温度场坐标单位
            parent = temperature_field
            option_name = 'position_unit'
            default = self.core_properties_dict_default['temperature_field_position_unit']
            temperature_field_position_unit = self._read_option(parent,option_name,default)
            # 稳态温度场路径   
            parent = temperature_field
            element_name = 'RequiredSteadyPath'
            option_name = 'value'
            default = False
            temperature_field_steady_path = self._read_element_option(parent,element_name,option_name,default)
            # 瞬态温度场路径与时间
            temperature_field_transient_path_list = []
            temperature_field_transient_time_list = []
            path_list = temperature_field.findall('TransientPath')
            for path in path_list:
                # 瞬态路径
                parent = path
                option_name = 'path'
                default = None
                transient_path = self._read_option(parent,option_name,default)
                if transient_path is None:
                    continue
                temperature_field_transient_path_list.append(transient_path)
                # 瞬态时间
                parent = path
                default_unit = self.core_properties_dict_default['transient_field_time_unit']
                target_unit = TIME
                data_type = 'float'
                default_value = False
                transient_time = self._read_option_value_unit_pair(parent,default_unit,target_unit,data_type,default_value)
                temperature_field_transient_time_list.append(transient_time)
            # 读取稳态温度场
            x_raw,y_raw,temperature_field_steady_raw = self._read_2d_field(temperature_field_steady_path,temperature_field_position_unit,temperature_field_unit,TEMPERATURE)
            temperature_field_steady = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,temperature_field_steady_raw,self.x_list,self.y_list)
            # discharge mesh stay time self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh']
            new_rows = np.full((self.core_properties_dict['discharge_mesh'],temperature_field_steady.shape[1]),273.15+30)
            temperature_field_steady = np.vstack([np.asarray(temperature_field_steady), new_rows])
            self.core_properties_dict['temperature_field_steady'] = temperature_field_steady
            # 读取瞬态温度场
            temperature_field_transient_list = []
            for transient_path in temperature_field_transient_path_list:
                x_raw,y_raw,temperature_field_transient_raw = self._read_2d_field(transient_path,temperature_field_position_unit,temperature_field_unit,TEMPERATURE)
                temperature_field_transient = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,temperature_field_transient_raw,self.x_list,self.y_list)
                temperature_field_transient_list.append(temperature_field_transient)
            self.core_properties_dict['temperature_field_transient_list'] = temperature_field_transient_list
            self.core_properties_dict['temperature_field_transient_time_list'] = temperature_field_transient_time_list

        if 'constflux' == self.solver_dict['nuit_mode']:
            # 中子注量率场
            neutron_flux_field = core_properties.find('RequiredNeutronFluxField')
            # 中子注量率场单位
            parent = neutron_flux_field
            option_name = 'unit'
            default = self.core_properties_dict_default['neutron_flux_field_unit']
            neutron_flux_field_unit = self._read_option(parent,option_name,default)
            # 中子注量率场坐标单位
            parent = neutron_flux_field
            option_name = 'position_unit'
            default = self.core_properties_dict_default['neutron_flux_field_position_unit']
            neutron_flux_field_position_unit = self._read_option(parent,option_name,default)
            # 稳态中子注量率场路径   
            parent = neutron_flux_field
            element_name = 'RequiredSteadyPath'
            option_name = 'value'
            default = False
            neutron_flux_field_steady_path = self._read_element_option(parent,element_name,option_name,default)
            # 瞬态中子注量率场路径与时间
            neutron_flux_field_transient_path_list = []
            neutron_flux_field_transient_time_list = []
            path_list = neutron_flux_field.findall('TransientPath')
            for path in path_list:
                # 瞬态路径
                parent = path
                option_name = 'path'
                default = None
                transient_path = self._read_option(parent,option_name,default)
                if transient_path is None:
                    continue
                neutron_flux_field_transient_path_list.append(transient_path)
                # 瞬态时间
                parent = path
                default_unit = self.core_properties_dict_default['transient_field_time_unit']
                target_unit = TIME
                data_type = 'float'
                default_value = False
                transient_time = self._read_option_value_unit_pair(parent,default_unit,target_unit,data_type,default_value)
                neutron_flux_field_transient_time_list.append(transient_time)
            # 读取稳态中子注量率场
            x_raw,y_raw,neutron_flux_field_steady_raw = self._read_2d_field(neutron_flux_field_steady_path,neutron_flux_field_position_unit,neutron_flux_field_unit,'/'+'[cm2]'+TIME,batch_swtich=True)
            neutron_flux_field_steady = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,neutron_flux_field_steady_raw,self.x_list,self.y_list,batch_swtich=True)
            for idx,neutron_flux_field in enumerate(neutron_flux_field_steady):
                # discharge mesh stay time self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh']
                new_rows = np.full((self.core_properties_dict['discharge_mesh'],neutron_flux_field.shape[1]),0.0)
                neutron_flux_field_steady[idx] = np.vstack([np.asarray(neutron_flux_field), new_rows])
            self.core_properties_dict['neutron_flux_field_steady'] = neutron_flux_field_steady
            maximum_batch = len(self.core_properties_dict['neutron_flux_field_steady'])
            # 读取瞬态中子注量率场
            neutron_flux_field_transient_list = []
            for transient_path in neutron_flux_field_transient_path_list:
                x_raw,y_raw,neutron_flux_field_transient_raw = self._read_2d_field(transient_path,neutron_flux_field_position_unit,neutron_flux_field_unit,'/'+'[cm2]'+TIME,batch_swtich=True)
                neutron_flux_field_transient = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,neutron_flux_field_transient_raw,self.x_list,self.y_list,batch_swtich=True)
                for idx,neutron_flux_field in enumerate(neutron_flux_field_transient):
                    # discharge mesh stay time self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh']
                    new_rows = np.full((self.core_properties_dict['discharge_mesh'],neutron_flux_field.shape[1]),0.0)
                    neutron_flux_field_transient[idx] = np.vstack([np.asarray(neutron_flux_field), new_rows])
                maximum_batch = min(maximum_batch,len(neutron_flux_field_transient))
                neutron_flux_field_transient_list.append(neutron_flux_field_transient)
            self.core_properties_dict['neutron_flux_field_transient_list'] = neutron_flux_field_transient_list
            self.core_properties_dict['neutron_flux_field_transient_time_list'] = neutron_flux_field_transient_time_list
            # 最大批次
            self.core_properties_dict['maximum_batch'] = maximum_batch - 1
        elif 'constpower' == self.solver_dict['nuit_mode']:
            # 功率场
            power_field = core_properties.find('RequiredPowerField')
            # 功率场单位
            parent = power_field
            option_name = 'unit'
            default = self.core_properties_dict_default['power_field_unit']
            power_field_unit = self._read_option(parent,option_name,default)
            # 功率场坐标单位
            parent = power_field
            option_name = 'position_unit'
            default = self.core_properties_dict_default['power_field_position_unit']
            power_field_position_unit = self._read_option(parent,option_name,default)
            # 稳态功率场路径   
            parent = power_field
            element_name = 'RequiredSteadyPath'
            option_name = 'value'
            default = False
            power_field_steady_path = self._read_element_option(parent,element_name,option_name,default)
            # 瞬态功率场路径与时间
            power_field_transient_path_list = []
            power_field_transient_time_list = []
            path_list = power_field.findall('TransientPath')
            for path in path_list:
                # 瞬态路径
                parent = path
                option_name = 'path'
                default = None
                transient_path = self._read_option(parent,option_name,default)
                if transient_path is None:
                    continue
                power_field_transient_path_list.append(transient_path)
                # 瞬态时间
                parent = path
                default_unit = self.core_properties_dict_default['transient_field_time_unit']
                target_unit = TIME
                data_type = 'float'
                default_value = False
                transient_time = self._read_option_value_unit_pair(parent,default_unit,target_unit,data_type,default_value)
                power_field_transient_time_list.append(transient_time)
            # 读取稳态功率场
            x_raw,y_raw,power_field_steady_raw = self._read_2d_field(power_field_steady_path,power_field_position_unit,power_field_unit,'[MW]',batch_swtich=True)
            power_field_steady = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,power_field_steady_raw,self.x_list,self.y_list,batch_swtich=True)
            for idx,power_field in enumerate(power_field_steady):
                # discharge mesh stay time self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh']
                new_rows = np.full((self.core_properties_dict['discharge_mesh'],power_field.shape[1]),0.0)
                power_field_steady[idx] = np.vstack([np.asarray(power_field), new_rows])
            self.core_properties_dict['power_field_steady'] = power_field_steady
            maximum_batch = len(self.core_properties_dict['power_field_steady'])
            # 读取瞬态功率场
            power_field_transient_list = []
            for transient_path in power_field_transient_path_list:
                x_raw,y_raw,power_field_transient_raw = self._read_2d_field(transient_path,power_field_position_unit,power_field_unit,'[MW]',batch_swtich=True)
                power_field_transient = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,power_field_transient_raw,self.x_list,self.y_list,batch_swtich=True)
                for idx,power_field in enumerate(power_field_transient):
                    # discharge mesh stay time self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh']
                    new_rows = np.full((self.core_properties_dict['discharge_mesh'],power_field.shape[1]),0.0)
                    power_field_transient[idx] = np.vstack([np.asarray(power_field), new_rows])
                maximum_batch = min(maximum_batch,len(power_field_transient))
                power_field_transient_list.append(power_field_transient)
            self.core_properties_dict['power_field_transient_list'] = power_field_transient_list
            self.core_properties_dict['power_field_transient_time_list'] = power_field_transient_time_list
            # 最大批次
            self.core_properties_dict['maximum_batch'] = maximum_batch - 1

        # 速度场
        pebble_velocity_field = core_properties.find('RequiredPebbleVelocityField')
        # 速度场单位
        parent = pebble_velocity_field
        option_name = 'unit'
        default = self.core_properties_dict_default['pebble_velocity_field_unit']
        pebble_velocity_field_unit = self._read_option(parent,option_name,default)
        # 速度场坐标单位
        parent = pebble_velocity_field
        option_name = 'position_unit'
        default = self.core_properties_dict_default['pebble_velocity_field_position_unit']
        pebble_velocity_field_position_unit = self._read_option(parent,option_name,default)
        # 稳态速度场路径   
        parent = pebble_velocity_field
        element_name = 'RequiredSteadyPath'
        option_name = 'value'
        default = False
        pebble_velocity_field_steady_path = self._read_element_option(parent,element_name,option_name,default)
        # 瞬态速度场路径与时间
        pebble_velocity_field_transient_path_list = []
        pebble_velocity_field_transient_time_list = []
        path_list = pebble_velocity_field.findall('TransientPath')
        for path in path_list:
            # 瞬态路径
            parent = path
            option_name = 'path'
            default = None
            transient_path = self._read_option(parent,option_name,default)
            if transient_path is None:
                continue
            pebble_velocity_field_transient_path_list.append(transient_path)
            # 瞬态时间
            parent = path
            default_unit = self.core_properties_dict_default['transient_field_time_unit']
            target_unit = TIME
            data_type = 'float'
            default_value = False
            transient_time = self._read_option_value_unit_pair(parent,default_unit,target_unit,data_type,default_value)
            pebble_velocity_field_transient_time_list.append(transient_time)
        # 读取稳态速度场
        x_raw,y_raw,pebble_velocity_field_steady_raw = self._read_2d_field(pebble_velocity_field_steady_path,pebble_velocity_field_position_unit,pebble_velocity_field_unit,LENGTH+'/'+TIME)
        pebble_velocity_field_steady = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,pebble_velocity_field_steady_raw,self.x_list,self.y_list)
        self.core_properties_dict['pebble_velocity_field_steady'] = pebble_velocity_field_steady
        # 读取瞬态速度场
        pebble_velocity_field_transient_list = []
        for transient_path in pebble_velocity_field_transient_path_list:
            x_raw,y_raw,pebble_velocity_field_transient_raw = self._read_2d_field(transient_path,pebble_velocity_field_position_unit,pebble_velocity_field_unit,LENGTH+'/'+TIME)
            pebble_velocity_field_transient = self.preprocessor.calculate_2d_interpolation(x_raw,y_raw,pebble_velocity_field_transient_raw,self.x_list,self.y_list)
            pebble_velocity_field_transient_list.append(pebble_velocity_field_transient)
        self.core_properties_dict['pebble_velocity_field_transient_list'] = pebble_velocity_field_transient_list
        self.core_properties_dict['pebble_velocity_field_transient_time_list'] = pebble_velocity_field_transient_time_list

        # 球流速度场->网格滞留时间分布
        grid_stay_time_field_steady = self.preprocessor.convert_pebble_velocity_to_grid_stay_time(pebble_velocity_field_steady,self.core_properties_dict['core_height'],self.core_properties_dict['layer_number'])
        new_rows = np.full((self.core_properties_dict['discharge_mesh'],grid_stay_time_field_steady.shape[1]),self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh'])
        self.core_properties_dict['grid_stay_time_field_steady'] = np.vstack([np.asarray(grid_stay_time_field_steady), new_rows])
       
        self.core_properties_dict['grid_stay_time_field_transient_list'] = []
        for pebble_velocity_field_transient in pebble_velocity_field_transient_list:
            grid_stay_time_field_transient = self.preprocessor.convert_pebble_velocity_to_grid_stay_time(pebble_velocity_field_transient,self.core_properties_dict['core_height'],self.core_properties_dict['layer_number'])
            new_rows = np.full((self.core_properties_dict['discharge_mesh'],grid_stay_time_field_transient.shape[1]),self.core_properties_dict['discharge_time']/self.core_properties_dict['discharge_mesh'])
            self.core_properties_dict['grid_stay_time_field_transient_list'].append(np.vstack([np.asarray(grid_stay_time_field_transient), new_rows]))
        self.core_properties_dict['grid_stay_time_field_transient_time_list'] = self.core_properties_dict['pebble_velocity_field_transient_time_list']

    def _read_tracers(self):
        '''
            读取示踪球
        '''
        tracers = self.core_basic_configuration.find('RequiredTracers')
        if tracers is None:
            print('缺少RequiredTracers信息')
            sys.exit()
        tracer_info_list = tracers.findall('Tracer')
        if len(tracer_info_list) == 0:
            print('缺少RequiredTracers信息')
            sys.exit()
        tracer_list = []
        for tracer in tracer_info_list:
            tracer_dict = {
                'tracer_number':None,
                'each_time_add_number':None,
                'initial_channel':None,
                'initial_layer':None,
                'add_start':None,
                'add_end':None,
                'material':None,
                'channel_range':None
            }
            # 示踪球数量
            parent = tracer
            element_name = 'TracerNumber'
            option_name = 'value'
            default_value = self.tracer_dict_default['tracer_number']                                       
            data_type = 'int'
            tracer_dict['tracer_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

            # 每次添加的个数
            parent = tracer
            element_name = 'EachTimeAddNumber'
            option_name = 'value'
            default_value = self.tracer_dict_default['each_time_add_number']                                       
            data_type = 'int'
            tracer_dict['each_time_add_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

            # 初始通道
            parent = tracer
            element_name = 'InitialChannel'
            option_name = 'value'
            default_value = self.tracer_dict_default['initial_channel']                                       
            data_type = 'str'
            initial_channel = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
            tracer_dict['initial_channel'] = initial_channel

            # 初始几何层
            parent = tracer
            element_name = 'InitialLayer'
            option_name = 'value'
            default_value = self.tracer_dict_default['initial_layer']                                       
            data_type = 'str'
            initial_layer = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
            tracer_dict['initial_layer'] = initial_layer

            # 循环通道的范围
            parent = tracer
            element_name = 'ChannelRange'
            option_name = 'value'
            default_value = self.tracer_dict_default['channel_range']                                       
            data_type = 'str'
            channel_range = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
            if channel_range not in ['all']:
                channel_range = [int(x) for x in channel_range]
            tracer_dict['channel_range'] = channel_range
            
            # 添加开始时刻
            parent = tracer
            element_name = 'AddStart'
            default_unit = self.tracer_dict_default['add_start_unit']  
            target_unit = TIME
            data_type = 'float'
            default_value = self.tracer_dict_default['add_start']  
            tracer_dict['add_start'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

            # 添加结束时刻
            parent = tracer
            element_name = 'AddEnd'
            default_unit = self.tracer_dict_default['add_end_unit']  
            target_unit = TIME
            data_type = 'float'
            default_value = self.tracer_dict_default['add_end']  
            tracer_dict['add_end'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

            # 材料
            material = tracer.find('Material')
            if material is None:
                material_dict = self.tracer_dict_default['material']
            else:
                material_dict = {}
                nuclide_list = material.findall('Nuclide')
                for nuclide in nuclide_list:
                    name = self._read_option(nuclide,'name',default=None)
                    if name is None:
                        continue
                    parent = nuclide
                    default_unit = self.tracer_dict_default['material_mass_unit']
                    target_unit = MASS
                    data_type = 'float'
                    default_value = self.tracer_dict_default['material_mass']
                    value = self._read_option_value_unit_pair(parent,default_unit,target_unit,data_type,default_value=default_value,default_time=False,decay_constant=None)
                    material_dict[name] = value
            tracer_dict['material'] = copy.deepcopy(material_dict)

            # 添加到示踪球列表
            tracer_list.append(tracer_dict)
        
        self.tracer_list = tracer_list

    def _read_solver(self):
        '''
            读取求解模型
        '''
        solver = self.core_basic_configuration.find('Solver')
        # Nuit处理的数据类型
        parent = solver
        element_name = 'NuitMode'
        option_name = 'value'
        default_value = self.solver_dict_defalt['nuit_mode']                                       
        self.solver_dict['nuit_mode'] = self._read_element_option(parent,element_name,option_name,default=default_value)

        # Nuit的矩阵求解器
        parent = solver
        element_name = 'NuitSolver'
        option_name = 'value'
        default_value = self.solver_dict_defalt['nuit_solver']                                       
        self.solver_dict['nuit_solver'] = self._read_element_option(parent,element_name,option_name,default=default_value)

    def _read_control(self):
        '''
            读取模拟时间及输出
        '''
        control = self.core_basic_configuration.find('Control')
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

        # 事故开始时间
        parent = control
        element_name = 'AccidentBegin'
        default_unit = self.control_dict_default['accident_begin_unit']  
        target_unit = TIME                                     
        data_type = 'float'
        default_value = self.control_dict_default['accident_begin']  
        self.control_dict['accident_begin'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

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

        # 输出核素
        parent = output
        element_name = 'OutputNuclide'
        option_name = 'value'
        default_value = self.control_dict_default['output_nuclide']                                       
        nuclides = self._read_element_option(parent,element_name,option_name,default=default_value)
        self.control_dict['output_nuclide'] = nuclides.split()

        # 输出类型
        parent = output
        element_name = 'OutputType'
        option_name = 'value'
        default_value = self.control_dict_default['output_type']                                       
        output_type = self._read_element_option(parent,element_name,option_name,default=default_value)
        self.control_dict['output_type'] = output_type.split()

        # 处理分批次数
        parent = control
        element_name = 'BatchProcessing'
        option_name = 'value'
        default_value = self.control_dict_default['batch_processing']                                       
        data_type = 'int'
        self.control_dict['batch_processing'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 需保存的中间文件
        parent = control
        element_name = 'SaveIntermediateFiles'
        option_name = 'value'
        default_value = self.control_dict_default['save_intermediate_files']                                       
        output_type = self._read_element_option(parent,element_name,option_name,default=default_value)
        self.control_dict['save_intermediate_files'] = output_type.split()
        
        # 对结果绘图
        parent = output
        element_name = 'OutputType'
        option_name = 'plot'
        default_value = self.control_dict_default['plot_swtich']                                       
        plot_swtich = self._read_element_option(parent,element_name,option_name,default=default_value)
        self.control_dict['plot_swtich'] = plot_swtich.strip()
        
        # 保存Tracer历史
        parent = output
        element_name = 'OutputType'
        option_name = 'tracer_history'
        default_value = self.control_dict_default['tracer_history']                                       
        plot_swtich = self._read_element_option(parent,element_name,option_name,default=default_value)
        self.control_dict['tracer_history'] = plot_swtich.strip()

    def _read_2d_field(self,xlsx_path,position_unit,field_unit,target_unit,batch_swtich=False):
        '''
            读取二维场
        '''
        if not batch_swtich: 
            df = pd.read_excel(xlsx_path,header=None)
            x_raw = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()
            y_raw = pd.to_numeric(df.iloc[1:, 0], errors="coerce").to_numpy()
            matrix_raw = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
            x,_ = self.preprocessor.unit_convert(x_raw,position_unit,LENGTH)
            y,_ = self.preprocessor.unit_convert(y_raw,position_unit,LENGTH)
            matrix,_ = self.preprocessor.unit_convert(matrix_raw,field_unit,target_unit)
            return x,y,matrix
        else:
            xls = pd.ExcelFile(xlsx_path)
            sheet_names = xls.sheet_names
            x_list = []
            y_list = []
            matrix_list = []
            for batch in range(len(sheet_names)):
                sheet_name = str(batch)
                df = pd.read_excel(xlsx_path,header=None,sheet_name=sheet_name)
                x_raw = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()
                y_raw = pd.to_numeric(df.iloc[1:, 0], errors="coerce").to_numpy()
                matrix_raw = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
                x,_ = self.preprocessor.unit_convert(x_raw,position_unit,LENGTH)
                y,_ = self.preprocessor.unit_convert(y_raw,position_unit,LENGTH)
                matrix,_ = self.preprocessor.unit_convert(matrix_raw,field_unit,target_unit)
                x_list.append(x)
                y_list.append(y)
                matrix_list.append(matrix)
            return x_list,y_list,matrix_list
        
    def _read_core_mars_configuration(self):
        '''
            读取Core联合Mars计算的配置
        ''' 
        # Mars输入卡路径
        parent = self.core_mars_configuration
        element_name = 'MarsInputPath'
        option_name = 'value'
        default_value = False
        self.core_mars_dict['mars_input_path'] = self._read_element_option(parent,element_name,option_name,default=default_value)
