from TOOL.cosmos_xml_reader import CosmosXMLReader
import xml.etree.ElementTree as ET
import numpy as np
import os
from itertools import accumulate

# Mars内部计算量纲
LENGTH = '[cm]'
AREA = '[cm2]'
VOLUME = '[cm3]'
TIME = '[s]'
MASS = '[g]'
MOLECULUS = '[mol]'
ENERGY = '[J]'
PRESSURE = '[Pa]'
TEMPERATURE = '[K]'
ACTIVITY = '[Bq]'
POWER = '[GW]'

class MarsXMLReader(CosmosXMLReader):
    '''
        读取MARS XML文件,提取相关信息
    '''
    def __init__(self,xml_path):
        super().__init__()
        self.xml_path = xml_path
        self.file_path_diffusion_coef_default = r'TOOL/DiffusionLib.dat'
        self.file_path_decay_constant_default = r'SourceTermAnalysis/TOOL/DecayLib.dat'
        self.root = ET.parse(self.xml_path).getroot()

        # 初始化
        self.fuel_properties_dict = {
            'geometry':{
                'r_element':None,
                'r_graphite_grain':None,
                'r_particle':None
            },
            'particle_number':None,
            'uranium_contamination':{
                'uranium_contamination_element':None,
                'uranium_contamination_graphite_grain':None,
                'uranium_contamination_particle':None,
                'uranium_contamination_kernel':None
            },
            'material_properties':{
                'molar_mass_kernel':None,
                'molar_volume_kernel':None,
                'density_kernel':None,
                'porosity_buffer':None,
                'density_buffer':None,
                'creep_coef_pyc':None,
                'creep_poisson_ratio_pyc':None,
                'poisson_ratio_pyc':None,
                'density_pyc':None,
                'density_sic':None,
                'tensile_strength_sic':None,
                'weibull_sic':None,
                'sic_thermal_decomposition_alpha':None,
                'sic_thermal_decomposition_beta':None,
                'sic_manufacturing_failure_fraction':None,
                'density_graphite_matrix':None,
            }
        }
        self.models_dict = {
            'intra_pebble_temperature':{
                'temperature_field_model':None,
                'temperature_zone_number':None,
                'transient_temperature_begin':None
            },
            'recoil':{
                'recoil_model':None,
                'r_recoil_element':None,
                'r_recoil_graphite_grain':None,
                'r_recoil_particle':None
            },
            'fuel_performance':{
                'failure_model':None,
                'failure_threshold':None,
                'step_failure_fraction_increment':None,
                'step_failure_fraction_time':None,
                'pressure_model':None,
                'redlich_kwong':{
                    'Tc_CO':None,
                    'Tc_Kr':None,
                    'Tc_Xe':None,
                    'Pc_CO':None,
                    'Pc_Kr':None,
                    'Pc_Xe':None,
                },
                'van_der_waals':{
                    'a_CO':None,
                    'a_Kr':None,
                    'a_Xe':None,
                    'b_CO':None,
                    'b_Kr':None,
                    'b_Xe':None,
                },
                'yield_Xe':None,
                'yield_Kr':None,
                'fast_neutron_share':None,
                'fp_corrosion_model':None,
                'pyc_swelling_model':None,
                'creep_coef_model_pyc':None,
                'weibull_coef_model_sic':'PANAMA',
                'sic_thermal_decomposition_model':'PANAMA',
                'sic_stress_model':None,
                'intergranular_corrosion_model':None,
            },
            'parameter':{
                'nuclide_diffusion_coef':'Arrhenius',
                'thermal_conductivity_kernel':'Idaho2004',
                'thermal_conductivity_buffer':'Idaho2004',
                'thermal_conductivity_pyc':'Idaho2004',
                'thermal_conductivity_sic':'Idaho2004',
                'thermal_conductivity_graphite_matrix':'Idaho2004',
                'heat_capacity_kernel':'Bison',
                'heat_capacity_buffer':'Bison',
                'heat_capacity_pyc':'Bison',
                'heat_capacity_sic':'Bison',
                'heat_capacity_graphite_matrix':'Bison',
                'density_kernel':'ExternalValue',
                'density_buffer':'ExternalValue',
                'density_pyc':'ExternalValue',
                'density_sic':'ExternalValue',
                'density_graphite_matrix':'ExternalValue',
            },
            'adsorption':{
                'henry_a':None,
                'henry_b':None,
                'freundlich_a':None,
                'freundlich_b':None,
                'freundlich_e':None,
                'freundlich_f':None,
                'c_convert':None,
                'mode':None,
                'pressure_env':None,
                'component_env':None,
                'mole_fraction_env':None,
                'mole_mass_env':None,
                'dynamic_viscosity_model':None,
                'binary_diffusion_coef_model':None,
                'coolant_velocity':None,
                'coolant_density_model':None,
                'epsilon':None,
                'A_iso':None,
                'B_iso':None,
                'D_iso':None,
                'E_iso':None,
                'd1_iso':None,
                'd2_iso':None,
            },
            'mass_transfer':{
                'mode':None,
                'element_mass_transfer_coef':None,
                'graphite_grain_mass_transfer_coef':None,
                'particle_mass_transfer_coef':None,
                'kernel_mass_transfer_coef':None,
            },
            'diffusion_model':None
        }
        self.conditions_dict = {
            'c_element_environment':None,
            'c_particle_environment':None,
            'c_graphite_grain_environment':None,
            'c_kernel_environment':None,
            'element_initial_inventory':None,
            'initial_temperature':None
        }
        self.solver_dict = {
            'mesh':{
                'element_mesh_number':None,
                'graphite_grain_mesh_number':None,
                'particle_mesh_number':None,
                'kernel_mesh_number':None,
                'element_mesh_bias':None,
                'graphite_grain_mesh_bias':None,
                'particle_mesh_bias':None,
                'kernel_mesh_bias':None,
            },
            'diffusion_solver':{
                'crank_nicolson_time_weight':None,
                'element_diffusion_solver':None,
                'graphite_grain_diffusion_solver':None,
                'particle_diffusion_solver':None,
                'kernel_diffusion_solver':None,
            },
            'steady_temperature_field_max_iterations':None,
            'steady_temperature_field_residual':None,
            'transient_temperature_field_max_iterations':None,
            'transient_temperature_field_residual':None,
            'steady_gases_release_fraction_max_iteration':None,
            'failure_adjustment_model':None,
            'failure_adjust_split_number':None,
            'failure_adjust_time_step_number':None
        }
        self.control_dict = {
            'fsar_mode':None,
            'time':None,
            'time_step':None,
            'output_path':None,
            'write_interval':None,
            'output_type':None,
            'distribution_type':None,
            'distribution_time':None
        }
        self.external_conditions_dict = {
            'temperature':None,
            'inventory':None,
            'nuclide':None,
            'decay_constant':None,
            'diffusion_coef':{
                'D_element':None,
                'D_particle':None,
                'D_kernel':None,
                'D_graphite_grain':None,
                'A_element':None,
                'A_particle':None,
                'A_kernel':None,
                'A_graphite_grain':None,
            },
            'element_power':None,
            'burnup':None,
            'neutron_flux':None,
            'accident_time':None,
        }
        self.core_mars_dict = {
            'core_time_step_divid_mars_time_step':None,
            'output_type':None
        }
        self.fsar_mode_dict = {
            'nuclide_list':None,
            'inventory_dict':None,
            'output_path':None,
            'decay_constant_dict':None,
            'diffusion_coef_dict':None,
            'temperature_list':None,
            'temperature_share_list':None,
            'time':None,
            'time_step':None,
        }

        # 默认值
        self.fuel_properties_dict_default = {
            'geometry':{
                'r_element':[2.5,3.0],
                'r_element_unit':'[cm]',
                'r_graphite_grain':[6e-4],
                'r_graphite_grain_unit':'[cm]',
                'r_particle':[2.50e-2,3.40e-2,3.80e-2,4.15e-2,4.55e-2],
                'r_particle_unit':'[cm]',
            },
            'particle_number':11600,
            'uranium_contamination':{
                'uranium_contamination_element':[7e-7*((2.5**3)/(3**3)),7e-7*((3**3-2.5**3)/(3**3))],
                'uranium_contamination_graphite_grain':[0.0],
                'uranium_contamination_particle':[0.0,5e-8,1e-4,1e-6,1e-6]
            },
            'material_properties':{
                'molar_mass_kernel':270.03,
                'molar_mass_kernel_unit':'[g]/[mol]',
                'density_kernel':10960,
                'density_kernel_unit':'[kg]/[m3]',
                'porosity_buffer':0.5,
                'density_buffer':0.95,
                'density_buffer_unit':'[g]/[cm3]',
                'creep_coef_pyc':2.715e-4,
                'creep_poisson_ratio_pyc':0.5,
                'poisson_ratio_pyc':0.33,
                'density_pyc':1.9,
                'density_unit_pyc':'[g]/[cm3]',
                'density_sic':3.2,
                'density_unit_sic':'[g]/[cm3]',
                'tensile_strength_sic':834,
                'tensile_strength_unit_sic':'[MPa]',
                'weibull_sic':8.02,
                'sic_thermal_decomposition_alpha':0.0001,
                'sic_thermal_decomposition_beta':4.0,
                'sic_manufacturing_failure_fraction':1.6e-4,
                'density_graphite_matrix':2.5,
                'density_unit_graphite_matrix':'[g]/[cm3]',
            }
        }
        self.models_dict_default = {
            'intra_pebble_temperature':{
                'temperature_field_model':'uniform',
                'temperature_zone_number':0,
                'transient_temperature_begin':1e30,
                'transient_temperature_begin_unit':'[s]'
            },
            'recoil':{
                'recoil_model':'off',
                'r_recoil_element':[0.0,0.0],
                'r_recoil_element_unit':'[cm]',
                'r_recoil_graphite_grain':[0.0],
                'r_recoil_graphite_grain_unit':'[cm]',
                'r_recoil_particle':[0.0,0.0,0.0,0.0,0.0],
                'r_recoil_particle_unit':'[cm]'
            },
            'fuel_performance':{
                'failure_model':'StepFailure',
                'failure_threshold':2e-5,
                'step_failure_fraction_increment':[],
                'step_failure_fraction_time':[],
                'step_failure_fraction_time_unit':'[s]',
                'pressure_model':'IdealGas',
                'redlich_kwong':{
                    'Tc_CO':132.91,
                    'Tc_CO_unit':'[K]',
                    'Tc_Kr':209.45,
                    'Tc_Kr_unit':'[K]',
                    'Tc_Xe':589.75,
                    'Tc_Xe_unit':'[K]',
                    'Pc_CO':3.5e6,
                    'Pc_CO_unit':'[Pa]',
                    'Pc_Kr':5.5e6,
                    'Pc_Kr_unit':'[Pa]',
                    'Pc_Xe':5.9e6,
                    'Pc_Xe_unit':'[Pa]',
                },
                'van_der_waals':{
                    'a_CO':1.505e-7,
                    'a_Kr':2.349e-7,
                    'a_Xe':4.250e-7,
                    'b_CO':3.985e-5,
                    'b_Kr':3.978e-5,
                    'b_Xe':5.105e-6,
                },
                'yield_Xe':0.15,
                'yield_Kr':0.16,
                'fast_neutron_share':0.3,
                'fp_corrosion_model':'Attenuation',
                'pyc_swelling_model':'TecDoc1647-1',
                'creep_coef_model_pyc':'TecDoc1647',
                'sic_stress_model':'Bubble',
                'intergranular_corrosion_model':'off',
            },
            'adsorption':{
                'henry_a':0.0,
                'henry_b':0.0,
                'freundlich_a':0.0,
                'freundlich_b':0.0,
                'freundlich_e':0.0,
                'freundlich_f':0.0,
                'c_convert':0.0,
                'c_convert_unit':'[mol]/[g]',
                'mode':'Disable',
                'pressure_env':1e5,
                'pressure_env_unit':'[Pa]',
                'component_env':['Helium'],
                'mole_fraction_env':[1.0],
                'mole_mass_env':[4e-3],
                'mole_mass_env_unit':'[kg]/[mol]',
                'dynamic_viscosity_model':'GETTER',
                'binary_diffusion_coef_model':'Chapman-Enskog',
                'coolant_velocity':10,
                'coolant_density_model':'GETTER',
                'coolant_velocity_unit':'[m]/[s]',
                'epsilon':1-0.63,
                'A_iso':0.0,
                'B_iso':0.0,
                'D_iso':0.0,
                'E_iso':0.0,
                'd1_iso':0.0,
                'd2_iso':0.0,
            },
            'mass_transfer':{
                'mode':'Constant',
                'element_mass_transfer_coef':50,
                'element_mass_transfer_coef_unit':'[cm]/[s]',
                'graphite_grain_mass_transfer_coef':100,
                'graphite_grain_mass_transfer_coef_unit':'[cm]/[s]',
                'particle_mass_transfer_coef':100,
                'particle_mass_transfer_coef_unit':'[cm]/[s]',
                'kernel_mass_transfer_coef':100,
                'kernel_mass_transfer_coef_unit':'[cm]/[s]',
            },
            'diffusion_model':'Numerical'
        }
        self.conditions_dict_default = {
            'c_element_environment':0.0,
            'c_element_environment_unit':'[Bq]/[cm3]',
            'c_particle_environment':0.0,
            'c_particle_environment_unit':'[Bq]/[cm3]',
            'element_initial_inventory':0.0,
            'element_initial_inventory_unit':'[Bq]',
            'initial_temperature':1000,
            'initial_temperature_unit':'[K]'
        }
        self.solver_dict_default = {
            'mesh':{
                'element_mesh_number':[9,20],
                'graphite_grain_mesh_number':[80],
                'particle_mesh_number':[39,39,39,39,39],
                'element_mesh_bias':[0.0,0.0],
                'graphite_grain_mesh_bias':[0.0],
                'particle_mesh_bias':[0.0,0.0,0.0,0.0,0.0],
            },
            'diffusion_solver':{
                'crank_nicolson_time_weight':0.5,
                'element_diffusion_solver':'fvm_euler_thomas',
                'graphite_grain_diffusion_solver':'fvm_euler_thomas',
                'particle_diffusion_solver':'fvm_euler_thomas',
                'kernel_diffusion_solver':'fvm_euler_thomas',
            },
            'steady_temperature_field_max_iterations':10000,
            'steady_temperature_field_residual':2e-4,
            'transient_temperature_field_max_iterations':20,
            'transient_temperature_field_residual':0.02,
            'steady_gases_release_fraction_max_iteration':3000,
            'failure_adjustment_model':'on',
            'failure_adjust_split_number':10,
            'failure_adjust_time_step_number':10
        }
        self.control_dict_default = {
            'fsar_mode':'off',
            'time':None,
            'time_unit':'[s]',
            'time_step':None,
            'time_step_unit':'[s]',
            'output_path':'OUTPUT/MARSOUTPUT',
            'output_type':['ReleaseRate'],
            'write_interval':[0],
            'write_interval_unit':'[s]',
            'distribution_type':[],
            'distribution_time':[]
        }
        self.external_conditions_dict_default = {
            'temperature':None,
            'temperature_unit':'[K]',
            'inventory':None,
            'inventory_unit':'[Bq]',
            'nuclide':None,
            'decay_constant':None,
            'decay_constant_unit':'/[s]',
            'diffusion_coef':{
                'D_graphite_matrix':None,
                'D_graphite_grain':None,
                'D_kernel':None,
                'D_buffer':None,
                'D_pyc':None,
                'D_sic':None,
                'A_graphite_matrix':None,
                'A_graphite_grain':None,
                'A_kernel':None,
                'A_buffer':None,
                'A_pyc':None,
                'A_sic':None,
                'D_graphite_matrix_unit':'[cm2]/[s]',
                'D_graphite_grain_unit':'[cm2]/[s]',
                'D_kernel_unit':'[cm2]/[s]',
                'D_buffer_unit':'[cm2]/[s]',
                'D_pyc_unit':'[cm2]/[s]',
                'D_sic_unit':'[cm2]/[s]',
                'A_graphite_matrix_unit':'[J]/[mol]',
                'A_graphite_grain_unit':'[J]/[mol]',
                'A_kernel_unit':'[J]/[mol]',
                'A_buffer_unit':'[J]/[mol]',
                'A_pyc_unit':'[J]/[mol]',
                'A_sic_unit':'[J]/[mol]',
            },
            'element_power':[0.0,0.0],
            'element_power_unit':'[J]/[s]',
            'burnup':[0.0,90.0,90.0],
            'burnup_unit':'[MW][d]/[t]',
            'neutron_flux':[1e12,1e12],
            'neutron_flux_unit':'[n]/[cm2][s]',
            'accident_time':1e30,
            'accident_time_unit':'[s]',
        }
        self.fsar_mode_dict_default = {
            'write_interval':-10,
            'output_path':'OUTPUT/HtrPmFullCoreRelease',
            'nuclide_list':['Cs137','Cs134','Sr89','Sr90','Ag110m1'],
            'inventory_raw_list':[1.91e16,1.53e16,2.51e17,1.25e16,2.55e14],
            'inventory_unit':'[Bq]',
            'temperature_unit':'[K]',
            'temperature_list':[1273.15,1223.15,1173.15,1123.15,1073.15,973.15,873.15],
            'temperature_share_list':[0.0102,0.0506,0.078,0.137,0.258,0.129,0.339],
            'time':1068.8*24*3600,
            'time_unit':'[s]',
            'time_step':0.3*24*3600,
            'time_step_unit':'[s]',
        }

    def read_configuration(self,tracer=None,target_nuclide=None,core_reader=None):

        # MARS独立计算
        if tracer is None:
            self._read_mars_mars_configuration() 
        # 联合CORE计算
        else:
            self._read_core_mars_configuration(tracer,target_nuclide,core_reader)
        
        self._read_shared_configuration()

        # 输出完整的输入卡
        self._write_completed_xml(tracer)
    
    def _read_core_mars_xml(self):
        '''
            读取联合CORE使用时XML中的输入参数
        '''
        core = self.root.find('CORE-MARS')
        # 输出类型
        parent = core
        element_name = 'OutputType'
        option_name = 'value'
        data_type = 'str-array'
        default_value = ['ReleaseRate']                                     
        output_type = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 时间步
        parent = core
        element_name = 'CoreTimeStepDividMarsTimeStep'
        default_unit = '[s]'
        target_unit = TIME                                     
        data_type = 'float'
        default_value = 0
        core_time_step_divid_mars_time_step = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        if core_time_step_divid_mars_time_step > 0:
            pass
        elif core_time_step_divid_mars_time_step < 0: 
            core_time_step_divid_mars_time_step = self._read_element_option(parent,element_name,'value',default=False,data_type='int')
        elif core_time_step_divid_mars_time_step == 0:
            core_time_step_divid_mars_time_step = -30 
        
        self.core_mars_dict['core_time_step_divid_mars_time_step'] = core_time_step_divid_mars_time_step
        self.core_mars_dict['output_type'] = output_type
    
        return output_type,core_time_step_divid_mars_time_step
    
    def _examine_shared_output(self,properties,core_control,output_type):
        '''
            检查CORE中需求的输出,是否在MARS中填写
        '''
        for prop in properties:
            if prop in core_control.output_type and prop not in output_type:
                output_type.append(prop)
                
        return output_type
        
    def _read_core_mars_configuration(self,tracer,target_nuclide,core_control):
        '''
            读取联合使用CORE时的输入参数
        '''
        # 读取强制条件
        self.control_dict['fsar_mode'] = 'off'
        # 读取输入卡中输入
        output_type,core_time_step_divide_mars_time_step = self._read_core_mars_xml()
        output_type = self._examine_shared_output(['FuelPerformance','ReleaseRate'],core_control,output_type)
            
        # 核素
        nuclide = target_nuclide
        # 衰变常数
        decay_constant = self.preprocessor.extract_decay_constant(nuclide,file_path=self.file_path_decay_constant_default)
        # 扩散系数
        D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic = self._extract_diffusion_coef(nuclide,file_path=self.file_path_diffusion_coef_default)
        # 时间步的中值时刻
        time_sequence = [0.0] + list(accumulate(tracer.time_step_history))
        time_mid = np.array(time_sequence[0:-1])/2 + np.array(time_sequence[1:])/2
        # 温度
        temperature = (tracer.temperature_history,time_mid)
        # 盘存量
        inventory = (tracer.inventory_history_dict[target_nuclide],time_sequence)
        # 元件功率
        fission_power = np.array(tracer.power_history)*1e6
        decay_heat = np.array(tracer.decay_heat_history)*1e6
        element_power = (fission_power+decay_heat,time_mid) #[MW]->[W]
        # 燃耗
        burnup = (tracer.burnup_history,time_sequence)
        # 中子注量率
        neutron_flux = (tracer.neutron_flux_history,time_mid)
        # 事故开始时刻
        accident_time = core_control.accident_begin

        # 读取模拟时间与输出
        # 模拟时间段
        time = tracer.time_step_history
        # 模拟时间步长
        if core_time_step_divide_mars_time_step < 0:
            time_step = np.array(tracer.time_step_history) / (-core_time_step_divide_mars_time_step)
        else:
            time_step = np.ones_like(tracer.time_step_history) * core_time_step_divide_mars_time_step
        # 输出路径
        self.intermediate_dir = core_control.mars_intermediate_dir
        output_path = os.path.join(self.intermediate_dir,f'Tracer{tracer.idx}_{target_nuclide}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 输出间隔
        write_interval = [-1] * len(time_step)
        
        # 输出分布类型
        distribution_type = []
        # 输出分布时刻
        distribution_time = []

        self.external_conditions_dict['nuclide'] = nuclide
        self.external_conditions_dict['decay_constant'] = decay_constant
        self.external_conditions_dict['diffusion_coef']['D_element'] = [D_graphite_matrix,D_graphite_matrix]
        self.external_conditions_dict['diffusion_coef']['D_graphite_grain'] = [D_graphite_grain]
        self.external_conditions_dict['diffusion_coef']['D_particle'] = [D_kernel,D_buffer,D_pyc,D_sic,D_pyc]
        self.external_conditions_dict['diffusion_coef']['D_buffer'] = [D_buffer]
        self.external_conditions_dict['diffusion_coef']['D_pyc'] = [D_pyc]
        self.external_conditions_dict['diffusion_coef']['D_sic'] = [D_sic]
        self.external_conditions_dict['diffusion_coef']['D_kernel'] = [D_kernel]
        self.external_conditions_dict['diffusion_coef']['A_element'] = [A_graphite_matrix,A_graphite_matrix]
        self.external_conditions_dict['diffusion_coef']['A_graphite_grain'] = [A_graphite_grain]
        self.external_conditions_dict['diffusion_coef']['A_particle'] = [A_kernel,A_buffer,A_pyc,A_sic,A_pyc]
        self.external_conditions_dict['diffusion_coef']['A_buffer'] = [A_buffer]
        self.external_conditions_dict['diffusion_coef']['A_pyc'] = [A_pyc]
        self.external_conditions_dict['diffusion_coef']['A_sic'] = [A_sic]
        self.external_conditions_dict['diffusion_coef']['A_kernel'] = [A_kernel]
        self.external_conditions_dict['temperature'] = temperature
        self.external_conditions_dict['inventory'] = inventory
        self.external_conditions_dict['element_power'] = element_power
        self.external_conditions_dict['burnup'] = burnup
        self.external_conditions_dict['neutron_flux'] = neutron_flux
        self.external_conditions_dict['accident_time'] = accident_time

        self.control_dict['time'] = time
        self.control_dict['time_step'] = time_step
        self.control_dict['output_path'] = output_path
        self.control_dict['write_interval'] = write_interval
        self.control_dict['output_type'] = output_type
        self.control_dict['distribution_type'] = distribution_type
        self.control_dict['distribution_time'] = distribution_time
        self.control_dict['simulation_mode'] = f'tracer{tracer.idx}'

    def _read_mars_mars_configuration(self):
        '''
            读取独立使用MARS时的输入参数
        '''
        self.mars_mars_configuration = self.root.find('MARS-MARS')
        # 读取是否为HTRPM-FASR计算模式
        fsar_mode = self.mars_mars_configuration.find('HTRPMFSARMode')
        parent = fsar_mode
        option = 'value'
        default = self.control_dict_default['fsar_mode']
        self.control_dict['fsar_mode'] = self._read_option(parent,option,default)
  
        if self.control_dict['fsar_mode'] == 'off':
            # 读取强制条件
            self._read_mars_mars_external_conditions()
            # 读取模拟时间/输出
            self._read_mars_mars_control()
            # 独立计算模式标识
            self.control_dict['simulation_mode'] = 'independent' 
        else:
            # 读取FSAR模式的输入
            self._read_mars_fsar_mode_configuration()
            self.control_dict['simulation_mode'] = 'fsar'

    def _read_mars_fsar_mode_configuration(self):
        '''
            读取HTR-PM的FSAR计算模式的输入参数
        '''
        fsar_configuration = self.root.find('FSAR-MARS')
        
        # 核素
        parent = fsar_configuration
        element_name = 'Nuclide'
        option_name = 'value'
        data_type = 'str-array'
        default = self.fsar_mode_dict_default['nuclide_list']
        nuclide_list = self._read_element_option(parent,element_name,option_name,data_type=data_type,default=default)

        # 盘存量/衰变常数/扩散系数
        inventory_raw_list = self._read_element_option(fsar_configuration,'Inventory','value',data_type='float-array',default=self.fsar_mode_dict_default['inventory_raw_list'])
        inventory_unit = self._read_element_option(fsar_configuration,'Inventory','unit',default=self.fsar_mode_dict_default['inventory_unit'])
        inventory_dict = {}
        decay_constant_dict = {}
        diffusion_coef_dict = {} 
        for inventory_raw,nuclide in zip(inventory_raw_list,nuclide_list):
            decay_constant = self.preprocessor.extract_decay_constant(nuclide)
            D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic = self._extract_diffusion_coef(nuclide,file_path=self.file_path_diffusion_coef_default)
            inventory,_ = self.preprocessor.unit_convert(inventory_raw,inventory_unit,ACTIVITY,decay_constant=decay_constant)
            inventory_dict[nuclide] = inventory
            decay_constant_dict[nuclide] = decay_constant
            diffusion_coef_dict[nuclide] = [D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic]

        # 温度
        parent = fsar_configuration
        element_name = 'Temperature'
        default_unit = self.fsar_mode_dict_default['temperature_unit']
        target_unit = TEMPERATURE
        data_type = 'float-array'
        default = self.fsar_mode_dict_default['temperature_list']
        temperature_list = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default)
        
        # 温度份额
        parent = fsar_configuration
        element_name = 'TemperatureShare'
        option_name = 'value'
        data_type='float-array'
        default = self.fsar_mode_dict_default['temperature_share_list']
        temperature_share_list = self._read_element_option(parent,element_name,option_name,data_type=data_type,default=default)
        
        # 辐照时间
        parent = fsar_configuration
        element_name = 'IrradiationTime'
        default_unit = self.fsar_mode_dict_default['time_unit']
        target_unit = TIME
        data_type = 'float'
        default = self.fsar_mode_dict_default['time']
        time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default)        

        # 时间步长
        parent = fsar_configuration
        element_name = 'TimeStep'
        default_unit = self.fsar_mode_dict_default['time_unit']
        target_unit = TIME
        data_type = 'float'
        default = self.fsar_mode_dict_default['time_step']
        time_step = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default)        

        # 输出
        parent = fsar_configuration
        element_name = 'OutputPath'
        option_name = 'value'
        data_type = 'str'
        default = self.fsar_mode_dict_default['output_path']
        output_path = self._read_element_option(parent,element_name,option_name,data_type=data_type,default=default)

        # 输出间隔
        parent = fsar_configuration
        element_name = 'WriteInterval'
        option_name = 'value'                                 
        data_type = 'int'
        default_value = self.fsar_mode_dict_default['write_interval']  
        write_interval = self._read_element_option(parent,element_name,option_name,data_type=data_type,default=default_value)
        
        self.fsar_mode_dict['write_interval'] = write_interval
        self.fsar_mode_dict['nuclide_list'] = nuclide_list
        self.fsar_mode_dict['inventory_dict'] = inventory_dict
        self.fsar_mode_dict['decay_constant_dict'] = decay_constant_dict
        self.fsar_mode_dict['diffusion_coef_dict'] = diffusion_coef_dict
        self.fsar_mode_dict['temperature_list'] = temperature_list
        self.fsar_mode_dict['temperature_share_list'] = temperature_share_list
        self.fsar_mode_dict['time'] = time
        self.fsar_mode_dict['time_step'] = time_step
        self.fsar_mode_dict['output_path'] = output_path

        self.control_dict['time'] = [time]
        self.control_dict['time_step'] = [time_step]
        self.control_dict['write_interval'] = [write_interval]
        self.control_dict['output_type'] = ['ReleaseRate','Temperature','FuelPerformance','FPGeneration','Inventory','DiffusionCoef']
        self.control_dict['distribution_type'] = []
        self.control_dict['distribution_time'] = []
        
        self.external_conditions_dict['element_power'] = (np.array([0.0,0.0]),np.array([0.0,1e30]))
        self.external_conditions_dict['burnup'] = (np.array([0.0,0.0]),np.array([0.0,1e30]))
        self.external_conditions_dict['neutron_flux'] = (np.array([0.0,0.0]),np.array([0.0,1e30]))
        self.external_conditions_dict['accident_time'] = 1e30

         
        self.conditions_dict['c_element_environment'] = 0.0
        self.conditions_dict['c_particle_environment'] = 0.0
        self.conditions_dict['c_graphite_grain_environment'] = 0.0
        self.conditions_dict['c_kernel_environment'] = 0.0
        self.conditions_dict['element_initial_inventory'] = 0.0
        self.conditions_dict['initial_temperature'] = 1000
        

    def _read_mars_mars_external_conditions(self):
        '''
            读取MARS模拟时的强制条件
        '''
        
        simulation_conditions = self.mars_mars_configuration.find('SimulationConditions') if self.mars_mars_configuration is not None else None
        
        # 核素
        parent = simulation_conditions
        element_name = 'RequiredNuc'
        default_value = False
        option_name = 'name'
        self.external_conditions_dict['nuclide'] = self._read_element_option(parent,element_name,option_name,default=default_value)

        # 衰变常数
        nuc = simulation_conditions.find('RequiredNuc') if simulation_conditions is not None else None
        parent = nuc
        element_name = 'DecayConstant'
        default_unit = self.external_conditions_dict_default['decay_constant_unit']  
        target_unit = '/' + TIME                                     
        data_type = 'float'
        default_value = -1
        value = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        if value < 0:
            value = self.preprocessor.extract_decay_constant(self.external_conditions_dict['nuclide'],file_path=self.file_path_decay_constant_default)
        self.external_conditions_dict['decay_constant'] = value

        # 扩散系数
        diffusion_coef = nuc.find('DiffusionCoef') if nuc is not None else None
        parent = diffusion_coef
        element_name = 'GraphiteMatrixDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_graphite_matrix_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_graphite_matrix = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'GraphiteGrainDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_graphite_grain_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_graphite_grain = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'KernelDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_kernel_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_kernel = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'BufferDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_buffer_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_buffer = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'PyCDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_pyc_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_pyc = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'SiCDiffusionPreExponential'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['D_sic_unit']  
        target_unit = AREA + '/' + TIME                                     
        data_type = 'float-array'
        default_value = -1
        D_sic = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        
        element_name = 'GraphiteMatrixActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_graphite_matrix_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_graphite_matrix = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'GraphiteGrainActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_graphite_grain_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_graphite_grain = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'KernelActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_kernel_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_kernel = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'BufferActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_buffer_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_buffer = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'PyCActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_pyc_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_pyc = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        element_name = 'SiCActivationEnergy'
        default_unit = self.external_conditions_dict_default['diffusion_coef']['A_sic_unit']  
        target_unit = ENERGY + '/' + MOLECULUS                                     
        data_type = 'float-array'
        default_value = -1
        A_sic = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        diffusion_coef_list = [D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,
                               A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic]
        has_negative = any(np.any(np.array(d) < 0) for d in diffusion_coef_list)
        if has_negative:
            D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic = self._extract_diffusion_coef(self.external_conditions_dict['nuclide'],file_path=self.file_path_diffusion_coef_default)
            
        self.external_conditions_dict['diffusion_coef']['D_element'] = [D_graphite_matrix,D_graphite_matrix]
        self.external_conditions_dict['diffusion_coef']['D_graphite_grain'] = [D_graphite_grain]
        self.external_conditions_dict['diffusion_coef']['D_particle'] = [D_kernel,D_buffer,D_pyc,D_sic,D_pyc]
        self.external_conditions_dict['diffusion_coef']['D_buffer'] = [D_buffer]
        self.external_conditions_dict['diffusion_coef']['D_pyc'] = [D_pyc]
        self.external_conditions_dict['diffusion_coef']['D_sic'] = [D_sic]
        self.external_conditions_dict['diffusion_coef']['D_kernel'] = [D_kernel]
        self.external_conditions_dict['diffusion_coef']['A_element'] = [A_graphite_matrix,A_graphite_matrix]
        self.external_conditions_dict['diffusion_coef']['A_graphite_grain'] = [A_graphite_grain]
        self.external_conditions_dict['diffusion_coef']['A_particle'] = [A_kernel,A_buffer,A_pyc,A_sic,A_pyc]
        self.external_conditions_dict['diffusion_coef']['A_buffer'] = [A_buffer]
        self.external_conditions_dict['diffusion_coef']['A_pyc'] = [A_pyc]
        self.external_conditions_dict['diffusion_coef']['A_sic'] = [A_sic]
        self.external_conditions_dict['diffusion_coef']['A_kernel'] = [A_kernel]

        # 温度
        parent = simulation_conditions
        element_name = 'RequiredTemperature'
        default_unit = self.external_conditions_dict_default['temperature_unit']
        target_unit = TEMPERATURE
        data_type = 'time_series'
        default_value = False
        default_time = False
        value,time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time)
        self.external_conditions_dict['temperature'] = (value,time)

        # 盘存量
        parent = simulation_conditions
        element_name = 'RequiredInventory'
        default_unit = self.external_conditions_dict_default['inventory_unit']
        target_unit = ACTIVITY
        data_type = 'time_series'
        default_value = False
        default_time = False
        value,time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time,decay_constant=self.external_conditions_dict['decay_constant'])
        self.external_conditions_dict['inventory'] = (value,time)       

        # 元件功率
        parent = simulation_conditions
        element_name = 'ElementPower'
        default_unit = self.external_conditions_dict_default['element_power_unit']
        target_unit = ENERGY + '/' + TIME
        data_type = 'time_series'
        default_value = self.external_conditions_dict_default['element_power']
        default_time = [0.0,1e30]
        value,time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time,decay_constant=self.external_conditions_dict['decay_constant'])
        self.external_conditions_dict['element_power'] = (value,time)

        # 燃耗
        parent = simulation_conditions
        element_name = 'Burnup'
        default_unit = self.external_conditions_dict_default['burnup_unit']
        target_unit = '[GW]' + '[d]' + '/' + '[t]'
        data_type = 'time_series'
        default_value = self.external_conditions_dict_default['burnup']
        default_time = [0.0,86400,1e30]
        value,time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time,decay_constant=self.external_conditions_dict['decay_constant'])
        self.external_conditions_dict['burnup'] = (value,time)
        
        # 中子注量率
        parent = simulation_conditions
        element_name = 'NeutronFlux'
        default_unit = self.external_conditions_dict_default['neutron_flux_unit']
        target_unit = '[n]' + '/' + '[cm2]' + '[s]'
        data_type = 'time_series'
        default_value = self.external_conditions_dict_default['neutron_flux']
        default_time = [0.0,1e30]
        value,time = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time,decay_constant=self.external_conditions_dict['decay_constant'])
        self.external_conditions_dict['neutron_flux'] = (value,time)

        # 事故开始时间
        parent = simulation_conditions
        element_name = 'AccidentTime'
        default_unit = self.external_conditions_dict_default['accident_time_unit']
        target_unit = '[s]'
        data_type = 'float'
        default_value = self.external_conditions_dict_default['accident_time']
        value = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value=default_value,default_time=default_time,decay_constant=self.external_conditions_dict['decay_constant'])
        self.external_conditions_dict['accident_time'] = value

    def _read_mars_mars_control(self):
        '''
            读取MARS独立使用时的模拟时间与输出
        '''

        control = self.mars_mars_configuration.find('Control') if self.mars_mars_configuration is not None else None

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
        output = control.find('Output') if control is not None else None
        
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
        self.control_dict['write_interval'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        if 0 in self.control_dict['write_interval']:
            self.control_dict['write_interval'] = [-1 for _ in self.control_dict['time']]
           
        # 输出类型
        parent = output
        element_name = 'Type'
        option_name = 'value'
        data_type = 'str-array'
        default_value = self.control_dict_default['output_type']                                       
        self.control_dict['output_type'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 输出分布类型
        parent = output
        element_name = 'Distribution'
        option_name = 'value'
        data_type = 'str-array'
        default_value = self.control_dict_default['distribution_type']                                       
        self.control_dict['distribution_type'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)

        # 输出分布时刻
        parent = output
        element_name = 'Distribution'
        option_name = 'time'
        data_type = 'float-array'
        default_value = self.control_dict_default['distribution_time']                                       
        self.control_dict['distribution_time'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        
    def _read_shared_configuration(self):
        '''
            读取MARS/SUN的共享配置
        '''
        self.shared_configuration = self.root.find('MARS')

        # 读取燃料属性
        self._read_fuel_properties()

        # 读取模型参数
        self._read_models()

        # 读取边界/初始条件
        if self.control_dict['simulation_mode'] == 'independent' or 'tracer' in self.control_dict['simulation_mode']:
            self._read_conditions()

        # 读取求解模型
        self._read_solver()

    def _read_fuel_properties(self):
        '''
            燃料属性
        '''
        fuel_properties = self.shared_configuration.find('FuelProperties') if self.shared_configuration is not None else None
        geometry = fuel_properties.find('Geometry') if fuel_properties is not None else None
        uranium_contamination = fuel_properties.find('UraniumContaminationFraction') if fuel_properties is not None else None
        material = fuel_properties.find('MaterialProperties') if fuel_properties is not None else None
     
        #------------------------------------
        # Geometry 几何相关
        #------------------------------------
        # 元件半径:[燃料区,全球体]
        element_name = 'ElementRadius'
        default_unit = self.fuel_properties_dict_default['geometry']['r_element_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.fuel_properties_dict_default['geometry']['r_element']
        self.fuel_properties_dict['geometry']['r_element'] = self._read_element_option_value_unit_pair(geometry,element_name,default_unit,target_unit,data_type,default_value)
        self.fuel_properties_dict['geometry']['r_element'] = [0.0] + list(self.fuel_properties_dict['geometry']['r_element']) # 程序内需要以中心点为起点
        # 石墨晶粒半径:石墨晶粒半径
        element_name = 'GraphiteGrainRadius'
        default_unit = self.fuel_properties_dict_default['geometry']['r_graphite_grain_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.fuel_properties_dict_default['geometry']['r_graphite_grain']
        self.fuel_properties_dict['geometry']['r_graphite_grain'] = self._read_element_option_value_unit_pair(geometry,element_name,default_unit,target_unit,data_type,default_value)
        self.fuel_properties_dict['geometry']['r_graphite_grain'] = [0.0] + list(self.fuel_properties_dict['geometry']['r_graphite_grain'])
        # 颗粒半径:[核芯，缓冲层,IPyC,SiC,OPyC]
        element_name = 'TrisoParticleRadius'
        default_unit = self.fuel_properties_dict_default['geometry']['r_particle_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.fuel_properties_dict_default['geometry']['r_particle']
        self.fuel_properties_dict['geometry']['r_particle'] = self._read_element_option_value_unit_pair(geometry,element_name,default_unit,target_unit,data_type,default_value)
        self.fuel_properties_dict['geometry']['r_particle'] = [0.0] + list(self.fuel_properties_dict['geometry']['r_particle']) # 程序内需要以中心点为起点
        # 破损核芯半径
        self.fuel_properties_dict['geometry']['r_kernel'] = [0.0] + [list(self.fuel_properties_dict['geometry']['r_particle'])[1]]

        #------------------------------------
        # 颗粒数量
        #------------------------------------
        element_name = 'ParticleNumber'
        default_value = self.fuel_properties_dict_default['particle_number']
        option_name = 'value'
        data_type = 'int'
        self.fuel_properties_dict['particle_number'] = self._read_element_option(fuel_properties,element_name,option_name,default=default_value,data_type=data_type)

        #------------------------------------
        # UraniumContamination 铀污染
        #------------------------------------
        # 石墨基体孔隙铀污染
        element_name = 'GraphiteMatrixContamination'
        default_value = self.fuel_properties_dict_default['uranium_contamination']['uranium_contamination_element']
        option_name = 'value'
        data_type = 'float-array'
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_element'] = np.array(self._read_element_option(uranium_contamination,element_name,option_name,default=default_value,data_type=data_type))
        # 石墨晶粒铀污染
        element_name = 'GraphiteGrainContamination'
        default_value = self.fuel_properties_dict_default['uranium_contamination']['uranium_contamination_graphite_grain']
        option_name = 'value'
        data_type = 'float-array'
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_graphite_grain'] = np.array(self._read_element_option(uranium_contamination,element_name,option_name,default=default_value,data_type=data_type))
        # 颗粒涂层铀污染
        element_name = 'TrisoParticleContamination'
        default_value = self.fuel_properties_dict_default['uranium_contamination']['uranium_contamination_particle']
        option_name = 'value'
        data_type = 'float-array'
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'] = self._read_element_option(uranium_contamination,element_name,option_name,default=default_value,data_type=data_type)
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'][0] = 1 - sum(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'][1:]) - sum(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_graphite_grain']) - sum(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_element'])
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'] = np.array(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'])
        self.fuel_properties_dict['uranium_contamination']['uranium_contamination_kernel'] = np.array([self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle'][0]])
      
        #------------------------------------
        # MaterialProperties 材料固有属性
        #------------------------------------
        kernel = material.find('Kernel') if material is not None else None
        buffer = material.find('Buffer') if material is not None else None
        pyc = material.find('PyC') if material is not None else None
        sic = material.find('SiC') if material is not None else None
        graphite_matrix = material.find('GraphiteMatrix') if material is not None else None
        # 核芯属性/核芯摩尔质量
        element_name = 'KernelMolarMass'
        default_unit = self.fuel_properties_dict_default['material_properties']['molar_mass_kernel_unit']
        target_unit = MASS + '/' + MOLECULUS
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['molar_mass_kernel']
        self.fuel_properties_dict['material_properties']['molar_mass_kernel'] = self._read_element_option_value_unit_pair(kernel,element_name,default_unit,target_unit,data_type,default_value)
        # 核芯属性/核芯密度
        element_name = 'KernelDensity'
        default_unit = self.fuel_properties_dict_default['material_properties']['density_kernel_unit']
        target_unit = MASS + '/' + VOLUME
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['density_kernel']
        self.fuel_properties_dict['material_properties']['density_kernel'] = self._read_element_option_value_unit_pair(kernel,element_name,default_unit,target_unit,data_type,default_value)
        # 核芯属性/核芯体积
        self.fuel_properties_dict['material_properties']['molar_volume_kernel'] = self.fuel_properties_dict['material_properties']['molar_mass_kernel'] / self.fuel_properties_dict['material_properties']['density_kernel']
        # 缓冲层属性/缓冲层孔隙率
        element_name = 'BufferPorosity'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['porosity_buffer']
        self.fuel_properties_dict['material_properties']['porosity_buffer'] = self._read_element_option(buffer,element_name,option_name,default=default_value,data_type=data_type)
        # 缓冲层属性/缓冲层密度
        element_name = 'BufferDensity'
        default_unit = self.fuel_properties_dict_default['material_properties']['density_buffer_unit']
        target_unit = MASS + '/' + VOLUME
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['density_buffer']
        self.fuel_properties_dict['material_properties']['density_buffer'] = self._read_element_option_value_unit_pair(buffer,element_name,default_unit,target_unit,data_type,default_value)
        # PyC属性/PyC蠕变系数
        element_name = 'PyCCreepCoef'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['creep_coef_pyc']
        self.fuel_properties_dict['material_properties']['creep_coef_pyc'] = self._read_element_option(pyc,element_name,option_name,default=default_value,data_type=data_type)
        # PyC属性/PyC蠕变泊松比
        element_name = 'PyCCreepPoissonRatio'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['creep_poisson_ratio_pyc']
        self.fuel_properties_dict['material_properties']['creep_poisson_ratio_pyc'] = self._read_element_option(pyc,element_name,option_name,default=default_value,data_type=data_type)
        # PyC属性/PyC蠕变泊松比
        element_name = 'PyCPoissonRatio'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['poisson_ratio_pyc']
        self.fuel_properties_dict['material_properties']['poisson_ratio_pyc'] = self._read_element_option(pyc,element_name,option_name,default=default_value,data_type=data_type)
        # PyC属性/PyC密度
        element_name = 'PyCDensity'
        default_unit = self.fuel_properties_dict_default['material_properties']['density_unit_pyc']
        target_unit = MASS + '/' + VOLUME
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['density_pyc']
        self.fuel_properties_dict['material_properties']['density_pyc'] = self._read_element_option_value_unit_pair(pyc,element_name,default_unit,target_unit,data_type,default_value)
        # SiC属性/SiC密度
        element_name = 'SiCDensity'
        default_unit = self.fuel_properties_dict_default['material_properties']['density_unit_sic']
        target_unit = MASS + '/' + VOLUME
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['density_sic']
        self.fuel_properties_dict['material_properties']['density_sic'] = self._read_element_option_value_unit_pair(sic,element_name,default_unit,target_unit,data_type,default_value)
        # SiC属性/SiC抗拉强度
        element_name = 'SiCTensileStrength'
        default_unit = self.fuel_properties_dict_default['material_properties']['tensile_strength_unit_sic']
        target_unit = '[MPa]'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['tensile_strength_sic']
        self.fuel_properties_dict['material_properties']['tensile_strength_sic'] = self._read_element_option_value_unit_pair(sic,element_name,default_unit,target_unit,data_type,default_value)
        # SiC属性/SiC Weibull常数
        element_name = 'SiCWeibull'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['weibull_sic']
        self.fuel_properties_dict['material_properties']['weibull_sic'] = self._read_element_option(sic,element_name,option_name,default=default_value,data_type=data_type)
        # SiC属性/SiC热分解系数alpha
        element_name = 'SiCThermalDecompositionAlpha'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['sic_thermal_decomposition_alpha']
        self.fuel_properties_dict['material_properties']['sic_thermal_decomposition_alpha'] = self._read_element_option(sic,element_name,option_name,default=default_value,data_type=data_type)
        # SiC属性/SiC热分解系数beta
        element_name = 'SiCThermalDecompositionBeta'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['sic_thermal_decomposition_beta']
        self.fuel_properties_dict['material_properties']['sic_thermal_decomposition_beta'] = self._read_element_option(sic,element_name,option_name,default=default_value,data_type=data_type)
        # SiC属性/SiC制造破损率
        element_name = 'SiCManufacturingFailureFraction'
        option_name = 'value'
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['sic_manufacturing_failure_fraction']
        self.fuel_properties_dict['material_properties']['sic_manufacturing_failure_fraction'] = self._read_element_option(sic,element_name,option_name,default=default_value,data_type=data_type)
        # 基体石墨属性/基体石墨密度
        element_name = 'GraphiteMatrixDensity'
        default_unit = self.fuel_properties_dict_default['material_properties']['density_unit_graphite_matrix']
        target_unit = MASS + '/' + VOLUME
        data_type = 'float'
        default_value = self.fuel_properties_dict_default['material_properties']['density_graphite_matrix']
        self.fuel_properties_dict['material_properties']['density_graphite_matrix'] = self._read_element_option_value_unit_pair(graphite_matrix,element_name,default_unit,target_unit,data_type,default_value)

    def _read_models(self):
        '''
            读取模型及模型参数
        '''
        models = self.shared_configuration.find('Models') if self.shared_configuration is not None else None
        temperature = models.find('IntraPebbleTemperature') if models is not None else None
        recoil = models.find('Recoil') if models is not None else None
        fuel_performance = models.find('FuelPerformance') if models is not None else None
        adsorption = models.find('Adsorption') if models is not None else None
        mass_transfer = models.find('MassTransfer') if models is not None else None
        diffusion = models.find('Diffusion') if models is not None else None

        #------------------------------------
        # IntraPebbleTemperature 球内温度场
        #------------------------------------
        # 温度场模型
        parent = temperature
        option_name = 'mode'
        default_value = self.models_dict_default['intra_pebble_temperature']['temperature_field_model']
        self.models_dict['intra_pebble_temperature']['temperature_field_model'] = self._read_option(parent,option_name,default=default_value)
        # 石墨基体温度分区数
        parent = temperature
        element_name = 'GraphiteMatrixTemperatureZonesNumber'
        default_value = self.models_dict_default['intra_pebble_temperature']['temperature_zone_number']
        option_name = 'value'
        data_type = 'int'
        self.models_dict['intra_pebble_temperature']['temperature_zone_number'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # 瞬态温度场计算时间
        parent = temperature
        element_name = 'TransientTemperatureFieldBegin'
        default_unit = self.models_dict_default['intra_pebble_temperature']['transient_temperature_begin_unit']
        target_unit = TIME
        data_type = 'float'
        default_value = self.models_dict_default['intra_pebble_temperature']['transient_temperature_begin']
        self.models_dict['intra_pebble_temperature']['transient_temperature_begin'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)

        #------------------------------------
        # Recoil 反冲
        #------------------------------------
        # 反冲模型
        parent = recoil
        option_name = 'mode'
        default_value = self.models_dict_default['recoil']['recoil_model']
        self.models_dict['recoil']['recoil_model'] = self._read_option(parent,option_name,default=default_value)
        # 元件反冲半径
        parent = recoil
        element_name = 'ElementRecoilRadius'
        default_unit = self.models_dict_default['recoil']['r_recoil_element_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.models_dict_default['recoil']['r_recoil_element']
        self.models_dict['recoil']['r_recoil_element'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # 石墨晶粒反冲半径
        parent = recoil
        element_name = 'GraphiteGrainRecoilRadius'
        default_unit = self.models_dict_default['recoil']['r_recoil_graphite_grain_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.models_dict_default['recoil']['r_recoil_graphite_grain']
        self.models_dict['recoil']['r_recoil_graphite_grain'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # 颗粒反冲半径
        parent = recoil
        element_name = 'TrisoParticleRecoilRadius'
        default_unit = self.models_dict_default['recoil']['r_recoil_particle_unit']
        target_unit = LENGTH
        data_type = 'float-array'
        default_value = self.models_dict_default['recoil']['r_recoil_particle']
        self.models_dict['recoil']['r_recoil_particle'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # 破损核芯反冲半径
        self.models_dict['recoil']['r_recoil_kernel'] = np.array([self.models_dict['recoil']['r_recoil_particle'][0]])

        #------------------------------------
        # FuelPerformance 燃料性能
        #------------------------------------
        # 颗粒失效模型
        parent = fuel_performance
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['failure_model']
        self.models_dict['fuel_performance']['failure_model'] = self._read_option(parent,option_name,default=default_value)   
        # Weibull颗粒失效阈值
        parent = fuel_performance
        element_name = 'FailureThreshold'
        default_value = self.models_dict_default['fuel_performance']['failure_threshold']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['failure_threshold'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # 阶梯失效份额
        parent = fuel_performance
        element_name = 'StepFailureFractionIncrement'
        default_value = self.models_dict_default['fuel_performance']['step_failure_fraction_increment']
        option_name = 'value'
        data_type = 'float-array'
        failure_fraction = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        default_value = self.models_dict_default['fuel_performance']['step_failure_fraction_time']
        option_name = 'time'
        data_type = 'float-array'
        failure_fraction_time_raw = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        default_value = self.models_dict_default['fuel_performance']['step_failure_fraction_time_unit']
        option_name = 'time_unit'
        data_type = 'str'
        failure_fraction_time_unit = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        failure_fraction_time,_ = self.preprocessor.unit_convert(failure_fraction_time_raw,failure_fraction_time_unit,TIME)
        self.models_dict['fuel_performance']['step_failure_fraction_increment'] = failure_fraction
        self.models_dict['fuel_performance']['step_failure_fraction_time'] = failure_fraction_time
        # 压力
        pressure = fuel_performance.find('Pressure') if fuel_performance is not None else None
        redlich_kwong = pressure.find('RedlichKwong') if pressure is not None else None
        van_der_waals = pressure.find('VanDerWaals') if pressure is not None else None
        # 压力:压力模型
        parent = pressure
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['pressure_model']
        self.models_dict['fuel_performance']['pressure_model'] = self._read_option(parent,option_name,default=default_value)
        # 压力:RedlichKwong模型
        # TcCO
        parent = redlich_kwong
        element_name = 'TcCO'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_CO_unit']
        target_unit = TEMPERATURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_CO']
        self.models_dict['fuel_performance']['redlich_kwong']['Tc_CO'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # TcKr
        parent = redlich_kwong
        element_name = 'TcKr'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_Kr_unit']
        target_unit = TEMPERATURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_Kr']
        self.models_dict['fuel_performance']['redlich_kwong']['Tc_Kr'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # TcXe
        parent = redlich_kwong
        element_name = 'TcXe'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_Xe_unit']
        target_unit = TEMPERATURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Tc_Xe']
        self.models_dict['fuel_performance']['redlich_kwong']['Tc_Xe'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # PcCO
        parent = redlich_kwong
        element_name = 'PcCO'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_CO_unit']
        target_unit = PRESSURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_CO']
        self.models_dict['fuel_performance']['redlich_kwong']['Pc_CO'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # PcKr
        parent = redlich_kwong
        element_name = 'PcKr'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_Kr_unit']
        target_unit = PRESSURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_Kr']
        self.models_dict['fuel_performance']['redlich_kwong']['Pc_Kr'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # PcXe
        parent = redlich_kwong
        element_name = 'PcXe'
        default_unit = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_Xe_unit']
        target_unit = PRESSURE
        data_type = 'float'
        default_value = self.models_dict_default['fuel_performance']['redlich_kwong']['Pc_Xe']
        self.models_dict['fuel_performance']['redlich_kwong']['Pc_Xe'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value)
        # 压力:VanDerWaals模型
        # a_CO
        parent = van_der_waals
        element_name = 'ConstantACO'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['a_CO']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['a_CO'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # a_Kr
        parent = van_der_waals
        element_name = 'ConstantAKr'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['a_Kr']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['a_Kr'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # a_Xe
        parent = van_der_waals
        element_name = 'ConstantAXe'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['a_Xe']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['a_Xe'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # b_CO
        parent = van_der_waals
        element_name = 'ConstantBCO'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['b_CO']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['b_CO'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # b_Kr
        parent = van_der_waals
        element_name = 'ConstantBKr'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['b_Kr']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['b_Kr'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # b_Xe
        parent = van_der_waals
        element_name = 'ConstantBXe'
        default_value = self.models_dict_default['fuel_performance']['van_der_waals']['b_Xe']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['van_der_waals']['b_Xe'] = self._read_element_option(parent,element_name,option_name,default=default_value,data_type=data_type)
        # Xe产额
        parent = fuel_performance
        element_name = 'XeYield'
        default_value = self.models_dict_default['fuel_performance']['yield_Xe']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['yield_Xe'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # Kr产额
        parent = fuel_performance
        element_name = 'KrYield'
        default_value = self.models_dict_default['fuel_performance']['yield_Kr']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['yield_Kr'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 快中子份额
        parent = fuel_performance
        element_name = 'FastNeutronShare'
        default_value = self.models_dict_default['fuel_performance']['fast_neutron_share']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['fuel_performance']['fast_neutron_share'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 腐蚀 / 膨胀 / 蠕变 / 应力 / 晶间腐蚀模型
        fp_corrosion = fuel_performance.find('FPCorrosion') if fuel_performance is not None else None
        pyc_swelling = fuel_performance.find('PyCSwelling') if fuel_performance is not None else None
        creep_coef_pyc = fuel_performance.find('PyCCreepCoef') if fuel_performance is not None else None
        sic_stress = fuel_performance.find('SiCStress') if fuel_performance is not None else None
        intergranular_corrosion = fuel_performance.find('IntergranularCorrosion') if fuel_performance is not None else None
        # 腐蚀 
        parent = fp_corrosion
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['fp_corrosion_model']
        self.models_dict['fuel_performance']['fp_corrosion_model'] = self._read_option(parent, option_name, default=default_value)
        # 膨胀
        parent = pyc_swelling
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['pyc_swelling_model']
        self.models_dict['fuel_performance']['pyc_swelling_model'] = self._read_option(parent, option_name, default=default_value)
        # 蠕变
        parent = creep_coef_pyc
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['creep_coef_model_pyc']
        self.models_dict['fuel_performance']['creep_coef_model_pyc'] = self._read_option(parent, option_name, default=default_value)
        # 应力
        parent = sic_stress
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['sic_stress_model']
        self.models_dict['fuel_performance']['sic_stress_model'] = self._read_option(parent, option_name, default=default_value)
        # 晶间腐蚀模型
        parent = intergranular_corrosion
        option_name = 'mode'
        default_value = self.models_dict_default['fuel_performance']['intergranular_corrosion_model']
        self.models_dict['fuel_performance']['intergranular_corrosion_model'] = self._read_option(parent, option_name, default=default_value)

        #------------------------------------
        # 吸附
        #------------------------------------
        fresco_simplify = adsorption.find('FrescoSimplify')
        # H_A
        parent = fresco_simplify
        element_name = 'HenryConstantA'
        default_value = self.models_dict_default['adsorption']['henry_a']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['henry_a'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # H_B
        parent = fresco_simplify
        element_name = 'HenryConstantB'
        default_value = self.models_dict_default['adsorption']['henry_b']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['henry_b'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # F_A
        parent = fresco_simplify
        element_name = 'FreundlichConstantA'
        default_value = self.models_dict_default['adsorption']['freundlich_a']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['freundlich_a'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # F_B
        parent = fresco_simplify
        element_name = 'FreundlichConstantB'
        default_value = self.models_dict_default['adsorption']['freundlich_b']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['freundlich_b'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # F_E
        parent = fresco_simplify
        element_name = 'FreundlichConstantE'
        default_value = self.models_dict_default['adsorption']['freundlich_e']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['freundlich_e'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # F_F
        parent = fresco_simplify
        element_name = 'FreundlichConstantF'
        default_value = self.models_dict_default['adsorption']['freundlich_f']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['freundlich_f'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 转捩浓度
        parent = fresco_simplify
        element_name = 'ConvertConcentration'
        default_value = self.models_dict_default['adsorption']['c_convert']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['adsorption']['c_convert_unit']
        target_unit = MOLECULUS + '/' + MASS
        self.models_dict['adsorption']['c_convert'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # A_iso
        parent = adsorption
        element_name = 'A_iso'
        default_value = self.models_dict_default['adsorption']['A_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['A_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # B_iso
        parent = adsorption
        element_name = 'B_iso'
        default_value = self.models_dict_default['adsorption']['B_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['B_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # D_iso
        parent = adsorption
        element_name = 'D_iso'
        default_value = self.models_dict_default['adsorption']['D_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['D_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # E_iso
        parent = adsorption
        element_name = 'E_iso'
        default_value = self.models_dict_default['adsorption']['E_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['E_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # d1_iso
        parent = adsorption
        element_name = 'd1_iso'
        default_value = self.models_dict_default['adsorption']['d1_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['d1_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # d2_iso
        parent = adsorption
        element_name = 'd2_iso'
        default_value = self.models_dict_default['adsorption']['d2_iso']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['d2_iso'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 环境压力
        parent = adsorption
        element_name = 'EnvironmentPressure'
        default_value = self.models_dict_default['adsorption']['pressure_env']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['adsorption']['pressure_env_unit']
        target_unit = PRESSURE
        self.models_dict['adsorption']['pressure_env'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 环境组分
        parent = adsorption
        element_name = 'EnvironmentComponent'
        default_value = self.models_dict_default['adsorption']['component_env']
        option_name = 'value'
        data_type = 'str-array'
        self.models_dict['adsorption']['component_env'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 环境摩尔份额
        parent = adsorption
        element_name = 'ComponentMolarFraction'
        default_value = self.models_dict_default['adsorption']['mole_fraction_env']
        option_name = 'value'
        data_type = 'float-array'
        self.models_dict['adsorption']['mole_fraction_env'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 环境摩尔质量
        parent = adsorption
        element_name = 'EnvironmentMolarMass'
        default_value = self.models_dict_default['adsorption']['mole_mass_env']
        option_name = 'value'
        data_type = 'float-array'
        default_unit = self.models_dict_default['adsorption']['mole_mass_env_unit']
        target_unit = '[kg]/[mol]'
        self.models_dict['adsorption']['mole_mass_env'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 环境动力粘度
        parent = adsorption
        element_name = 'DynamicViscosity'
        default_value = self.models_dict_default['adsorption']['dynamic_viscosity_model']
        option_name = 'mode'
        data_type = 'str'
        self.models_dict['adsorption']['dynamic_viscosity_model'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 环境密度
        parent = adsorption
        element_name = 'CoolantDensity'
        default_value = self.models_dict_default['adsorption']['coolant_density_model']
        option_name = 'mode'
        data_type = 'str'
        self.models_dict['adsorption']['coolant_density_model'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 环境/核素二元扩散系数
        parent = adsorption
        element_name = 'BinaryDiffusionCoef'
        default_value = self.models_dict_default['adsorption']['binary_diffusion_coef_model']
        option_name = 'mode'
        data_type = 'str'
        self.models_dict['adsorption']['binary_diffusion_coef_model'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 冷却剂流速
        parent = adsorption
        element_name = 'CoolantVelocity'
        default_value = self.models_dict_default['adsorption']['coolant_velocity']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['adsorption']['coolant_velocity_unit']
        target_unit = '[m]/[s]'
        self.models_dict['adsorption']['coolant_velocity'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 堆芯孔隙度
        parent = adsorption
        element_name = 'CorePorosity'
        default_value = self.models_dict_default['adsorption']['epsilon']
        option_name = 'value'
        data_type = 'float'
        self.models_dict['adsorption']['epsilon'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)
        # 模式
        self.models_dict['adsorption']['mode'] = self._read_option(parent=adsorption,option_name='mode',default=self.models_dict_default['adsorption']['mode'])
        
        #------------------------------------
        # 传质
        #------------------------------------
        self.models_dict['mass_transfer']['mode'] = self._read_option(parent=mass_transfer,option_name='mode',default=self.models_dict_default['mass_transfer']['mode'])
        # 元件传质系数
        parent = mass_transfer
        element_name = 'ElementMassTransferCoef'
        default_value = self.models_dict_default['mass_transfer']['element_mass_transfer_coef']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['mass_transfer']['element_mass_transfer_coef_unit']
        target_unit = LENGTH + '/' + TIME
        self.models_dict['mass_transfer']['element_mass_transfer_coef'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 石墨晶粒传质系数
        parent = mass_transfer
        element_name = 'GraphiteGrainMassTransferCoef'
        default_value = self.models_dict_default['mass_transfer']['graphite_grain_mass_transfer_coef']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['mass_transfer']['graphite_grain_mass_transfer_coef_unit']
        target_unit = LENGTH + '/' + TIME
        self.models_dict['mass_transfer']['graphite_grain_mass_transfer_coef'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 完整颗粒传质系数
        parent = mass_transfer
        element_name = 'TrisoParticleMassTransferCoef'
        default_value = self.models_dict_default['mass_transfer']['particle_mass_transfer_coef']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['mass_transfer']['particle_mass_transfer_coef_unit']
        target_unit = LENGTH + '/' + TIME
        self.models_dict['mass_transfer']['particle_mass_transfer_coef'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)
        # 破损核心传质系数
        parent = mass_transfer
        element_name = 'FailureKernelMassTransferCoef'
        default_value = self.models_dict_default['mass_transfer']['kernel_mass_transfer_coef']
        option_name = 'value'
        data_type = 'float'
        default_unit = self.models_dict_default['mass_transfer']['particle_mass_transfer_coef_unit']
        target_unit = LENGTH + '/' + TIME
        self.models_dict['mass_transfer']['kernel_mass_transfer_coef'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,default_time=False,decay_constant=None)

        #------------------------------------
        # 扩散模型
        #------------------------------------
        parent = diffusion
        option_name = 'mode'
        default_value = self.models_dict_default['diffusion_model']
        self.models_dict['diffusion_model'] = self._read_option(parent, option_name, default=default_value)

    def _read_conditions(self):
        '''
            读取边界/初始条件
        '''
        conditions = self.shared_configuration.find('Conditions') if self.shared_configuration is not None else None
        boundary_condition = conditions.find('BoundaryCondition') if conditions is not None else None
        initial_condition = conditions.find('InitialCondition') if conditions is not None else None

        decay_constant = self.external_conditions_dict['decay_constant']
        
        # 元件环境浓度 ElementEnvironmentConcentration
        parent = boundary_condition
        element_name = 'ElementEnvironmentConcentration'
        default_unit = self.conditions_dict_default['c_element_environment_unit']  
        target_unit = ACTIVITY + '/' + VOLUME                                     
        data_type = 'float'
        default_value = self.conditions_dict_default['c_element_environment']
        self.conditions_dict['c_element_environment'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,decay_constant=decay_constant)

        # 颗粒环境浓度 ParticleEnvironmentConcentration
        parent = boundary_condition
        element_name = 'ParticleEnvironmentConcentration'
        default_unit = self.conditions_dict_default['c_particle_environment_unit'] 
        target_unit = ACTIVITY + '/' + VOLUME
        data_type = 'float'
        default_value = self.conditions_dict_default['c_particle_environment']
        self.conditions_dict['c_particle_environment'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,decay_constant=decay_constant)

        # 晶粒环境浓度
        self.conditions_dict['c_graphite_grain_environment'] = self.conditions_dict['c_particle_environment']
        
        # 破损核芯环境浓度
        self.conditions_dict['c_kernel_environment'] = self.conditions_dict['c_particle_environment']

        # 元件初始存量 ElementInitialInventory
        parent = initial_condition
        element_name = 'ElementInitialInventory'
        default_unit = self.conditions_dict_default['element_initial_inventory_unit'] 
        target_unit = ACTIVITY                                                       
        data_type = 'float'
        default_value = self.conditions_dict_default['element_initial_inventory']
        self.conditions_dict['element_initial_inventory'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,decay_constant=decay_constant)
        
        # 元件初始环境温度 InitialTemperature
        parent = initial_condition
        element_name = 'InitialTemperature'
        default_unit = self.conditions_dict_default['initial_temperature_unit'] 
        target_unit = TEMPERATURE                                                       
        data_type = 'float'
        default_value = self.conditions_dict_default['initial_temperature']
        self.conditions_dict['initial_temperature'] = self._read_element_option_value_unit_pair(parent,element_name,default_unit,target_unit,data_type,default_value,decay_constant=decay_constant)

    def _read_solver(self):
        '''
            读取求解数值方法
        '''
        solver = self.shared_configuration.find('Solver') if self.shared_configuration is not None else None
        mesh = solver.find('Mesh') if solver is not None else None
        diffusion_solver = solver.find('DiffusionSolver') if solver is not None else None
        temperature_solver = solver.find('IntraPebbleTemperatureSolver') if solver is not None else None
        fuel_performance_solver = solver.find('FuelPerformanceSolver') if solver is not None else None
        failure_adjustment = solver.find('FailureAdjustment') if solver is not None else None

        #------------------------------------
        # Mesh 网格相关
        #------------------------------------
        parent = mesh
        element_name = 'ElementMeshNumber'
        option_name = 'value'
        data_type = 'int-array'
        default_value = self.solver_dict_default['mesh']['element_mesh_number']
        self.solver_dict['mesh']['element_mesh_number'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = mesh
        element_name = 'GraphiteGrainMeshNumber'
        option_name = 'value'
        data_type = 'int-array'
        default_value = self.solver_dict_default['mesh']['graphite_grain_mesh_number']
        self.solver_dict['mesh']['graphite_grain_mesh_number'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = mesh
        element_name = 'TrisoParticleMeshNumber'
        option_name = 'value'
        data_type = 'int-array'
        default_value = self.solver_dict_default['mesh']['particle_mesh_number']
        self.solver_dict['mesh']['particle_mesh_number'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        self.solver_dict['mesh']['kernel_mesh_number'] = [self.solver_dict['mesh']['particle_mesh_number'][0]]
        
        # 网格偏置
        parent = mesh
        element_name = 'ElementMeshBias'
        option_name = 'value'
        data_type = 'float-array'
        default_value = self.solver_dict_default['mesh']['element_mesh_bias']
        self.solver_dict['mesh']['element_mesh_bias'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = mesh
        element_name = 'GraphiteGrainMeshBias'
        option_name = 'value'
        data_type = 'float-array'
        default_value = self.solver_dict_default['mesh']['graphite_grain_mesh_bias']
        self.solver_dict['mesh']['graphite_grain_mesh_bias'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = mesh
        element_name = 'TrisoParticleMeshBias'
        option_name = 'value'
        data_type = 'float-array'
        default_value = self.solver_dict_default['mesh']['particle_mesh_bias']
        self.solver_dict['mesh']['particle_mesh_bias'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        self.solver_dict['mesh']['kernel_mesh_bias'] = [self.solver_dict['mesh']['particle_mesh_bias'][0]]

        #------------------------------------
        # DiffusionSolver 扩散求解器
        #------------------------------------
        # Crank-Nicolson 时间权重
        parent = diffusion_solver
        option_name = 'time_weight'
        default_value = self.solver_dict_default['diffusion_solver']['crank_nicolson_time_weight']
        time_weight_raw = self._read_option(parent, option_name, default=default_value)
        self.solver_dict['diffusion_solver']['crank_nicolson_time_weight'] = float(time_weight_raw)

        # 各区域扩散求解模式
        element = diffusion_solver.find('ElementDiffusionSolver') if diffusion_solver is not None else None
        parent  = element
        option_name = 'mode'
        default_value = self.solver_dict_default['diffusion_solver']['element_diffusion_solver']
        self.solver_dict['diffusion_solver']['element_diffusion_solver'] = self._read_option(parent, option_name, default=default_value)

        element = diffusion_solver.find('GraphiteGrainDiffusionSolver') if diffusion_solver is not None else None
        parent = element
        option_name = 'mode'
        default_value = self.solver_dict_default['diffusion_solver']['graphite_grain_diffusion_solver']
        self.solver_dict['diffusion_solver']['graphite_grain_diffusion_solver'] = self._read_option(parent, option_name, default=default_value)

        element = diffusion_solver.find('TrisoParticleDiffusionSolver') if diffusion_solver is not None else None
        parent = element
        option_name = 'mode'
        default_value = self.solver_dict_default['diffusion_solver']['particle_diffusion_solver']
        self.solver_dict['diffusion_solver']['particle_diffusion_solver'] = self._read_option(parent, option_name, default=default_value)

        element = diffusion_solver.find('FailureKernelDiffusionSolver') if diffusion_solver is not None else None
        parent = element
        option_name = 'mode'
        default_value = self.solver_dict_default['diffusion_solver']['kernel_diffusion_solver']
        self.solver_dict['diffusion_solver']['kernel_diffusion_solver'] = self._read_option(parent, option_name, default=default_value)

        #------------------------------------
        # IntraPebbleTemperatureSolver 球内温度场求解器
        #------------------------------------
        parent = temperature_solver
        element_name = 'SteadyTemperatureFieldMaxIterations'
        option_name = 'value'
        data_type = 'int'
        default_value = self.solver_dict_default['steady_temperature_field_max_iterations']
        self.solver_dict['steady_temperature_field_max_iterations'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = temperature_solver
        element_name = 'SteadyTemperatureFieldResidual'
        option_name = 'value'
        data_type = 'float'
        default_value = self.solver_dict_default['steady_temperature_field_residual']
        self.solver_dict['steady_temperature_field_residual'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = temperature_solver
        element_name = 'TransientTemperatureFieldMaxIterations'
        option_name = 'value'
        data_type = 'int'
        default_value = self.solver_dict_default['transient_temperature_field_max_iterations']
        self.solver_dict['transient_temperature_field_max_iterations'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        parent = temperature_solver
        element_name = 'TransientTemperatureFieldResidual'
        option_name = 'value'
        data_type = 'float'
        default_value = self.solver_dict_default['transient_temperature_field_residual']
        self.solver_dict['transient_temperature_field_residual'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        #------------------------------------
        # FuelPerformanceSolver 燃料性能求解器
        #------------------------------------
        parent = fuel_performance_solver
        element_name = 'SteadyGasesReleaseFractionMaxIterations'
        option_name = 'value'
        data_type = 'int'
        default_value = self.solver_dict_default['steady_gases_release_fraction_max_iteration']
        self.solver_dict['steady_gases_release_fraction_max_iteration'] = self._read_element_option(parent, element_name, option_name,default=default_value, data_type=data_type)

        #------------------------------------
        # FailureAdjustment 失效调整
        #------------------------------------
        parent = failure_adjustment
        option_name = 'mode'
        default_value = self.solver_dict_default['failure_adjustment_model']
        self.solver_dict['failure_adjustment_model'] = self._read_option(parent, option_name, default=default_value)

        # split_num
        parent = failure_adjustment
        option_name = 'split_num'
        default_value = self.solver_dict_default['failure_adjust_split_number']
        split_num_raw = self._read_option(parent, option_name, default=default_value)
        self.solver_dict['failure_adjust_split_number'] = int(split_num_raw)

        # adjust_time_step_num
        parent = failure_adjustment
        option_name = 'adjust_time_step_num'
        default_value = self.solver_dict_default['failure_adjust_time_step_number']
        adjust_raw = self._read_option(parent, option_name, default=default_value)
        self.solver_dict['failure_adjust_time_step_number'] = int(adjust_raw)

    def _extract_diffusion_coef(self,nuc,temperature=1000,file_path=r'TOOL/DiffusionLib.dat'):
        '''
            根据核素名提取扩散系数
        '''
        D_list = []
        A_list = []
        def find_first_larger(numbers, target):
            """找到第一个大于 target 的下标，找不到则打印提示并返回 -1。"""
            for idx, val in enumerate(numbers):
                if val > target:
                    return idx
            print(
                "temperature has transend the limited value "
                f"nuc: {nuc} temperature: {temperature}"
            )
            return -1

        D_list = []
        A_list = []

        with open(file_path, "r") as f:
            lines_iter = iter(f)
            key_lines = None

            for line in lines_iter:
                nuc_field = line[9:14].strip()
                if nuc_field and (nuc_field in nuc):
                    key_lines = [line]
                    key_lines.extend(next(lines_iter, "") for _ in range(13))
                    break

            if not key_lines or len(key_lines) < 14:
                raise ValueError(
                    f"Cannot find diffusion coefficient block for nuc {nuc} "
                    f"in file {file_path}"
                )

            control_lines = key_lines[0::2]
            info_lines = key_lines[1::2]

            for i in range(7):
                ctrl = control_lines[i]
                info = info_lines[i]

                num_str = ctrl[40:57].strip()
                if not num_str:
                    raise ValueError(f"Empty group number in control line: {ctrl!r}")
                num = int(num_str)

                temp_region_str = ctrl[57:72].strip()
                if not temp_region_str:
                    index = 0
                else:
                    temperature_range = [float(v) for v in ctrl[58:].split()]
                    index = find_first_larger(temperature_range, temperature)

                D = np.empty(num, dtype=float)
                A = np.empty(num, dtype=float)

                base = 22 + 36 * index * num
                for k in range(num):
                    offset = base + 36 * k

                    d_str = info[offset : offset + 16].strip()
                    a_str = info[offset + 18 : offset + 18 + 16].strip()

                    D[k] = float(d_str)
                    A[k] = float(a_str)

                D_list.append(D.copy())
                A_list.append(A.copy())
            
            D_kernel = D_list[0]
            D_buffer = D_list[1]
            D_pyc = D_list[2]
            D_sic = D_list[3]
            D_graphite_matrix = D_list[5]
            D_graphite_grain = D_list[6]

            A_kernel = A_list[0]
            A_buffer = A_list[1]
            A_pyc = A_list[2]
            A_sic = A_list[3]
            A_graphite_matrix = A_list[5]
            A_graphite_grain = A_list[6]
          
            return D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic
    
    def assert_no_none_attrib(self,root):
        for elem in root.iter():
            for k, v in elem.attrib.items():
                if v is None:
                    raise ValueError(f"XML attribute None: <{elem.tag} {k}=None>, attrib={elem.attrib}")

    def _write_completed_xml(self,tracer=None):
        '''
            保存用于最终计算的输入卡
        '''
        def leaf(parent, tag, **attrib):
            """创建一个只有属性、没有子节点的元素"""
            return ET.SubElement(parent, tag, attrib)
        
        def to_xml_value(x):
            if x is None:
                raise ValueError("to_xml_value got None")

            if isinstance(x, str):
                return x
            try:
                itr = list(x)
            except TypeError:
                return str(x)
            if len(itr) == 1:
                if itr[0] is None:
                    raise ValueError("to_xml_value got [None]")
                return str(itr[0])

            if any(v is None for v in itr):
                raise ValueError("to_xml_value got iterable containing None")

            return " ".join(str(v) for v in itr)
        
        # 1. 根节点
        root = ET.Element("COSMOS")

        # --------------------------------------------------
        # 2. MARS
        # --------------------------------------------------
        mars = ET.SubElement(root, "MARS")

        fuel_props = ET.SubElement(mars, "FuelProperties")
        geometry = ET.SubElement(fuel_props, "Geometry")
        leaf(geometry, "ElementRadius", unit=LENGTH, value=to_xml_value(self.fuel_properties_dict['geometry']['r_element'][1:]))
        leaf(geometry, "GraphiteGrainRadius", unit=LENGTH, value=to_xml_value(self.fuel_properties_dict['geometry']['r_graphite_grain'][1:]))
        leaf(geometry, "TrisoParticleRadius", unit=LENGTH, value=to_xml_value(self.fuel_properties_dict['geometry']['r_particle'][1:]))

        leaf(fuel_props, "ParticleNumber", value=to_xml_value(self.fuel_properties_dict['particle_number']))

        ucf = ET.SubElement(fuel_props, "UraniumContaminationFraction")
        leaf(ucf, "GraphiteMatrixContamination", value=to_xml_value(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_element']))
        leaf(ucf, "GraphiteGrainContamination", value=to_xml_value(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_graphite_grain']))
        leaf(ucf, "TrisoParticleContamination", value=to_xml_value(self.fuel_properties_dict['uranium_contamination']['uranium_contamination_particle']))

        mat = ET.SubElement(fuel_props, "MaterialProperties")

        kernel = ET.SubElement(mat, "Kernel")
        leaf(kernel, "KernelMolarMass", unit=MASS+'/'+MOLECULUS, value=to_xml_value(self.fuel_properties_dict['material_properties']['molar_mass_kernel']))
        leaf(kernel, "KernelDensity", unit=MASS+'/'+VOLUME, value=to_xml_value(self.fuel_properties_dict['material_properties']['density_kernel']))

        buffer = ET.SubElement(mat, "Buffer")
        leaf(buffer, "BufferPorosity", value=to_xml_value(self.fuel_properties_dict['material_properties']['porosity_buffer']))
        leaf(buffer, "BufferDensity", unit=MASS+'/'+VOLUME, value=to_xml_value(self.fuel_properties_dict['material_properties']['density_buffer']))

        pyc = ET.SubElement(mat, "PyC")
        leaf(pyc, "PyCCreepCoef", value=to_xml_value(self.fuel_properties_dict['material_properties']['creep_coef_pyc']))
        leaf(pyc, "PyCCreepPoissonRatio", value=to_xml_value(self.fuel_properties_dict['material_properties']['creep_poisson_ratio_pyc']))
        leaf(pyc, "PyCPoissonRatio", value=to_xml_value(self.fuel_properties_dict['material_properties']['poisson_ratio_pyc']))
        leaf(pyc, "PyCDensity", unit=MASS+'/'+VOLUME, value=to_xml_value(self.fuel_properties_dict['material_properties']['density_pyc']))

        sic = ET.SubElement(mat, "SiC")
        leaf(sic, "SiCDensity", unit=MASS+'/'+VOLUME, value=to_xml_value(self.fuel_properties_dict['material_properties']['density_sic']))
        leaf(sic, "SiCTensileStrength", unit='[MPa]', value=to_xml_value(self.fuel_properties_dict['material_properties']['tensile_strength_sic']))
        leaf(sic, "SiCWeibull", value=to_xml_value(self.fuel_properties_dict['material_properties']['weibull_sic']))
        leaf(sic, "SiCThermalDecompositionAlpha", value=to_xml_value(self.fuel_properties_dict['material_properties']['sic_thermal_decomposition_alpha']))
        leaf(sic, "SiCThermalDecompositionBeta", value=to_xml_value(self.fuel_properties_dict['material_properties']['sic_thermal_decomposition_beta']))
        leaf(sic, "SiCManufacturingFailureFraction", value=to_xml_value(self.fuel_properties_dict['material_properties']['sic_manufacturing_failure_fraction']))

        gm = ET.SubElement(mat, "GraphiteMatrix")
        leaf(gm, "GraphiteMatrixDensity", unit=MASS+'/'+VOLUME, value=to_xml_value(self.fuel_properties_dict['material_properties']['density_graphite_matrix']))
        
        # ---------- Models ----------
        models = ET.SubElement(mars, "Models")

        ipt = ET.SubElement(models, "IntraPebbleTemperature", mode=to_xml_value(self.models_dict['intra_pebble_temperature']['temperature_field_model']))
        leaf(ipt, "GraphiteMatrixTemperatureZonesNumber", value=to_xml_value(self.models_dict['intra_pebble_temperature']['temperature_zone_number']))
        leaf(ipt, "TransientTemperatureFieldBegin", unit=TIME, value=to_xml_value(self.models_dict['intra_pebble_temperature']['transient_temperature_begin']))

        recoil = ET.SubElement(models, "Recoil", mode=to_xml_value(self.models_dict['recoil']['recoil_model']))
        leaf(recoil, "ElementRecoilRadius", unit=LENGTH, value=to_xml_value(self.models_dict['recoil']['r_recoil_element']))
        leaf(recoil, "GraphiteGrainRecoilRadius", unit=LENGTH, value=to_xml_value(self.models_dict['recoil']['r_recoil_graphite_grain']))
        leaf(recoil, "TrisoParticleRecoilRadius", unit=LENGTH, value=to_xml_value(self.models_dict['recoil']['r_recoil_particle']))

        fp = ET.SubElement(models, "FuelPerformance", mode=to_xml_value(self.models_dict['fuel_performance']['failure_model']))
        leaf(fp, "FailureThreshold", value=to_xml_value(self.models_dict['fuel_performance']['failure_threshold']))
        leaf(fp, "StepFailureFractionIncrement", value=to_xml_value(self.models_dict['fuel_performance']['step_failure_fraction_increment']), time_unit=TIME, time=to_xml_value(self.models_dict['fuel_performance']['step_failure_fraction_time']))

        press = ET.SubElement(fp, "Pressure", mode=to_xml_value(self.models_dict['fuel_performance']['pressure_model']))

        rk = ET.SubElement(press, "RedlichKwong")
        leaf(rk, "TcCO", unit=TEMPERATURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Tc_CO']))
        leaf(rk, "TcKr", unit=TEMPERATURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Tc_Kr']))
        leaf(rk, "TcXe", unit=TEMPERATURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Tc_Xe']))
        leaf(rk, "PcCO", unit=PRESSURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Pc_CO']))
        leaf(rk, "PcKr", unit=PRESSURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Pc_Kr']))
        leaf(rk, "PcXe", unit=PRESSURE, value=to_xml_value(self.models_dict['fuel_performance']['redlich_kwong']['Pc_Xe']))

        vdw = ET.SubElement(press, "VanDerWaals")
        leaf(vdw, "ConstantACO", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['a_CO']))
        leaf(vdw, "ConstantAKr", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['a_Kr']))
        leaf(vdw, "ConstantAXe", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['a_Xe']))
        leaf(vdw, "ConstantBCO", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['b_CO']))
        leaf(vdw, "ConstantBKr", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['b_Kr']))
        leaf(vdw, "ConstantBXe", value=to_xml_value(self.models_dict['fuel_performance']['van_der_waals']['b_Xe']))

        leaf(fp, "XeYield", value=to_xml_value(self.models_dict['fuel_performance']['yield_Xe']))
        leaf(fp, "KrYield", value=to_xml_value(self.models_dict['fuel_performance']['yield_Kr']))
        leaf(fp, "FastNeutronShare", value=to_xml_value(self.models_dict['fuel_performance']['fast_neutron_share']))
        ET.SubElement(fp, "FPCorrosion", mode=to_xml_value(self.models_dict['fuel_performance']['fp_corrosion_model']))
        ET.SubElement(fp, "PyCSwelling", mode=to_xml_value(self.models_dict['fuel_performance']['pyc_swelling_model']))
        ET.SubElement(fp, "PyCCreepCoef", mode=to_xml_value(self.models_dict['fuel_performance']['creep_coef_model_pyc']))
        ET.SubElement(fp, "SiCStress", mode=to_xml_value(self.models_dict['fuel_performance']['sic_stress_model']))
        ET.SubElement(fp, "IntergranularCorrosion", mode=to_xml_value(self.models_dict['fuel_performance']['intergranular_corrosion_model']))

        ads = ET.SubElement(models, "Adsorption",mode=to_xml_value(self.models_dict['adsorption']['mode']))
        fresco_simplify = ET.SubElement(ads, "FrescoSimplify")
        leaf(fresco_simplify, "HenryConstantA", value=to_xml_value(self.models_dict['adsorption']['henry_a']))
        leaf(fresco_simplify, "HenryConstantB", value=to_xml_value(self.models_dict['adsorption']['henry_b']))
        leaf(fresco_simplify, "FreundlichConstantA", value=to_xml_value(self.models_dict['adsorption']['freundlich_a']))
        leaf(fresco_simplify, "FreundlichConstantB", value=to_xml_value(self.models_dict['adsorption']['freundlich_b']))
        leaf(fresco_simplify, "FreundlichConstantE", value=to_xml_value(self.models_dict['adsorption']['freundlich_e']))
        leaf(fresco_simplify, "FreundlichConstantF", value=to_xml_value(self.models_dict['adsorption']['freundlich_f']))
        leaf(fresco_simplify, "ConvertConcentration", unit=MOLECULUS + '/' + MASS,value=to_xml_value(self.models_dict['adsorption']['c_convert']))
        leaf(ads,'A_iso',value=to_xml_value(self.models_dict['adsorption']['A_iso']))
        leaf(ads,'B_iso',value=to_xml_value(self.models_dict['adsorption']['B_iso']))
        leaf(ads,'D_iso',value=to_xml_value(self.models_dict['adsorption']['D_iso']))
        leaf(ads,'E_iso',value=to_xml_value(self.models_dict['adsorption']['E_iso']))
        leaf(ads,'d1_iso',value=to_xml_value(self.models_dict['adsorption']['d1_iso']))
        leaf(ads,'d2_iso',value=to_xml_value(self.models_dict['adsorption']['d2_iso']))
        leaf(ads,'EnvironmentPressure',unit=PRESSURE,value=to_xml_value(self.models_dict['adsorption']['pressure_env']))
        leaf(ads,'EnvironmentComponent',value=to_xml_value(self.models_dict['adsorption']['component_env']))
        leaf(ads,'EnvironmentMolarFraction',value=to_xml_value(self.models_dict['adsorption']['mole_fraction_env']))
        leaf(ads,'EnvironmentMolarMass',unit='[kg]/[mol]',value=to_xml_value(self.models_dict['adsorption']['mole_mass_env']))
        leaf(ads,'DynamicViscosity',value=to_xml_value(self.models_dict['adsorption']['dynamic_viscosity_model']))
        leaf(ads,'CoolantDensity',value=to_xml_value(self.models_dict['adsorption']['coolant_density_model']))
        leaf(ads,'BinaryDiffusionCoef',value=to_xml_value(self.models_dict['adsorption']['binary_diffusion_coef_model']))
        leaf(ads,'CoolantVelocity',unit='[m]/[s]',value=to_xml_value(self.models_dict['adsorption']['coolant_velocity']))
        leaf(ads,'CorePorosity',value=to_xml_value(self.models_dict['adsorption']['epsilon']))
        
        mt = ET.SubElement(models, "MassTransfer",mode=to_xml_value(self.models_dict['mass_transfer']['mode']))
        leaf(mt, "ElementMassTransferCoef", unit=LENGTH+'/'+TIME, value=to_xml_value(self.models_dict['mass_transfer']['element_mass_transfer_coef']))
        leaf(mt, "GraphiteGrainMassTransferCoef", unit=LENGTH+'/'+TIME, value=to_xml_value(self.models_dict['mass_transfer']['graphite_grain_mass_transfer_coef']))
        leaf(mt, "TrisoParticleMassTransferCoef", unit=LENGTH+'/'+TIME, value=to_xml_value(self.models_dict['mass_transfer']['particle_mass_transfer_coef']))
        leaf(mt, "FailureKernelMassTransferCoef", unit=LENGTH+'/'+TIME, value=to_xml_value(self.models_dict['mass_transfer']['kernel_mass_transfer_coef']))
        
        ET.SubElement(models, "Diffusion", mode=to_xml_value(self.models_dict['diffusion_model']))

        # ---------- Conditions ----------
        conds = ET.SubElement(mars, "Conditions")
        bc = ET.SubElement(conds, "BoundaryCondition")
        leaf(bc, "ElementEnvironmentConcentration", unit=ACTIVITY+'/'+VOLUME, value=to_xml_value(self.conditions_dict['c_element_environment']))
        leaf(bc, "ParticleEnvironmentConcentration", unit=ACTIVITY+'/'+VOLUME, value=to_xml_value(self.conditions_dict['c_particle_environment']))

        ic = ET.SubElement(conds, "InitialCondition")
        leaf(ic, "ElementInitialInventory", unit=ACTIVITY, value=to_xml_value(self.conditions_dict['element_initial_inventory']))
        leaf(ic, "InitialTemperature", unit=TEMPERATURE, value=to_xml_value(self.conditions_dict['initial_temperature']))
        
        # ---------- Solver ----------
        solver = ET.SubElement(mars, "Solver")

        mesh = ET.SubElement(solver, "Mesh")
        leaf(mesh, "ElementMeshNumber", value=to_xml_value(self.solver_dict['mesh']['element_mesh_number']))
        leaf(mesh, "GraphiteGrainMeshNumber", value=to_xml_value(self.solver_dict['mesh']['graphite_grain_mesh_number']))
        leaf(mesh, "TrisoParticleMeshNumber", value=to_xml_value(self.solver_dict['mesh']['particle_mesh_number']))
        leaf(mesh, "ElementMeshBias", value=to_xml_value(self.solver_dict['mesh']['element_mesh_bias']))
        leaf(mesh, "GraphiteGrainMeshBias", value=to_xml_value(self.solver_dict['mesh']['graphite_grain_mesh_bias']))
        leaf(mesh, "TrisoParticleMeshBias", value=to_xml_value(self.solver_dict['mesh']['particle_mesh_bias']))

        ds = ET.SubElement(solver, "DiffusionSolver", time_weight=to_xml_value(self.solver_dict['diffusion_solver']['crank_nicolson_time_weight']))
        ET.SubElement(ds, "ElementDiffusionSolver", mode=to_xml_value(self.solver_dict['diffusion_solver']['element_diffusion_solver']))
        ET.SubElement(ds, "GraphiteGrainDiffusionSolver", mode=to_xml_value(self.solver_dict['diffusion_solver']['graphite_grain_diffusion_solver']))
        ET.SubElement(ds, "TrisoParticleDiffusionSolver", mode=to_xml_value(self.solver_dict['diffusion_solver']['particle_diffusion_solver']))
        ET.SubElement(ds, "FailureKernelDiffusionSolver", mode=to_xml_value(self.solver_dict['diffusion_solver']['kernel_diffusion_solver']))

        ipts = ET.SubElement(solver, "IntraPebbleTemperatureSolver")
        leaf(ipts, "SteadyTemperatureFieldMaxIterations", value=to_xml_value(self.solver_dict['steady_temperature_field_max_iterations']))
        leaf(ipts, "SteadyTemperatureFieldResidual", value=to_xml_value(self.solver_dict['steady_temperature_field_residual']))
        leaf(ipts, "TransientTemperatureFieldMaxIterations", value=to_xml_value(self.solver_dict['transient_temperature_field_max_iterations']))
        leaf(ipts, "TransientTemperatureFieldResidual", value=to_xml_value(self.solver_dict['transient_temperature_field_residual']))

        fps = ET.SubElement(solver, "FuelPerformanceSolver")
        leaf(fps, "SteadyGasesReleaseFractionMaxIterations", value=to_xml_value(self.solver_dict['steady_gases_release_fraction_max_iteration']))

        ET.SubElement(solver, "FailureAdjustment", mode=to_xml_value(self.solver_dict['failure_adjustment_model']), split_num=to_xml_value(self.solver_dict['failure_adjust_split_number']), adjust_time_step_num=to_xml_value(self.solver_dict['failure_adjust_time_step_number']))

        # --------------------------------------------------
        # 3. MARS-MARS
        # --------------------------------------------------
        mars_mars = ET.SubElement(root, "MARS-MARS")
        leaf(mars_mars,'HTRPMFSARMode',value=self.control_dict['fsar_mode'])
        
        if self.control_dict['simulation_mode'] == 'independent' or 'tracer' in self.control_dict['simulation_mode']:
            control = ET.SubElement(mars_mars, "Control")
            leaf(control, "RequiredSimulationTime", unit=TIME, value=to_xml_value(self.control_dict['time']))
            leaf(control, "RequiredTimeStep", unit=TIME, value=to_xml_value(self.control_dict['time_step']))

            output = ET.SubElement(control, "Output")
            leaf(output, "OutputPath", value='OUTPUT/MARSOUTPUT')
            leaf(output, "WriteInterval", unit=TIME, value=to_xml_value(self.control_dict['write_interval']))
            leaf(output, "Type", value=to_xml_value(self.control_dict['output_type']))
            leaf(output, "Distribution", value=to_xml_value(self.control_dict['distribution_type']), time=to_xml_value(self.control_dict['distribution_time']))

            sim = ET.SubElement(mars_mars, "SimulationConditions")

            req_nuc = ET.SubElement(sim, "RequiredNuc", name=to_xml_value(self.external_conditions_dict['nuclide']))
            leaf(req_nuc, "DecayConstant", unit='/'+TIME, value=to_xml_value(self.external_conditions_dict['decay_constant']))
            
            diffcoef = ET.SubElement(req_nuc, "DiffusionCoef")
            leaf(diffcoef, "GraphiteMatrixDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_element'][0]))
            leaf(diffcoef, "GraphiteGrainDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_graphite_grain'][0]))
            leaf(diffcoef, "KernelDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_kernel'][0]))
            leaf(diffcoef, "BufferDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_buffer'][0]))
            leaf(diffcoef, "PyCDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_pyc'][0]))
            leaf(diffcoef, "SiCDiffusionPreExponential", unit=AREA+'/'+TIME, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['D_sic'][0]))
            leaf(diffcoef, "GraphiteMatrixActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_element'][0]))
            leaf(diffcoef, "GraphiteGrainActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_graphite_grain'][0]))
            leaf(diffcoef, "KernelActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_kernel'][0]))
            leaf(diffcoef, "BufferActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_buffer'][0]))
            leaf(diffcoef, "PyCActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_pyc'][0]))
            leaf(diffcoef, "SiCActivationEnergy", unit=ENERGY+'/'+MOLECULUS, value=to_xml_value(self.external_conditions_dict['diffusion_coef']['A_sic'][0]))

            leaf(sim, "RequiredTemperature", unit=TEMPERATURE, value=to_xml_value(self.external_conditions_dict['temperature'][0]), time_unit=TIME, time=to_xml_value(self.external_conditions_dict['temperature'][1]))
            leaf(sim, "RequiredInventory", unit=ACTIVITY, value=to_xml_value(self.external_conditions_dict['inventory'][0]), time_unit=TIME, time=to_xml_value(self.external_conditions_dict['inventory'][1]))
            leaf(sim, "ElementPower", unit=ENERGY+'/'+TIME, value=to_xml_value(self.external_conditions_dict['element_power'][0]), time_unit=TIME, time=to_xml_value(self.external_conditions_dict['element_power'][1]))
            leaf(sim, "Burnup", unit='[GW][d]/[t]', value=to_xml_value(self.external_conditions_dict['burnup'][0]), time_unit=TIME, time=to_xml_value(self.external_conditions_dict['burnup'][1]))
            leaf(sim, "NeutronFlux", unit='[n]/[cm2][s]', value=to_xml_value(self.external_conditions_dict['neutron_flux'][0]), time_unit=TIME, time=to_xml_value(self.external_conditions_dict['neutron_flux'][1]))
            leaf(sim, "AccidentTime", value=to_xml_value(self.external_conditions_dict['accident_time']), unit=TIME)

        # --------------------------------------------------
        # 4. SUN-MARS
        # --------------------------------------------------
        if 'tracer' in self.control_dict['simulation_mode']:
            core = ET.SubElement(root, "CORE-MARS")
            leaf(core,'CoreTimeStepDividMarsTimeStep',unit=TIME,value=to_xml_value(self.core_mars_dict['core_time_step_divid_mars_time_step']))
            leaf(core,'OutputType',value=to_xml_value(self.core_mars_dict['output_type']))
      
        # --------------------------------------------------
        # 4. FSAR-MARS
        # --------------------------------------------------
        if self.control_dict['simulation_mode'] == 'fsar':
            fsar = ET.SubElement(root,'FSAR-MARS')
            leaf(fsar,'Nuclide',value=to_xml_value(self.fsar_mode_dict['nuclide_list']))
            leaf(fsar,'Inventory',value=to_xml_value(list(self.fsar_mode_dict['inventory_dict'].values())),unit=ACTIVITY)
            leaf(fsar,'Temperature',value=to_xml_value(self.fsar_mode_dict['temperature_list']),unit=TEMPERATURE)
            leaf(fsar,'TemperatureShare',value=to_xml_value(self.fsar_mode_dict['temperature_share_list']))
            leaf(fsar,'IrradiationTime',value=to_xml_value(self.fsar_mode_dict['time']),unit=TIME)
            leaf(fsar,'TimeStep',value=to_xml_value(self.fsar_mode_dict['time_step']),unit=TIME)
            leaf(fsar,'OutputPath',value=to_xml_value(self.fsar_mode_dict['output_path']))
        # --------------------------------------------------
        # 写文件
        # --------------------------------------------------
        tree = ET.ElementTree(root)

        # Python 3.9+ 可以直接缩进美化
        ET.indent(tree, space="    ", level=0)
        if tracer is None:
            dirpath = os.path.dirname(self.xml_path)          
            basename = os.path.basename(self.xml_path)        
            completed_name = "completed_" + basename          
            completed_xml_path = os.path.join(dirpath,completed_name)
        else:
            nuc = self.external_conditions_dict['nuclide']
            completed_name = f'Tracer{tracer.idx}_{nuc}.xml'
            completed_xml_path = os.path.join(self.intermediate_dir,completed_name)
        self.assert_no_none_attrib(root)
        tree.write(completed_xml_path, encoding="utf-8", xml_declaration=True)

