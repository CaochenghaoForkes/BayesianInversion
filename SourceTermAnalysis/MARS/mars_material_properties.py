import numpy as np
from MARS.mars_xml_reader import MarsXMLReader
from TOOL.cosmos_general_function import Registry
import CoolProp.CoolProp as CP
import re

R = 8.314462618
NA = 6.02214076e23

class MarsMaterialProperties:
    '''
        材料属性
    '''
    class NuclideDiffusionCoef:
        '''
            核素扩散系数[cm2]/[s]
        '''
        def __init__(self,register:Registry,accident_begin):
            # 模型注册
            self.accident_begin = accident_begin
            register.register('nuclide_diffusion_coef','Arrhenius',self._arrhenius)
        
        def _arrhenius(self,temperature_num,mesh_number,D,A,category,time):
            '''
                Arrhenius关系式计算扩散系数
            '''
            diffusion_coef_num = np.zeros(np.sum(mesh_number))
            D_old = D[0]
            material_idx = 0
            for i in range(len(mesh_number)):
                if np.any(D_old != D[i]):
                    D_old = D[i]
                    material_idx += 1
                for A_,D_ in zip(A[i],D[i]):
                    diffusion_coef_num[np.sum(mesh_number[0:i]) : np.sum(mesh_number[0:i+1])] += D_*np.exp(-A_/R/temperature_num[np.sum(mesh_number[0:i]) : np.sum(mesh_number[0:i+1])])
                  
                            
            return diffusion_coef_num

    class ThermalConductivityKernel:
        '''
            核芯的导热系数[W]/[m][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_conductivity_kernel','Idaho2004',self._idaho_2004)

        def _idaho_2004(self,temperature_section):
            '''
                INEEL. Development of improved models and designs for coated-particle gas reactor fuels[R]. Idaho National Engineering and Environmental Laboratory, INEEL/EXT-05-02615,2004.
            '''
            T = temperature_section - 273.15
            part_1 = 0.0132*np.exp((1.88e-3)*temperature_section)
            part_2 = np.where(T < 1650, 4040/(464 + T), 1.9)
            thermal_conductivity_kernel_num = part_1 + part_2
            thermal_conductivity_kernel_num = np.ones_like(thermal_conductivity_kernel_num) * 3.46
            return thermal_conductivity_kernel_num
    
    class ThermalConductivityBuffer:
        '''
            缓冲层的导热系数[W]/[m][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_conductivity_buffer','Idaho2004',self._idaho_2004)

        def _idaho_2004(self,temperature_section):
            '''
                INEEL. Development of improved models and designs for coated-particle gas reactor fuels[R]. Idaho National Engineering and Environmental Laboratory, INEEL/EXT-05-02615,2004.
            '''
            thermal_conductivity_buffer_num = 0.5 * np.ones_like(temperature_section)
            thermal_conductivity_buffer_num = np.ones_like(thermal_conductivity_buffer_num) * 1.0
            return thermal_conductivity_buffer_num
    
    class ThermalConductivityPyC:
        '''
            PyC的导热系数[W]/[m][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_conductivity_pyc','Idaho2004',self._idaho_2004)

        def _idaho_2004(self,temperature_section):
            '''
                INEEL. Development of improved models and designs for coated-particle gas reactor fuels[R]. Idaho National Engineering and Environmental Laboratory, INEEL/EXT-05-02615,2004.
            '''
            thermal_conductivity_pyc_num = 4.0 * np.ones_like(temperature_section)
            
            return thermal_conductivity_pyc_num

    class ThermalConductivitySiC:
        '''
            SiC的导热系数[W]/[m][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_conductivity_sic','Idaho2004',self._idaho_2004)

        def _idaho_2004(self,temperature_section):
            '''
                INEEL. Development of improved models and designs for coated-particle gas reactor fuels[R]. Idaho National Engineering and Environmental Laboratory, INEEL/EXT-05-02615,2004.
            '''
            T = temperature_section - 273.15
            part_1 = 42.58
            part_2 = (1.5564e4) * np.power(T,-1)
            part_3 = (1.2977e7) * np.power(T,-2)
            part_4 = (1.8458e9) * np.power(T,-3)
            part_5 = (3.91112e-2) * np.exp((2.24732e-3)*T)
            thermal_conductivity_sic = (part_1-part_2+part_3-part_4) * part_5
            if any(thermal_conductivity_sic) <= 0:
                # 上面的SiC导热系数模型基于Price对CVD \beta-SiC实验的拟合结果,在室温下单晶SiC导热率约500W/(mK),自结合SiC约200W/(mK),超出模型范围的负的SiC导热系数取250平均值
                thermal_conductivity_sic = np.ones_like(thermal_conductivity_sic) * 250 
            thermal_conductivity_sic = np.ones_like(thermal_conductivity_sic) * 18.3
            return thermal_conductivity_sic
    
    class ThermalConductivityGraphiteMatrix:
        '''
            基体石墨的导热系数[W]/[m][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_conductivity_graphite_matrix','Idaho2004',self._idaho_2004)

        def _idaho_2004(self,temperature_section):
            '''
                INEEL. Development of improved models and designs for coated-particle gas reactor fuels[R]. Idaho National Engineering and Environmental Laboratory, INEEL/EXT-05-02615,2004.
            '''
            T = temperature_section - 273.15
            thermal_conductivity_graphite_matrix = 47.4 * (1 - (9.7556e-4) * (T + 173.15) * np.exp((-(9.7556e-4) * (T + 273.15))))
            thermal_conductivity_graphite_matrix = np.ones_like(thermal_conductivity_graphite_matrix) * 25
            return thermal_conductivity_graphite_matrix

    class HeatCapacityKernel:
        '''
            核芯的热容[J]/[kg][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('heat_capacity_kernel','Bison',self._bison)

        def _bison(self,temperature_section,molar_mass_kernel):
            '''
                Jiang, W., Hales, J. D., Spencer, B. W., Collin, B. P., Slaughter, A. E., Novascone, S. R., Toptan, A., Gamble, K. A., & Gardner, R. (2021). TRISO particle fuel performance and failure analysis with BISON. Journal of Nuclear Materials, 548, 152795. https://doi.org/10.1016/j.jnucmat.2021.152795
            '''
            molar_mass_kernel /= 1000.0 # [g]/[mol] -> [kg]/[mol]
            t = np.copy(temperature_section/1000.0)
            heat_capacity_kernel = (52.1743 + 87.951*t - 84.2411*t**2 + 31.542*t**3 - 2.6334*t**4 - 0.71391/(t**2))
            heat_capacity_kernel /= molar_mass_kernel

            return heat_capacity_kernel
    
    class HeatCapacityBuffer:
        '''
            缓冲层的热容[J]/[kg][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('heat_capacity_buffer','Bison',self._bison)

        def _bison(self,temperature_section):
            '''
                Jiang, W., Hales, J. D., Spencer, B. W., Collin, B. P., Slaughter, A. E., Novascone, S. R., Toptan, A., Gamble, K. A., & Gardner, R. (2021). TRISO particle fuel performance and failure analysis with BISON. Journal of Nuclear Materials, 548, 152795. https://doi.org/10.1016/j.jnucmat.2021.152795
            '''
            heat_capacity_kernel = np.ones_like(temperature_section) * 720

            return heat_capacity_kernel
    
    class HeatCapacityPyC:
        '''
            PyC的热容[J]/[kg][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('heat_capacity_pyc','Bison',self._bison)

        def _bison(self,temperature_section):
            '''
                Jiang, W., Hales, J. D., Spencer, B. W., Collin, B. P., Slaughter, A. E., Novascone, S. R., Toptan, A., Gamble, K. A., & Gardner, R. (2021). TRISO particle fuel performance and failure analysis with BISON. Journal of Nuclear Materials, 548, 152795. https://doi.org/10.1016/j.jnucmat.2021.152795
            '''
            heat_capacity_pyc = np.ones_like(temperature_section) * 720

            return heat_capacity_pyc
    
    class HeatCapacitySiC:
        '''
            SiC的热容[J]/[kg][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('heat_capacity_sic','Bison',self._bison)

        def _bison(self,temperature_section):
            '''
                Jiang, W., Hales, J. D., Spencer, B. W., Collin, B. P., Slaughter, A. E., Novascone, S. R., Toptan, A., Gamble, K. A., & Gardner, R. (2021). TRISO particle fuel performance and failure analysis with BISON. Journal of Nuclear Materials, 548, 152795. https://doi.org/10.1016/j.jnucmat.2021.152795
            '''
            heat_capacity_sic = 925.65 + 0.3772*temperature_section - (7.9259e-5)*(temperature_section**2) - (3.1946e7)/(temperature_section**2)

            return heat_capacity_sic
    
    class HeatCapacityGraphiteMatrix:
        '''
            石墨基体的热容[J]/[kg][K]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('heat_capacity_graphite_matrix','Bison',self._bison)

        def _bison(self,temperature_section):
            '''
                Jiang, W., Hales, J. D., Spencer, B. W., Collin, B. P., Slaughter, A. E., Novascone, S. R., Toptan, A., Gamble, K. A., & Gardner, R. (2021). TRISO particle fuel performance and failure analysis with BISON. Journal of Nuclear Materials, 548, 152795. https://doi.org/10.1016/j.jnucmat.2021.152795
            '''
            heat_capacity_graphite_matrix = np.ones_like(temperature_section) * 720

            return heat_capacity_graphite_matrix

    class DensityKernel:
        '''
            核芯的密度[g]/[cm3]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('density_kernel','ExternalValue',self._external_value)
        
        def _external_value(self,external_density_kernel):
            '''
                用户指定的恒定密度
            '''
            density_kernel = external_density_kernel
            
            return density_kernel
    
    class DensityBuffer:
        '''
            缓冲层的密度[g]/[cm3]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('density_buffer','ExternalValue',self._external_value)
        
        def _external_value(self,external_density_buffer):
            '''
                用户指定的恒定密度
            '''
            density_buffer = external_density_buffer
            
            return density_buffer
    
    class DensityPyC:
        '''
            PyC的密度[g]/[cm3]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('density_pyc','ExternalValue',self._external_value)
        
        def _external_value(self,external_density_pyc):
            '''
                用户指定的恒定密度
            '''
            density_pyc = external_density_pyc
            
            return density_pyc
    
    class DensitySiC:
        '''
            SiC的密度[g]/[cm3]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('density_sic','ExternalValue',self._external_value)
        
        def _external_value(self,external_density_sic):
            '''
                用户指定的恒定密度
            '''
            density_sic = external_density_sic
            
            return density_sic
    
    class DensityGraphiteMatrix:
        '''
            石墨基体的密度[g]/[cm3]
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('density_graphite_matrix','ExternalValue',self._external_value)
        
        def _external_value(self,external_density_graphite_matrix):
            '''
                用户指定的恒定密度
            '''
            density_graphite_matrix = external_density_graphite_matrix
            
            return density_graphite_matrix

    class PyCSwellingRate:
        '''
            热解碳热膨胀率
        ''' 
        def __init__(self,register:Registry):
            # 模型注册
            register.register('pyc_swelling_rate','TecDoc1647-1',self._tec_doc_1647_1)
            register.register('pyc_swelling_rate','TecDoc1647-2',self._tec_doc_1647_2)
            register.register('pyc_swelling_rate','TecDoc1647-3',self._tec_doc_1647_3)

        def _tec_doc_1647_1(self,time_step,fast_neutron_flux,fast_neutron_flux_cumulant,sr_pyc_cumulant_old,st_pyc_cumulant_old):
            '''
                IAEA. Advances in High Temperature Gas Cooled Reactor Fuel Technology [R].IAEA-TECDOC-CD-1674, 2012.
            '''
            
            ar = [-2.22642e-2,2.00861e-2,-7.77024e-3,1.36334e-3,0,0]
            at = [-1.91253e-2,2.63307e-3,1.69251e-3,-3.53804e-4,0,0]

            fast_neutron_flux_cumulant_powers = (fast_neutron_flux_cumulant/1e25) ** np.arange(6)
            sr = np.dot(ar,fast_neutron_flux_cumulant_powers)
            st = np.dot(at,fast_neutron_flux_cumulant_powers)

            sr_pyc_cumulant_new = sr_pyc_cumulant_old + sr * (fast_neutron_flux/1e25) * time_step
            st_pyc_cumulant_new = st_pyc_cumulant_old + st * (fast_neutron_flux/1e25) * time_step

            return sr_pyc_cumulant_new,st_pyc_cumulant_new,sr,st
        
        def _tec_doc_1647_2(self,time_step,fast_neutron_flux,fast_neutron_flux_cumulant,sr_pyc_cumulant_old,st_pyc_cumulant_old):
            '''
                IAEA. Advances in High Temperature Gas Cooled Reactor Fuel Technology [R].IAEA-TECDOC-CD-1674, 2012.
            '''
            
            ar = [-2.12522e-2,1.83715e-2,-5.05553e-3,7.27026e-4,0,0]
            at = [-1.79113e-2,-3.42182e-3,5.03465e-3,-8.88086e-4,0,0]

            fast_neutron_flux_cumulant_powers = (fast_neutron_flux_cumulant/1e25) ** np.arange(6)
            sr = np.dot(ar,fast_neutron_flux_cumulant_powers)
            st = np.dot(at,fast_neutron_flux_cumulant_powers)

            sr_pyc_cumulant_new = sr_pyc_cumulant_old + sr * (fast_neutron_flux/1e25) * time_step
            st_pyc_cumulant_new = st_pyc_cumulant_old + st * (fast_neutron_flux/1e25) * time_step

            return sr_pyc_cumulant_new,st_pyc_cumulant_new,sr,st
        
        def _tec_doc_1647_3(self,time_step,fast_neutron_flux,fast_neutron_flux_cumulant,sr_pyc_cumulant_old,st_pyc_cumulant_old):
            '''
                IAEA. Advances in High Temperature Gas Cooled Reactor Fuel Technology [R].IAEA-TECDOC-CD-1674, 2012.
            '''
            
            ar = [-1.80613e-2,9.82884e-3,-2.25937e-3,4.03266e-4,0,0]
            at = [-1.78392e-2,1.71315e-3,2.32979e-3,-4.91648e-4,0,0]

            fast_neutron_flux_cumulant_powers = (fast_neutron_flux_cumulant/1e25) ** np.arange(6)
            sr = np.dot(ar,fast_neutron_flux_cumulant_powers)
            st = np.dot(at,fast_neutron_flux_cumulant_powers)

            sr_pyc_cumulant_new = sr_pyc_cumulant_old + sr * (fast_neutron_flux/1e25) * time_step
            st_pyc_cumulant_new = st_pyc_cumulant_old + st * (fast_neutron_flux/1e25) * time_step

            return sr_pyc_cumulant_new,st_pyc_cumulant_new,sr,st

    class PyCCreepCoef:
        '''
            热解碳蠕变系数
        '''     
        def __init__(self,register:Registry):
            # 模型注册
            register.register('pyc_creep_coef','ExternalValue',self._external_value)
            register.register('pyc_creep_coef','HoF',self._hof)
            register.register('pyc_creep_coef','TecDoc1647',self._tec_doc_1674)
        
        def _external_value(self,external_pyc_creep_coef):
            '''
                用户指定PyC蠕变系数
            '''
            pyc_creep_coef = external_pyc_creep_coef
            
            return pyc_creep_coef
        
        def _hof(self,temperature_ipyc,temperature_opyc,density_ipyc,density_opyc):
            '''
                Ho F. Material models of pyrocarbon and pyrolytic silicon carbide[R]. CEGA002820(1),1993.
            '''
            temperature_mean = (temperature_ipyc+temperature_opyc) / 2
            density_mean = (density_ipyc+density_opyc) / 2
            pyc_creep_coef = (1+2.38*(1.9-density_mean))*(4.0147e-10*temperature_mean**2 - 4.85e-7*temperature_mean + 2.193e-4)

            return pyc_creep_coef

        def _tec_doc_1674(self,temperature_ipyc,temperature_opyc):
            '''
                IAEA. Advances in High Temperature Gas Cooled Reactor Fuel Technology [R].IAEA-TECDOC-CD-1674, 2012.
            '''
            temperature_mean = (temperature_ipyc+temperature_opyc) / 2
            pyc_creep_coef = 4.386e-4 - 9.7e-7*temperature_mean + 8.0294e-10*temperature_mean**2

            return pyc_creep_coef

    class WeibullCoef:
        '''
            SiC的Weibull参数
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('weibull_coef','PANAMA',self._panama)
        
        def _panama(self,temperature_sic_irradiation,time,sic_tensile_strength_raw,sic_weibull_raw,fast_neutron_flux_cumulant,intergranular_corrosion):
            '''
                曹建主, 奚树人. PANAMA 程序及其在 10MW 高温气冷实验堆安全分析中的应用[J]. 核动力工程, 1998, 19(2): 162-167.
            '''
            gamma_s = np.power(10,(0.556+0.065*1e4/temperature_sic_irradiation))
            gamma_m = np.power(10,(0.394+0.065*1e4/temperature_sic_irradiation))
            sic_tensile_strength = sic_tensile_strength_raw * (1-(fast_neutron_flux_cumulant/1e25/gamma_s))
            sic_weibull = sic_weibull_raw * (1-(fast_neutron_flux_cumulant/1e25/gamma_m))
            sic_tensile_strength = max(196,sic_tensile_strength)
            sic_weibull = max(2,sic_weibull)
            # 晶间腐蚀
            if intergranular_corrosion == 'on':
                y = 0.565*np.exp(-187400/R/temperature_sic_irradiation)
                sic_weibull *= (0.44+0.56*np.exp(-y*time))
            
            return sic_weibull,sic_tensile_strength

    class AdsorbCoef:
        '''
            吸附解吸常数
        '''
        def __init__(self,register:Registry,decay_constant):
            # 模型注册
            register.register('adsorb_coef','FRESCO',self._fresco)
            register.register('adsorb_coef','Isotherm',self._isotherm)
            register.register('adsorb_coef','Disable',self._disable)
            self.decay_constant = decay_constant
            
        def _fresco(self,A_H,B_H,A_F,B_F,E_F,F_F,density_graphite_matrix,c_convert,temperature,c_fvm):
            '''
                吸附等温线计算分配系数(FRESCO-II简化)
            '''
            c = c_fvm[-1]
            c = c/self.decay_constant/NA # [mol]/[cm3]
            c_mass = c / density_graphite_matrix # [mol]/[g]
            c_mass *= 1e6
            c_convert *= 1e6
            if c_mass > c_convert:
                # Freundlich 模式
                x1 = A_F + B_F/temperature
                x2 = (E_F + F_F/temperature)*np.log(c_mass)
                adsorb_coef = np.exp(x1+x2)/temperature
                
            else:
                # Henry 模式
                adsorb_coef = np.exp(A_H+B_H/temperature) / temperature
            if adsorb_coef < 1e-15:
                adsorb_coef = 1e-15
          
            return adsorb_coef
        
        def _isotherm(self,A,B,D,E,d1,d2,density_graphite_matrix,temperature,c_fvm):
            '''
                吸附等温线
            '''
            c = c_fvm[-1] # [Bq]/[cm3]
            c = c/self.decay_constant/NA # [mol]/[cm3]
            c_mass = c / density_graphite_matrix * 1e6 # [umol]/[g]
            
            ln_ct = d1 - d2*temperature
            
            # Henry Pressure
            part_1 = A + B/temperature
            part_2 = D - 1 + E/temperature
            pressure_henry = c_mass * np.exp(part_1 + part_2*ln_ct)
            
            # Freundlich Pressure
            part_1 = A + B/temperature
            part_2 = D + E/temperature
            
            mask = np.power(c_mass,part_2) if c_mass != 0.0 else 0.0
            pressure_freundlich = np.exp(part_1) * mask
            
            pressure = pressure_henry + pressure_freundlich # [bar]
            
            c_boundary = pressure * 1e5 / (R * temperature) # [mol]/[m3]
            c_boundary = c_boundary / 1e6 # [mol]/[cm3]
            c_boundary = c_boundary * NA * self.decay_constant # [Bq]/[cm3]
            
            if c_fvm[-1] > 1e-18:
                adsorb_coef = c_boundary / c_fvm[-1]
            else:
                adsorb_coef = 1e-15
            if adsorb_coef < 1e-15:
                adsorb_coef = 1e-15
      
            return adsorb_coef
            
        def _disable(self):
            adsorb_coef = 1.0
            
            return adsorb_coef
    
    class DynamicViscosity:
        '''
            冷却剂的动力粘度
        '''
        def __init__(self,register:Registry):
            register.register('mu','CoolProp',self._coolprop)
            register.register('mu','Sutherland',self._sutherland)
            register.register('mu','GETTER',self._getter_for_he)

        def _herring_zipperer_mixtures(self,mole_fraction,mole_mass,dynamic_viscosity_list):
                # Y. S. Touloukian, S. C. Saxena, P. Hestermans, ThermophysicalPropertiesofMatterVolume II Viscosity, IFI/Plenum, 1975.
                part_1 = np.sum(dynamic_viscosity_list*np.array(mole_fraction)*np.sqrt(mole_mass))
                part_2 = np.sum(np.array(mole_fraction)*np.sqrt(mole_mass))
                
                return part_1 / part_2

        def _coolprop(self,pressure,temperature,component):
            
            P = pressure
            T_m = temperature
            comp = component
            mu = CP.PropsSI('VISCOSITY', 'P', P, 'T', T_m, comp)
            
            return mu

        def _sutherland(self,temperature,component,mole_fraction,mole_mass):
            
            # Sutherland  https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.27.html
            T_m = temperature
            comp = component
            mole_fraction = mole_fraction
            mole_mass = mole_mass
            surtherland_dict = {
                'Air':{'u0':1.716e-5,'T0v':273,'Su':111,'k0':0.0241,'T0k':273,'Sk':194},
                'Argon':{'u0':2.125e-5,'T0v':273,'Su':114,'k0':0.0163,'T0k':273,'Sk':170},
                'Nitrogen':{'u0':1.663e-5,'T0v':273,'Su':107,'k0':0.0242,'T0k':273,'Sk':150},
                'Water':{'u0':1.12e-5,'T0v':350,'Su':1064,'k0':0.0181,'T0k':300,'Sk':2200},
                'Oxygen':{'u0':1.919e-5,'T0v':273,'Su':139,'k0':0.0244,'T0k':273,'Sk':240},
                'CarbonMonoxide':{'u0':1.657e-5,'T0v':273,'Su':136,'k0':0.0232,'T0k':273,'Sk':180},
                'CO2':{'u0':1.37e-5,'T0v':273,'Su':222,'k0':0.0146,'T0k':273,'Sk':1800},
                'Hydrogen':{'u0':8.411e-5,'T0v':273,'Su':97,'k0':0.168,'T0k':273,'Sk':120},
                'Helium':{'u0':1.9571e-5,'T0v':293.15,'Su':73.8,'k0':0.168,'T0k':None,'Sk':None}
            }

            def sutherland(T_m,component):
                part_1 = np.power(T_m/surtherland_dict[component]['T0v'],3/2)
                part_2 = (surtherland_dict[component]['T0v']+surtherland_dict[component]['Su'])/(T_m+surtherland_dict[component]['Su'])
                dynamic_viscosity = surtherland_dict[component]['u0']*part_1*part_2
                return dynamic_viscosity 
            
            dynamic_viscosity_list = np.zeros(len(comp))
            for i in range(len(comp)):
                dynamic_viscosity_list[i] = sutherland(T_m,comp[i])

            mu = self._herring_zipperer_mixtures(mole_fraction,mole_mass,dynamic_viscosity_list)
            
            return mu

        def _getter_for_he(self,temperature):
            '''
                Reverse Engineering of GETTER A Fission Product Release Code for PBMR
            '''
            return 3.674e-7*np.power(temperature,0.7)

    class BinaryDiffusionCoef:
        '''
            FP在冷却剂中的二元扩散系数
        '''
        def __init__(self,register:Registry,nuclide):
            self.nuclide = nuclide
            match = re.search(r'(\d+)',nuclide)
            self.nuclide_mole_mass = int(match.group(1)) # [g]/[mol]
            register.register('binary_diffusion_coef','Chapman-Enskog',self._chapman_enskog)
            register.register('binary_diffusion_coef','SPATRA',self._spatra_simplify)
            register.register('binary_diffusion_coef','LiJian',self._lijian)
        
        def _chapman_enskog(self,temperature,pressure):
            '''
                SPATRA(Chapman-Enskog)(He)
            '''
            SigmaBDict = {'I':3.9,'Cs':4.0,'Sr':4.0,'Ag':4.0}
            for key in SigmaBDict:
                if key in self.nuclide:
                    SigmaB = SigmaBDict[key]
                    break
            
            EKB = 300.0
            EKBHEL = 10.2
            SigmaA = 2.576
            SigmaAB = (SigmaA+SigmaB)/2
            tao = temperature/(2*np.sqrt(EKB*EKBHEL))
            if tao > 1:
                omega = np.power(tao,-0.17)
            else:
                omega = np.power(tao,-0.52)        
            FAB = np.sqrt(1/4.0 + 1/self.nuclide_mole_mass)/(SigmaAB*omega)
            diffusion_coef_nuc = 1.8828e-2 * (temperature**1.5) * FAB / pressure
            
            return diffusion_coef_nuc # [m2]/[s]
        
        def _lijian(self,temperature,pressure):
            '''
                Li Jian
            '''
            SigmaBDict = {'I':3.9,'Cs':4.0,'Sr':4.0,'Ag':4.0}
            for key in SigmaBDict:
                if key in self.nuclide:
                    SigmaB = SigmaBDict[key]
                    break
            
            EKB = 300.0
            EKBHEL = 10.2
            SigmaA = 2.576
            SigmaAB = (SigmaA+SigmaB)/2
            tao = temperature/(2*np.sqrt(EKB*EKBHEL))
            if tao > 1:
                omega = np.power(tao,-0.17)
            else:
                omega = np.power(tao,-0.52)
            part_1 = 1.8583e-3*np.sqrt(1/4.0+1/self.nuclide_mole_mass)
            part_2 = np.sqrt(temperature**1.5)/(pressure * SigmaAB**2 * omega)
            diffusion_coef_nuc = part_1*part_2
            
            return diffusion_coef_nuc # [m2]/[s]
        
        def _spatra_simplify(self,temperature,pressure):
            '''
                SPATRA中的简化计算(He)
            '''
            diffusion_coef_nuc = (1e-3)*np.power(temperature,1.5)/pressure

            return diffusion_coef_nuc # [m2]/[s]

    class CoolantDensity:
        '''
            冷却剂密度
        '''
        def __init__(self,registry:Registry):
            registry.register('coolant_density','IdealGas',self._ideal_gas)
            registry.register('coolant_density','GETTER',self._getter_for_he)
            
        def _ideal_gas(self,pressure,temperature,mole_fraction,mole_mass):
            # 理想气体方程
            P = pressure
            T_m = temperature
            mole_fraction = mole_fraction
            mole_mass = mole_mass
            M_mix = np.sum(np.array(mole_fraction)*np.array(mole_mass))
            rho_m = P*M_mix/(R*T_m)
            
            return rho_m
        
        def _getter_for_he(self,temperature,pressure):
            '''
                Reverse Engineering of GETTER A Fission Product Release Code for PBMR
            '''
            pressure = pressure/1e5
            part_1 = 48.14*pressure/temperature
            part_2 = 1 + 0.4446*pressure/np.power(temperature,1.2)
            
            return part_1 * np.power(part_2,-1)
        
    def __init__(self,reader:MarsXMLReader,category):
        
        self.category = category
        ## 全部球体都具备的属性
        # 核素原始扩散系数
        self.D = reader.external_conditions_dict['diffusion_coef']['D_'+category]
        self.A = reader.external_conditions_dict['diffusion_coef']['A_'+category]
        self.decay_constant = reader.external_conditions_dict['decay_constant']
        self.nuclide = reader.external_conditions_dict['nuclide']
        # 材料的外部密度
        self.density_kernel_external = reader.fuel_properties_dict['material_properties']['density_kernel']
        self.density_buffer_external = reader.fuel_properties_dict['material_properties']['density_buffer']
        self.density_pyc_external = reader.fuel_properties_dict['material_properties']['density_pyc']
        self.density_sic_external = reader.fuel_properties_dict['material_properties']['density_sic']
        self.density_graphite_matrix_external = reader.fuel_properties_dict['material_properties']['density_graphite_matrix']
        # 吸附解吸系数
        self.A_H = reader.models_dict['adsorption']['henry_a']
        self.B_H = reader.models_dict['adsorption']['henry_b']
        self.A_F = reader.models_dict['adsorption']['freundlich_a']
        self.B_F = reader.models_dict['adsorption']['freundlich_b']
        self.E_F = reader.models_dict['adsorption']['freundlich_e']
        self.F_F = reader.models_dict['adsorption']['freundlich_f']
        self.c_convert = reader.models_dict['adsorption']['c_convert']
        self.A_iso = reader.models_dict['adsorption']['A_iso']
        self.B_iso = reader.models_dict['adsorption']['B_iso']
        self.D_iso = reader.models_dict['adsorption']['D_iso']
        self.E_iso = reader.models_dict['adsorption']['E_iso']
        self.d1_iso = reader.models_dict['adsorption']['d1_iso']
        self.d2_iso = reader.models_dict['adsorption']['d2_iso']
        # 环境流动介质
        self.pressure_env = reader.models_dict['adsorption']['pressure_env']
        self.component_env = reader.models_dict['adsorption']['component_env']
        self.mole_fraction_env = reader.models_dict['adsorption']['mole_fraction_env']
        self.mole_mass_env = reader.models_dict['adsorption']['mole_mass_env']
        self.coolant_velocity = reader.models_dict['adsorption']['coolant_velocity']
        self.epsilon = reader.models_dict['adsorption']['epsilon']

        ## 开启注册表
        self.register = Registry()
        # 注册扩散系数模型
        self.nuclide_diffusion_model = reader.models_dict['parameter']['nuclide_diffusion_coef']
        nuclide_diffusion_calculator = self.NuclideDiffusionCoef(self.register,reader.external_conditions_dict['accident_time'])
        # 注册核芯导热系数模型
        self.thermal_conductivity_kernel_model = reader.models_dict['parameter']['thermal_conductivity_kernel']
        thermal_conductivity_kernel_calculator = self.ThermalConductivityKernel(self.register)
        # 注册缓冲层导热系数模型
        self.thermal_conductivity_buffer_model = reader.models_dict['parameter']['thermal_conductivity_buffer']
        thermal_conductivity_buffer_calculator = self.ThermalConductivityBuffer(self.register)
        # 注册PyC导热系数模型
        self.thermal_conductivity_pyc_model = reader.models_dict['parameter']['thermal_conductivity_pyc']
        thermal_conductivity_pyc_calculator = self.ThermalConductivityPyC(self.register)
        # 注册SiC导热系数模型
        self.thermal_conductivity_sic_model = reader.models_dict['parameter']['thermal_conductivity_sic']
        thermal_conductivity_sic_calculator = self.ThermalConductivitySiC(self.register)
        # 注册石墨基体导热系数模型
        self.thermal_conductivity_graphite_matrix_model = reader.models_dict['parameter']['thermal_conductivity_graphite_matrix']
        thermal_conductivity_graphite_matrix_calculator = self.ThermalConductivityGraphiteMatrix(self.register)
        # 注册核芯热容模型
        self.heat_capacity_kernel_model = reader.models_dict['parameter']['heat_capacity_kernel']
        heat_capacity_kernel_calculator = self.HeatCapacityKernel(self.register)
        # 注册缓冲层热容模型
        self.heat_capacity_buffer_model = reader.models_dict['parameter']['heat_capacity_buffer']
        heat_capacity_buffer_calculator = self.HeatCapacityBuffer(self.register)
        # 注册PyC热容模型
        self.heat_capacity_pyc_model = reader.models_dict['parameter']['heat_capacity_pyc']
        heat_capacity_pyc_calculator = self.HeatCapacityPyC(self.register)
        # 注册SiC热容模型
        self.heat_capacity_sic_model = reader.models_dict['parameter']['heat_capacity_sic']
        heat_capacity_sic_calculator = self.HeatCapacitySiC(self.register)
        # 注册石墨基体热容模型
        self.heat_capacity_graphite_matrix_model = reader.models_dict['parameter']['heat_capacity_graphite_matrix']
        heat_capacity_graphite_matrix_calculator = self.HeatCapacityGraphiteMatrix(self.register)
        # 注册核芯密度模型
        self.density_kernel_model = reader.models_dict['parameter']['density_kernel']
        density_kernel_calculator = self.DensityKernel(self.register)
        # 注册缓冲层密度模型
        self.density_buffer_model = reader.models_dict['parameter']['density_buffer']
        density_buffer_calculator = self.DensityBuffer(self.register)
        # 注册PyC密度模型
        self.density_pyc_model = reader.models_dict['parameter']['density_pyc']
        density_pyc_calculator = self.DensityPyC(self.register)
        # 注册SiC密度模型
        self.density_sic_model = reader.models_dict['parameter']['density_sic']
        density_sic_calculator = self.DensitySiC(self.register)
        # 注册石墨基体密度模型
        self.density_graphite_matrix_model = reader.models_dict['parameter']['density_graphite_matrix']
        density_graphite_matrix_calculator = self.DensityGraphiteMatrix(self.register)
        # 注册PyC热膨胀率模型
        self.swelling_rate_pyc_model = reader.models_dict['fuel_performance']['pyc_swelling_model']
        swelling_rate_pyc_calculator = self.PyCSwellingRate(self.register)
        # 注册PyC蠕变系数模型
        self.creep_coef_pyc_model = reader.models_dict['fuel_performance']['creep_coef_model_pyc']
        creep_coef_pyc_calculator = self.PyCCreepCoef(self.register)
        # 注册SiC Weibull参数模型
        self.weibull_coef_sic_model = reader.models_dict['fuel_performance']['weibull_coef_model_sic']
        weibull_coef_sic_calculator = self.WeibullCoef(self.register)
        # 注册基体石墨的边界吸附系数
        self.adsorb_coef_model = reader.models_dict['adsorption']['mode']
        adsorb_coef_calculator = self.AdsorbCoef(self.register,self.decay_constant)
        # 注册动力粘度
        self.dynamic_viscosity_model = reader.models_dict['adsorption']['dynamic_viscosity_model']
        dynamic_viscosity_calculator = self.DynamicViscosity(self.register)
        # 注册二元扩散系数
        self.binary_diffusion_coef_model = reader.models_dict['adsorption']['binary_diffusion_coef_model']
        binary_diffusion_coef_calculator = self.BinaryDiffusionCoef(self.register,self.nuclide)
        # 注册冷却剂密度
        self.coolant_density_model = reader.models_dict['adsorption']['coolant_density_model']
        coolant_density_calculator = self.CoolantDensity(self.register)
        
    def calculate_nuclide_diffusion_coef(self,temperature_num,mesh_number,time):
        '''
            计算核素扩散系数
        '''
        return self.register.get('nuclide_diffusion_coef',self.nuclide_diffusion_model)(temperature_num,mesh_number,self.D,self.A,self.category,time)
    
    def calculate_thermal_conductivity_kernel(self,temperature_section):
        '''
            计算核芯导热系数
        '''
        return self.register.get('thermal_conductivity_kernel',self.thermal_conductivity_kernel_model)(temperature_section)
    
    def calculate_thermal_conductivity_buffer(self,temperature_section):
        '''
            计算缓冲层导热系数
        '''
        return self.register.get('thermal_conductivity_buffer',self.thermal_conductivity_buffer_model)(temperature_section)
    
    def calculate_thermal_conductivity_pyc(self,temperature_section):
        '''
            计算PyC导热系数
        '''
        return self.register.get('thermal_conductivity_pyc',self.thermal_conductivity_pyc_model)(temperature_section)
    
    def calculate_thermal_conductivity_sic(self,temperature_section):
        '''
            计算SiC导热系数
        '''
        return self.register.get('thermal_conductivity_sic',self.thermal_conductivity_sic_model)(temperature_section)
    
    def calculate_thermal_conductivity_graphite_matrix(self,temperature_section):
        '''
            计算石墨基体导热系数
        '''
        return self.register.get('thermal_conductivity_graphite_matrix',self.thermal_conductivity_graphite_matrix_model)(temperature_section)

    def calculate_heat_capacity_kernel(self,temperature_section,molar_mass_kernel):
        '''
            计算核芯热容
        '''
        return self.register.get('heat_capacity_kernel',self.heat_capacity_sic_model)(temperature_section,molar_mass_kernel)
    
    def calculate_heat_capacity_buffer(self,temperature_section):
        '''
            计算缓冲层热容
        '''
        return self.register.get('heat_capacity_buffer',self.heat_capacity_buffer_model)(temperature_section)
    
    def calculate_heat_capacity_pyc(self,temperature_section):
        '''
            计算PyC热容
        '''
        return self.register.get('heat_capacity_pyc',self.heat_capacity_pyc_model)(temperature_section)
    
    def calculate_heat_capacity_sic(self,temperature_section):
        '''
            计算SiC热容
        '''
        return self.register.get('heat_capacity_sic',self.heat_capacity_sic_model)(temperature_section)

    def calculate_heat_capacity_graphite_matrix(self,temperature_section):
        '''
            计算石墨基体热容
        '''
        return self.register.get('heat_capacity_graphite_matrix',self.heat_capacity_graphite_matrix_model)(temperature_section)
    
    def calculate_density_kernel(self):
        '''
            计算核芯密度
        '''
        return self.register.get('density_kernel',self.density_kernel_model)(self.density_kernel_external)
    
    def calculate_density_buffer(self):
        '''
            计算缓冲层密度
        '''
        return self.register.get('density_buffer',self.density_buffer_model)(self.density_buffer_external)
    
    def calculate_density_pyc(self):
        '''
            计算PyC密度
        '''
        return self.register.get('density_pyc',self.density_pyc_model)(self.density_pyc_external)
    
    def calculate_density_sic(self):
        '''
            计算SiC密度
        '''
        return self.register.get('density_sic',self.density_sic_model)(self.density_sic_external)

    def calculate_density_graphite_matrix(self):
        '''
            计算石墨基体密度
        '''
        return self.register.get('density_graphite_matrix',self.density_graphite_matrix_model)(self.density_graphite_matrix_external)
    
    def calculate_swelling_rate_pyc(self,time_step,fast_neutron_flux,fast_neutron_flux_cumulant,sr_pyc_cumulant_old,st_pyc_cumulant_old):
        '''
            计算热解碳的热膨胀系数
        '''
        return self.register.get('pyc_swelling_rate',self.swelling_rate_pyc_model)(time_step,fast_neutron_flux,fast_neutron_flux_cumulant,sr_pyc_cumulant_old,st_pyc_cumulant_old)

    def calculate_creep_coef_pyc(self,external_pyc_creep_coef,temperature_ipyc,temperature_opyc,density_ipyc,density_opyc):
        '''
            计算热解碳的蠕变系数
        '''
        if self.creep_coef_pyc_model == 'ExternalValue':
            return self.register.get('pyc_creep_coef',self.creep_coef_pyc_model)(external_pyc_creep_coef)
        elif self.creep_coef_pyc_model == 'HoF':
            return self.register.get('pyc_creep_coef',self.creep_coef_pyc_model)(temperature_ipyc,temperature_opyc,density_ipyc,density_opyc)
        elif self.creep_coef_pyc_model == 'TecDoc1647':
            return self.register.get('pyc_creep_coef',self.creep_coef_pyc_model)(temperature_ipyc,temperature_opyc)

    def calculate_weibull_coef_sic(self,temperature_sic_irradiation,time,sic_tensile_strength_raw,sic_weibull_raw,fast_neutron_flux_cumulant,intergranular_corrosion):
        '''
            SiC的Weibull参数
        '''
        return self.register.get('weibull_coef',self.weibull_coef_sic_model)(temperature_sic_irradiation,time,sic_tensile_strength_raw,sic_weibull_raw,fast_neutron_flux_cumulant,intergranular_corrosion)

    def calculate_adsorb_coef(self,temperature,c_fvm):
        '''
            计算吸附分配系数
        '''
        if self.adsorb_coef_model == 'FRESCO':
            return self.register.get('adsorb_coef',self.adsorb_coef_model)(self.A_H,self.B_H,self.A_F,self.B_F,self.E_F,self.F_F,self.density_graphite_matrix_external,self.c_convert,temperature,c_fvm)
        elif self.adsorb_coef_model == 'Isotherm':
            return self.register.get('adsorb_coef',self.adsorb_coef_model)(self.A_iso,self.B_iso,self.D_iso,self.E_iso,self.d1_iso,self.d2_iso,self.density_graphite_matrix_external,temperature,c_fvm)
        elif self.adsorb_coef_model == 'Disable':
            return self.register.get('adsorb_coef',self.adsorb_coef_model)()

    def _calculate_dynamic_viscosity(self,temperature):
        '''
            计算动力粘度
        '''
        if self.dynamic_viscosity_model == 'CoolProp':
            self.dynamic_viscosity = self.register.get('mu',self.dynamic_viscosity_model)(self.pressure_env,temperature,self.component_env)
        elif self.dynamic_viscosity_model == 'Sutherland':
            self.dynamic_viscosity = self.register.get('mu',self.dynamic_viscosity_model)(temperature,self.component_env,self.mole_fraction_env,self.mole_mass_env)
        elif self.dynamic_viscosity_model == 'GETTER':
            self.dynamic_viscosity = self.register.get('mu',self.dynamic_viscosity_model)(temperature)
        return self.dynamic_viscosity

    def _calculate_binary_diffusion_coef(self,temperature):
        '''
            计算二元扩散系数
        '''
        if self.binary_diffusion_coef_model == 'Chapman-Enskog':
            self.binary_diffusion_coef = self.register.get('binary_diffusion_coef',self.binary_diffusion_coef_model)(temperature,self.pressure_env)
        elif self.binary_diffusion_coef_model == 'SPATRA':
            self.binary_diffusion_coef = self.register.get('binary_diffusion_coef',self.binary_diffusion_coef_model)(temperature,self.pressure_env)
        elif self.binary_diffusion_coef_model == 'LiJian':
            self.binary_diffusion_coef = self.register.get('binary_diffusion_coef',self.binary_diffusion_coef_model)(temperature,self.pressure_env)
        return self.binary_diffusion_coef

    def _calculate_coolant_density(self,temperature):
        '''
            计算冷却剂密度
        '''
        if self.coolant_density_model == 'IdealGas':
            self.coolant_density = self.register.get('coolant_density',self.coolant_density_model)(self.pressure_env,temperature,self.mole_fraction_env,self.mole_mass_env)     
        elif self.coolant_density_model == 'GETTER':
            self.coolant_density = self.register.get('coolant_density',self.coolant_density_model)(temperature,self.pressure_env)     
        return self.coolant_density