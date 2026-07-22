import numpy as np
from TOOL.cosmos_general_function import Registry
from MARS.mars_xml_reader import MarsXMLReader

R = 8.3143
k = np.power(2,1/3) - 1

class MarsFuelPerformance:
    '''
        燃料性能类
    '''        
    class Pressure:
        '''
            计算SiC内压
        '''
        @classmethod
        def init_shared_properties(cls,reader:MarsXMLReader):
            '''
                初始化共享参数
            '''
            # RK方程参数
            cls.Tc_CO_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Tc_CO']
            cls.Tc_Kr_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Tc_Kr']
            cls.Tc_Xe_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Tc_Xe']
            cls.Pc_CO_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Pc_CO']
            cls.Pc_Kr_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Pc_Kr']
            cls.Pc_Xe_rk = reader.models_dict['fuel_performance']['redlich_kwong']['Pc_Xe']
            cls.a_CO_rk,cls.b_CO_rk = cls._get_rk_a_b(cls.Tc_CO_rk,cls.Pc_CO_rk)
            cls.a_Kr_rk,cls.b_Kr_rk = cls._get_rk_a_b(cls.Tc_Kr_rk,cls.Pc_Kr_rk)
            cls.a_Xe_rk,cls.b_Xe_rk = cls._get_rk_a_b(cls.Tc_Xe_rk,cls.Pc_Xe_rk)
            # VanDerWaals方程参数
            cls.a_CO_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['a_CO']
            cls.a_Kr_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['a_Kr']
            cls.a_Xe_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['a_Xe']
            cls.b_CO_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['b_CO']
            cls.b_Kr_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['b_Kr']
            cls.b_Xe_van_der_waals = reader.models_dict['fuel_performance']['van_der_waals']['b_Xe']

        @classmethod
        def _get_rk_a_b(cls,T_c,P_c):
            '''
                计算RK方程参数
            '''
            a = R**2*(np.power(T_c,2.5))/9/k/P_c
            b = k*R*T_c/3/P_c
            return a,b

        @classmethod
        def calculate_transient_co_release_fraction(cls,time,time_mid_history,time_step_history,temperature_mid_list_history):
            '''
                计算稳态运行时CO释放份额的瞬态变化(CO释放份额与瞬时变化的温度历史相关)
            '''
            # 计算瞬态CO释放份额
            gh_list_history = -10.08 - 8500.0/temperature_mid_list_history
            gh_list_history = np.power(10,gh_list_history)
            co_release_fraction = 2 * np.sum(gh_list_history[:,0] * (time - time_mid_history) * time_step_history)
            co_release_fraction = 0.625 if co_release_fraction >= 0.625 else co_release_fraction
            # 反推平均辐照温度
            co_release_fraction_convert = np.zeros_like(temperature_mid_list_history[0])
            for gh_list,time_mid,time_step in zip(gh_list_history,time_mid_history,time_step_history):
                co_release_fraction_convert += 2*gh_list*(time-time_mid)*time_step
            temperature_mid_list_irradiation = co_release_fraction_convert / time**2
            temperature_mid_list_irradiation = np.log10(temperature_mid_list_irradiation)
            temperature_mid_list_irradiation = -(temperature_mid_list_irradiation+10.08)/8500.0
            temperature_mid_list_irradiation = 1/temperature_mid_list_irradiation

            return co_release_fraction,temperature_mid_list_irradiation
        
        @classmethod
        def calculate_steady_co_release_fraction(cls,temperature_mid_list,temperature_mid_list_irradiation,time_accident):
            '''
                计算事故阶段CO释放份额的瞬态变化(事故停堆后核芯的辐照与温度历史已经确定,此时CO释放份额按照确定规律变化)
            '''
            temperature_mid_irradiation_kernel = temperature_mid_list_irradiation[0]
            temperature_kernel = temperature_mid_list[0]
            co_release_fraction = np.power(10,-10.08-8500/temperature_mid_irradiation_kernel+2*np.log10(time_accident) - 0.404*(1e4/temperature_kernel - 1e4/(temperature_mid_irradiation_kernel+75)))
            co_release_fraction = 0.625 if co_release_fraction >= 0.625 else co_release_fraction

            return co_release_fraction
        
        @classmethod
        def calculate_fp_release_fraction(cls,time_step,temperature_mid_list,temperature_mid_list_irradiation,steady_gases_release_fraction_max_iteration,tao_a,time=None,time_accident=None):
            '''
                计算稳态运行时裂变产物释放份额的瞬态变化(裂变产物释放份额与瞬时变化的温度历史相关)
            '''
            N = steady_gases_release_fraction_max_iteration
            temperature_irradiation_kernel = temperature_mid_list_irradiation[0]
            temperature_kernel = temperature_mid_list[0]

            def a_n(n, tau):
                npi = np.pi * n
                return (1.0 - np.exp(-(npi ** 2) * tau)) / (npi ** 4)

            def tauk(tau):
                if tau == 0.0:
                    return 0.0

                n = np.arange(1, N)          
                terms = a_n(n, tau)           

                mask = terms > 1e-20
                if not np.any(mask):
                    return 1.0
                total = terms[mask].sum()

                return 1.0 - 6.0 * total / tau
            
            ds_1 = np.power(10,-2.30-0.8116e4/temperature_irradiation_kernel)
            ds_2 = np.power(10,-2.30-0.8116e4/temperature_kernel)

            if time is not None:
                tao_i = ds_1 * time
                tao_a = 0.0
            else:
                tao_i = ds_1 * time_accident
                tao_a += ds_2 * time_step
            tao_s = tao_i + tao_a
            FDA = tauk(tao_a)
            FDS = tauk(tao_s)

            fp_release_fraction = FDS + tao_a*(FDS - FDA) / tao_i if tao_i != 0.0 else 0.0
            
            return fp_release_fraction,tao_a

        @classmethod
        def convert_gwd_t_to_fima(cls,burnup_gwd_t):
            '''
                转换燃耗单位[GW][d]/[t] - > [FIMA%]
            '''
            return burnup_gwd_t / 10.0

        def __init__(self,register:Registry):
            # 模型注册
            register.register('pressure','IdealGas',self._ideal_gas)
            register.register('pressure','RedlichKwong',self._redlich_kwong)
            register.register('pressure','VanDerWaals',self._vander_walls)
        
        def _ideal_gas(self,temperature_buffer,yield_Xe,yield_Kr,CO_release_fraction,fp_release_fraction,volume_kernel_m3,volume_buffer_pore_m3,molar_volume_kernel,burnup_fima):
            '''
                理想气体状态方程
            '''
            # molar_volume_kernel [m3]/[molar]
            fission_gases_yield = yield_Kr + yield_Xe
            molar_gas = volume_kernel_m3/molar_volume_kernel*burnup_fima/100*(fp_release_fraction*fission_gases_yield+CO_release_fraction)
            pressure = molar_gas * R * temperature_buffer / volume_buffer_pore_m3

            return pressure
        
        def _redlich_kwong(self,temperature_buffer,yield_Xe,yield_Kr,CO_release_fraction,fp_release_fraction,volume_kernel_m3,volume_buffer_pore_m3,molar_volume_kernel,burnup_fima):
            '''
                RedlichKwong
            '''
            a_rk_CO = self.a_CO_rk
            a_rk_Kr = self.a_Kr_rk
            a_rk_Xe = self.a_Xe_rk
            b_rk_CO = self.b_CO_rk
            b_rk_Kr = self.b_Kr_rk
            b_rk_Xe = self.b_Xe_rk
            total = CO_release_fraction + fp_release_fraction * (yield_Kr + yield_Xe)
            frac_CO = CO_release_fraction / total
            frac_Kr = fp_release_fraction * yield_Kr / total
            frac_Xe = fp_release_fraction * yield_Xe / total
            a = (np.sqrt(a_rk_CO)*frac_CO + np.sqrt(a_rk_Kr)*frac_Kr + np.sqrt(a_rk_Xe)*frac_Xe)**2
            b = b_rk_CO*frac_CO + b_rk_Xe*frac_Xe + b_rk_Kr*frac_Kr
            fission_gases_yield = yield_Kr + yield_Xe
            molar_gas = volume_kernel_m3/molar_volume_kernel*burnup_fima/100*(fp_release_fraction*fission_gases_yield+CO_release_fraction)
            void_gases_molar_volume = volume_buffer_pore_m3/molar_gas
            pressure = R*temperature_buffer/(void_gases_molar_volume-b) - a/(np.sqrt(temperature_buffer)*void_gases_molar_volume*(void_gases_molar_volume+b))

            return pressure
        
        def _vander_walls(self,temperature_buffer,yield_Xe,yield_Kr,CO_release_fraction,fp_release_fraction,volume_kernel_m3,volume_buffer_pore_m3,molar_volume_kernel,burnup_fima):
            '''
                Van der Waals
            '''
            a_vdw_CO = self.a_CO_van_der_waals
            a_vdw_Kr = self.a_Kr_van_der_waals
            a_vdw_Xe = self.a_Xe_van_der_waals
            b_vdw_CO = self.b_CO_van_der_waals
            b_vdw_Kr = self.b_Kr_van_der_waals
            b_vdw_Xe = self.b_Xe_van_der_waals
            fission_gases_yield = yield_Kr + yield_Xe
            molar_gas = volume_kernel_m3/molar_volume_kernel*burnup_fima/100*(fp_release_fraction*fission_gases_yield+CO_release_fraction)
            total = CO_release_fraction + fp_release_fraction * (yield_Kr + yield_Xe)
            frac_CO = CO_release_fraction / total
            frac_Kr = fp_release_fraction * yield_Kr / total
            frac_Xe = fp_release_fraction * yield_Xe / total
            a = (np.sqrt(a_vdw_CO)*frac_CO + np.sqrt(a_vdw_Kr)*frac_Kr + np.sqrt(a_vdw_Xe)*frac_Xe)**2
            b = b_vdw_CO*frac_CO + b_vdw_Xe*frac_Xe + b_vdw_Kr*frac_Kr
            pressure = molar_gas*R*temperature_buffer/(volume_buffer_pore_m3-molar_gas*b) - a*(molar_gas/volume_buffer_pore_m3)**2

            return pressure
        
    class FPCorrosion:
        '''
            金属腐蚀
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('sic_thickness','Attenuation',self._attenuation)
            register.register('sic_thickness','Disable',self._disable)
        
        def _attenuation(self,time,accident_time,temperature_mid_list,time_step,corrosion_coef_old,sic_thickness_raw):
            '''
                计算这一时间步的SiC腐蚀系数sic_thcikness = sic_thcikness_raw + corrosion_coef
            '''
            temperature_sic = temperature_mid_list[3]
            if time <= accident_time:
                corrosion_coef = 1.0
            else:
                corrosion_rate = (5.87e-7)*np.exp(-179500.0/R/temperature_sic) # [m]/[s] -> [cm]/[s]
                corrosion_coef_change = corrosion_rate*time_step/sic_thickness_raw
                corrosion_coef = corrosion_coef_old + corrosion_coef_change
            
            return corrosion_coef
        
        def _disable(self):
            return 1.0
            
    class Stress:
        '''
            应力分布
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('stress','Bubble',self._bubble)
            register.register('stress','RigidSiC',self._rigid_sic)

        def _bubble(self,pressure,sic_thickness,r_average_sic):
            '''
                曹建主, 奚树人. PANAMA 程序及其在 10MW 高温气冷实验堆安全分析中的应用[J]. 核动力工程, 1998, 19(2): 162-167.
            '''
            pressure = pressure / 1e6
            sic_thickness = sic_thickness / 1e6
            stress_t_sic = r_average_sic*pressure/(2*sic_thickness)
            stress_t_ipyc = 0.0
            stress_t_opyc = 0.0

            return stress_t_sic,stress_t_ipyc,stress_t_opyc
        
        def _rigid_sic(self,pressure,pyc_creep_poisson_ratio,pyc_poisson_ratio,sr,st,pyc_creep_coef,r_real):
            '''
                SAWA, K., SUMITA, J., WATANABE, T., Fuel failure and fission gas release analysis code in HTGR, Rep. JAERI-DATA/Code 99-034, Japan Atomic Energy Research Institute, Oarai (1999)
            '''
            pressure = pressure / 1e6
            r = r_real
            Wa = (r[3]/r[2])**3
            Ga = np.log(r[2]/r[3])
            Wb = (r[4]/r[5])**3
            Gb = np.log(r[5]/r[4])
            stress_r_sic = -3*(1-pyc_poisson_ratio)/(1+pyc_poisson_ratio+2*(1-2*pyc_poisson_ratio)*Wa)*pressure
            mask1 = (Wa - 1)* (sr+2*st) + 3*Ga*(sr-st) 
            mask2 = 2*Wa*(2*pyc_creep_poisson_ratio-1) - (1+pyc_creep_poisson_ratio)
            stress_r_sic_ipyc = 2*mask1/3/pyc_creep_coef/mask2
            stress_r_sic = stress_r_sic+stress_r_sic_ipyc
            mask3 = (Wb - 1)* (sr+2*st) + 3*Gb*(sr-st) 
            mask4 = 2*Wb*(2*pyc_creep_poisson_ratio-1) - (1+pyc_creep_poisson_ratio)
            stress_r_outer_SiC = 2*mask3/3/pyc_creep_coef/mask4
            mask5 = stress_r_outer_SiC*r[4]**3 - stress_r_sic*r[3]**3 + ((stress_r_outer_SiC-stress_r_sic)*r[4]**3)/2
            stress_t_sic = mask5/(r[4]**3-r[3]**3)
            stress_t_ipyc = pressure*(Wa+2)/2/(Wa-1)
            stress_t_opyc = pressure*(2*Wb+1)/(2*(1-Wb))

            return stress_t_sic,stress_t_ipyc,stress_t_opyc
        
    class SiCThermalDecomposition:
        '''
            SiC 热分解
        '''
        def __init__(self,register:Registry):
            # 模型注册
            register.register('thermal_decomposition','PANAMA',self._panama)
        
        def _panama(self,time,time_accident,alpha,beta,sic_thickness_raw,temperature_sic,time_step,activate_intgral):
            '''
               曹建主, 奚树人. PANAMA 程序及其在 10MW 高温气冷实验堆安全分析中的应用[J]. 核动力工程, 1998, 19(2): 162-167. 
            '''
            if time < time_accident:
                return 0.0,activate_intgral
            k = 375/sic_thickness_raw*np.exp(-556000/R/temperature_sic)
            activate_intgral += k*time_step
            alex = np.log10(alpha)+beta*np.log10(activate_intgral)
            if alex < -50.0:
                alex = -50.0
            ex = np.exp(alex*2.302585093)
            if ex < 1e-5:
                return ex,activate_intgral
            else:
                thermal_decomposition_rate = 1 - np.exp(-ex)
            
            return thermal_decomposition_rate,activate_intgral