import numpy as np
import sys
import copy
import bisect
import os
from typing import List
from TOOL.cosmos_general_function import TimeSeries,Registry
from MARS.mars_fuel_performance import MarsFuelPerformance
from MARS.mars_xml_reader import MarsXMLReader
from MARS.mars_material_properties import MarsMaterialProperties
from MARS.mars_solver import MarsSolver

R = 8.314462618

class Sphere:
    '''
        球体类
    '''
    class Geometry:
        '''
            球体几何与网格划分
        '''
        def __init__(self,category,reader:MarsXMLReader):
            self.category = category
            # 材料半径
            self.r_real = np.array(reader.fuel_properties_dict['geometry']['r_'+category])
            # 网格
            self.mesh_number = np.array(reader.solver_dict['mesh'][category+'_mesh_number']).astype(int)
            self.mesh_bias = reader.solver_dict['mesh'][category+'_mesh_bias']
            self.uniform_mesh = False if 'taylor' not in reader.solver_dict['diffusion_solver'][category+'_diffusion_solver'] else True
            
        def generate_mesh(self):
            '''
                生成网格与相关几何属性
            '''
            # 生成网格半径
            self._mesh_radius(self.uniform_mesh)
            # 球壳外表面积
            self._area()
            # 球壳体积
            self._volume()
            # 有限体积法中间量
            self._gamma()
            # 材料索引
            self._material_idx()

        def _mesh_radius(self,uniform_mesh):
            '''
                生成网格半径
            '''
            if uniform_mesh:
                self._uniform_mesh()
            else:
                self._non_uniform_mesh()
            
        def _non_uniform_mesh(self):
            '''
                非均匀网格
            '''
            self.r_num = [0]
            for i in range(len(self.mesh_number)):
                h = (self.r_real[i+1] - self.r_real[i]) / self.mesh_number[i]
                for j in range(self.mesh_number[i]):
                    self.r_num.append(self.r_num[-1]+h)
                self.r_num[-(self.mesh_number[i]+1):] = self._mesh_bias(self.r_num[-(self.mesh_number[i]+1):],self.mesh_bias[i])
            self.r_num = np.array(self.r_num)

        def _uniform_mesh(self):
            '''
                均匀网格
            '''
            self.r_num = [0]
            N_total = sum(self.mesh_number)
            self.mesh_number = np.zeros_like(self.mesh_number)
            h = self.r_real[-1]/N_total
            if h > min(np.array(self.r_real[1:]) - np.array(self.r_real[0:-1])):
                print('The grid num is too small')
                print('Your step: ' + str(h) )
                print('The minimum shell thickness: '+str(min(np.array(self.r_real[1:]) - np.array(self.r_real[0:-1]))))
                sys.exit()
            j = 0
            for i in range(N_total):
                self.r_num.append(self.r_num[-1]+h)
                if self.r_num[-1] <= self.r_real[j+1]:
                    self.mesh_number[j] += 1
                elif self.r_num[-1] > self.r_real[-1]:
                    self.mesh_number[j] += 1
                    break
                else:
                    self.mesh_number[j] += 1
                    j += 1
            self.r_num = np.array(self.r_num)
        
        def _mesh_bias(self,nod_list,grid_bias):
            '''
                网格偏差
            '''
            if grid_bias > 1 or grid_bias < -1:
                print('The grid bias should be between [-1,1]')
                if grid_bias > 1:
                    grid_bias = 1
                elif grid_bias < -1:
                    grid_bias = -1

            def linear_assignment(nod_num):
                right = 1 + grid_bias*( 2*nod_num/(len(nod_list)-1) - 1 )
                return right

            right = [linear_assignment(i) for i in range(1,len(nod_list))]
            Sum = sum(right)
            Lens = nod_list[-1] - nod_list[0]
            gap = np.array([Lens*r/Sum for r in right])
            positions_cumsum = np.cumsum(gap)
            new_nod_list = positions_cumsum + nod_list[0]
            new_nod_list = np.insert(new_nod_list, 0, nod_list[0])

            return new_nod_list

        def _area(self):
            '''
                各数值层的球壳面积
            '''
            self.area_num = 4*np.pi*(self.r_num**2)
            self.area_real = 4*np.pi*(self.r_real**2)

        def _volume(self):
            '''
                各壳层的体积
            '''
            self.volume_num = 4*np.pi*(self.r_num[1]**3)/3
            self.volume_num = np.append(self.volume_num, (4*np.pi*(self.r_num[2:]**3)/3 - 4*np.pi*(self.r_num[1:-1]**3)/3))
            self.volume_real = 4*np.pi*(self.r_real[1]**3)/3
            self.volume_real = np.append(self.volume_real, (4*np.pi*(self.r_real[2:]**3)/3 - 4*np.pi*(self.r_real[1:-1]**3)/3) )
            self.volume_total = sum(self.volume_real)

        def _gamma(self):
            '''
                有限体积法的中间量
            '''
            self.gamma = np.zeros_like(self.volume_num)
            self.gamma[0] = 0.4
            mask1 = 3/(self.r_num[2:]**3 - self.r_num[1:-1]**3)
            mask2 = 1/(self.r_num[2:] - self.r_num[1:-1])
            mask3 = (self.r_num[2:]**4 - self.r_num[1:-1]**4)/4 - (self.r_num[1:-1]*(self.r_num[2:]**3) -self.r_num[1:-1]**4)/3
            self.gamma[1:] = 1 - mask1*mask2*mask3

        def _material_idx(self):
            '''
                材料的起始与结束索引
            '''
            self.material_idx = []
            material_region_idx_end = 0
            for material_idx,mesh_number_section in enumerate(self.mesh_number):
                material_region_idx_start = material_region_idx_end
                material_region_idx_end += mesh_number_section
                self.material_idx.append((material_region_idx_start,material_region_idx_end))

    def __init__(self,category,reader):
        # 几何与网格
        self.geometry = self.Geometry(category,reader)
        self.geometry.generate_mesh()

class DiffusionField:
    '''
        核素扩散场
    '''
    @classmethod
    def set_shared_properties(cls,reader:MarsXMLReader):
        '''
            共有核素扩散属性
        '''
        # 初始核素盘存量
        cls.initial_inventory = reader.conditions_dict['element_initial_inventory']
        # 核素衰变常数
        cls.decay_constant = reader.external_conditions_dict['decay_constant']
        # 反冲模型
        cls.recoil_model = reader.models_dict['recoil']['recoil_model']
        # 总颗粒数
        cls.particle_number_total = reader.fuel_properties_dict['particle_number'] 
        # 时间离散系数
        cls.crank_nicolson_time_weight = reader.solver_dict['diffusion_solver']['crank_nicolson_time_weight']
        # 扩散释放模型
        cls.diffusion_model = reader.models_dict['diffusion_model']

    def __init__(self,category,reader:MarsXMLReader,geometry:Sphere.Geometry,diffusion_coef):
        
        self.category = category
        # 球体几何
        self.geometry = geometry
        # 铀污染份额/核素产生份额
        self.uranium_contamination = reader.fuel_properties_dict['uranium_contamination']['uranium_contamination_'+category]
        self.uranium_contamination_generate_share = self.uranium_contamination if category in ['element','graphite_grain'] else self.uranium_contamination/self.particle_number_total
        # 核素裂变反冲
        self.r_recoil = reader.models_dict['recoil']['r_recoil_'+category]
        self.recoil_reduction_correction_list,self.recoil_release_correction_list = self._calculate_recoil()
        # 环境核素浓度
        self.concentration_environment = reader.conditions_dict['c_'+category+'_environment']
        # 边界质量传递系数
        self.mass_transfer_coef = reader.models_dict['mass_transfer'][category+'_mass_transfer_coef']
        self.mass_transfer_mode = reader.models_dict['mass_transfer']['mode']
        # 核素扩散求解方法
        self.solver = reader.solver_dict['diffusion_solver'][category + '_diffusion_solver']
        # 初始化核素浓度
        self.concentration_num_fdm,self.concentration_num_fvm,self.inventory_material_layer,self.inventory = self._initialize_concentration(diffusion_coef)
        # 累计释放量
        self.diffusion_release_cumulant = 0.0
        self.recoil_release_cumulant = 0.0
        self.release_cumulant = 0.0

    def update_generation_rate(self,element_generation_rate):
        '''
            更新球体因裂变导致的核素产生率分布
        '''
        # 原始裂变产生率分布
        generation_rate_num_uranium_contamination_raw = self._calculate_uranium_contamination_generation_rate_raw(element_generation_rate)
        # 反冲修正
        generation_rate_num_uranium_contamination,self.recoil_release_rate = self._calculate_recoil_correction(generation_rate_num_uranium_contamination_raw)
        # 对于除元件以外的球体,产生率分布就是裂变导致的产生率分布
        self.generation_rate_num = generation_rate_num_uranium_contamination
        self.generation_rate = np.sum(self.generation_rate_num*self.geometry.volume_num)

    def update_concentration_field(self,time_step,diffusion_coef_num,adsorp_coef=None):
        '''
            更新浓度场
        '''
        if self.diffusion_model == 'Numerical':
            self._fick_diffusion(time_step,diffusion_coef_num,adsorp_coef)
        elif self.diffusion_model == 'Booth':
            self._booth(time_step,diffusion_coef_num)

    def _booth(self,time_step,diffusion_coef_num):
        '''
            Booth模型
        ''' 
        # Booth模型不输出浓度/盘存量
        self.concentration_num_fdm = np.zeros_like(self.concentration_num_fdm)
        self.concentration_num_fvm = np.zeros_like(self.concentration_num_fvm)
        self.inventory_material_layer = np.zeros_like(self.inventory_material_layer)
        self.inventory = 0.0
        # 更新释放率
        D = np.mean(diffusion_coef_num)
        x = np.sqrt(self.decay_constant*self.geometry.r_num[-1]**2/D)
        R = 3*self.generation_rate*(1/np.tanh(x) - 1/x)/x
        if self.category == 'particle':
            R = 0.0    
        self.diffusion_release_rate = R
        self.diffusion_release_cumulant = self._calculate_release_cumulant(self.diffusion_release_rate,self.diffusion_release_cumulant,time_step)
        self.release_cumulant = self.diffusion_release_cumulant + self.recoil_release_cumulant
        self.release_rate = self.diffusion_release_rate + self.recoil_release_rate

    def _fick_diffusion(self,time_step,diffusion_coef_num,adsorp_coef=None):
        '''
            Fick扩散计算浓度场与释放率
        '''
        # 记录原始浓度场/盘存量
        concentration_num_old_fdm = self.concentration_num_fdm
        concentration_num_old_fvm = self.concentration_num_fvm
        inventory_old = self.inventory
        inventory_material_layer_old = self.inventory_material_layer
        generation_rate_num = self.generation_rate_num
        # 扩散计算/更新浓度场
        self.concentration_num_fdm,self.concentration_num_fvm = self._calculate_diffusion(time_step,concentration_num_old_fdm,concentration_num_old_fvm,diffusion_coef_num,generation_rate_num,adsorp_coef)
        # 更新盘存量
        self.inventory_material_layer,self.inventory = self._calculate_inventory(self.concentration_num_fdm,self.concentration_num_fvm)
        # 释放率计算
        self.diffusion_release_rate = self._calculate_diffusion_release_rate(self.concentration_num_fdm,self.concentration_num_fvm,diffusion_coef_num,self.geometry.r_num,adsorp_coef)
        self.diffusion_release_cumulant = self._calculate_release_cumulant(self.diffusion_release_rate,self.diffusion_release_cumulant,time_step)
        self.recoil_release_cumulant = self._calculate_release_cumulant(self.recoil_release_rate,self.recoil_release_cumulant,time_step)
        self.release_cumulant = self.diffusion_release_cumulant + self.recoil_release_cumulant
        self.release_rate = self.diffusion_release_rate + self.recoil_release_rate

    def _calculate_uranium_contamination_generation_rate_raw(self,element_generation_rate):
        '''
            通过元件的总核素产生率计算得到该球体各壳层由铀裂变产生的不考虑反冲的体积产生率[Bq]/[cm3][s]
        '''
        generation_rate_volumetric = element_generation_rate * self.uranium_contamination_generate_share / self.geometry.volume_real
        generation_rate_num_uranium_contamination_raw = np.repeat(generation_rate_volumetric,self.geometry.mesh_number)
        
        return generation_rate_num_uranium_contamination_raw
    
    def _calculate_recoil_propapility(self,r,r_inner_sphere_edge,r_recoil):
        '''
            计算r处的反冲概率
        '''
        r0 = r_inner_sphere_edge
        if self.recoil_model == 'non_uniform':
            # 按照确定位置的反冲球计算该位置的反冲概率
            recoil_probability = 0.5 - (r0**2 - r_recoil**2 - r**2) / (4*r_recoil*r)
        elif self.recoil_model == 'uniform':
            # 按照总反冲概率与受到反冲影响区域的总体积计算该区域各位置的平均反冲概率
            influnced_volume = 4 * np.pi * (r_inner_sphere_edge**3 - (r_inner_sphere_edge-r_recoil)**3) / 3
            recoil_probability_total = np.pi * r_recoil * (r_inner_sphere_edge**2 - (r_recoil**2)/12)
            recoil_probability = recoil_probability_total / influnced_volume if influnced_volume != 0 else 0
        elif self.recoil_model == 'off':
            recoil_probability = 0.0
        
        return recoil_probability

    def _calculate_recoil(self):
        '''
            计算裂变造成的反冲
        '''
        mesh_number = self.geometry.mesh_number
        r_num = self.geometry.r_num
        recoil_correction_coef_list = np.zeros(sum(mesh_number))
        for material_idx,mesh_number_section in enumerate(mesh_number):
            material_region_idx_start = self.geometry.material_idx[material_idx][0]
            material_region_idx_end = self.geometry.material_idx[material_idx][1]
            r_inner_sphere_edge = r_num[material_region_idx_end]
            material_thcikness = r_num[material_region_idx_end] - r_num[material_region_idx_start]
            r_recoil = self.r_recoil[material_idx]
            if material_thcikness <= r_recoil:
                recoil_influnced_region_idx_list = list(range(material_region_idx_start,material_region_idx_end))
            elif r_recoil > 0:
                recoil_influnced_region_idx_list = []
                for idx in range(material_region_idx_end, -1, -1):
                    r = r_num[idx]
                    distance = r_inner_sphere_edge - r
                    if distance <= r_recoil:
                        recoil_influnced_region_idx_list.append(idx-1)
                    else:
                        break
            else:
                recoil_influnced_region_idx_list = []
            for recoil_influnced_region_idx in recoil_influnced_region_idx_list:
                r = (r_num[recoil_influnced_region_idx] + r_num[recoil_influnced_region_idx+1]) / 2
                recoil_correction_coef_list[recoil_influnced_region_idx] = self._calculate_recoil_propapility(r,r_inner_sphere_edge,r_recoil)
        recoil_reduction_correction_list = 1.0 - recoil_correction_coef_list # 该数组乘generation_rate_num_uranium_contamination_raw数组即得到反冲修正后的产生率
        recoil_release_correction_list = recoil_correction_coef_list * self.geometry.volume_num # 该数组乘generation_rate_num_uranium_contamination_raw数组即可得到反冲导致的释放

        return recoil_reduction_correction_list,recoil_release_correction_list

    def _calculate_recoil_correction(self,generation_rate_num_uranium_contamination_raw):
        '''
            计算反冲对核素产生率分布以及释放的修正
        '''
        volume_real = self.geometry.volume_real
        mesh_number = self.geometry.mesh_number
        generation_rate_num_uranium_contamination = copy.deepcopy(generation_rate_num_uranium_contamination_raw)
        recoil_release_list = []
        
        # 反冲导致的核素产生率减少
        for material_idx,mesh_number_section in enumerate(mesh_number):
            material_region_idx_start = self.geometry.material_idx[material_idx][0]
            material_region_idx_end = self.geometry.material_idx[material_idx][1]
            generation_rate_num_uranium_contamination[material_region_idx_start:material_region_idx_end] = generation_rate_num_uranium_contamination_raw[material_region_idx_start:material_region_idx_end] * self.recoil_reduction_correction_list[material_region_idx_start:material_region_idx_end]
            recoil_release = sum(generation_rate_num_uranium_contamination_raw[material_region_idx_start:material_region_idx_end] * self.recoil_release_correction_list[material_region_idx_start:material_region_idx_end])
            recoil_release_list.append(recoil_release) # [Bq]/[s]
        
        # 反冲导致的核素产生率增加
        for material_idx,mesh_number_section in enumerate(mesh_number):
            if material_idx == 0:
                continue
            material_region_idx_start = self.geometry.material_idx[material_idx][0]
            material_region_idx_end = self.geometry.material_idx[material_idx][1]
            recoil_release_rate = recoil_release_list[material_idx-1]
            material_volume = volume_real[material_idx]
            recoil_generation_rate_increase = recoil_release_rate / material_volume
            # 保守的假设从内层材料反冲到外层材料的核素均匀的铺满整个外层材料(实际上外层材料只有靠内侧r_recoil的数值层才会被增加)
            generation_rate_num_uranium_contamination[material_region_idx_start:material_region_idx_end] += recoil_generation_rate_increase
        
        # 球体最外材料的反冲即为该球体的反冲释放率
        sphere_recoil_release_rate = recoil_release_list[-1] 

        return generation_rate_num_uranium_contamination,sphere_recoil_release_rate

    def _calculate_inventory(self,concentration_num_new_fdm,concentration_num_new_fvm):
        '''
            计算盘存量
        '''
        # 各数值壳层内的盘存量
        if 'fvm' in self.solver:
            inventory_num = self.geometry.volume_num * (self.geometry.gamma*concentration_num_new_fvm[1:]+(1-self.geometry.gamma)*concentration_num_new_fvm[0:-1])
        elif 'fdm' in self.solver:
            inventory_num = self.geometry.volume_num * concentration_num_new_fdm
        
        # 各材料内的盘存量
        inventory_material_layer = np.zeros_like(self.geometry.mesh_number,dtype='float')
        for material_idx,mesh_number_section in enumerate(self.geometry.mesh_number):
            inventory_material_layer[material_idx] = sum(inventory_num[self.geometry.material_idx[material_idx][0]:self.geometry.material_idx[material_idx][1]])
        
        # 总盘存量
        inventory = sum(inventory_material_layer)
      
        return inventory_material_layer,inventory
    
    def _calculate_diffusion_release_rate(self,concentration_num_fdm,concentration_num_fvm,diffusion_coef_num,r_num,adsorp_coef=None):
        '''
            计算扩散释放率 [Bq]/[s]
        '''
        if self.category == 'element':
            diffusion_release_rate = self.geometry.area_num[-1] * (concentration_num_fvm[-1]-self.concentration_environment/adsorp_coef) * (self.mass_transfer_coef*adsorp_coef)
        else:
            diffusion_release_rate = self.geometry.area_num[-1] * (concentration_num_fvm[-1]-self.concentration_environment) * self.mass_transfer_coef
  
        if diffusion_release_rate < 0.0:
            diffusion_release_rate = 0.0 
        return diffusion_release_rate

    def _calculate_release_cumulant(self,release_rate,release_cumulant,time_step):
        '''
            计算累计释放量
        '''
        release_cumulant *= np.exp(-self.decay_constant*time_step)
        release_cumulant += release_rate*time_step
        
        return release_cumulant

    def _initialize_concentration(self,diffusion_coef_num,adsorp_coef=1.0):
        '''
            初始化核素浓度
        '''
        # 初始化
        concentration_num_old_fdm = np.zeros(sum(self.geometry.mesh_number))
        concentration_num_old_fvm = np.zeros(sum(self.geometry.mesh_number)+1)
        # 第一次扩散计算
        generation_rate_num = self._calculate_uranium_contamination_generation_rate_raw(1.0)
        concentration_num_raw_fdm,concentration_num_raw_fvm = self._calculate_diffusion(1e22,concentration_num_old_fdm,concentration_num_old_fvm,diffusion_coef_num,generation_rate_num,adsorp_coef)
        # 归一化/第二次扩散计算
        inventory_material_layer,inventory = self._calculate_inventory(concentration_num_raw_fdm,concentration_num_raw_fvm)
        correction_coef = self.initial_inventory / inventory * sum(self.uranium_contamination_generate_share) if inventory != 0 else 0
        concentration_num_modified_fdm = concentration_num_raw_fdm * correction_coef
        concentration_num_modified_fvm = concentration_num_raw_fvm * correction_coef
        generation_rate_num_modified = generation_rate_num * correction_coef
        concentration_num_new_fdm,concentration_num_new_fvm = self._calculate_diffusion(1e22,concentration_num_modified_fdm,concentration_num_modified_fvm,diffusion_coef_num,generation_rate_num_modified,adsorp_coef)
        inventory_material_layer_new,inventory_new = self._calculate_inventory(concentration_num_new_fdm,concentration_num_new_fvm)  
        
        return concentration_num_new_fdm,concentration_num_new_fvm,inventory_material_layer_new,inventory_new

    def _calculate_diffusion(self,time_step,concentration_num_old_fdm,concentration_num_old_fvm,diffusion_coef_num,generation_rate_num,adsorp_coef=None):
        '''
            计算核素扩散
        '''
        solver = self.solver
        x_env = self.concentration_environment
        decay_constant = self.decay_constant
        boundary_transport_coef = self.mass_transfer_coef
        x_k_fdm = concentration_num_old_fdm
        x_k_fvm = concentration_num_old_fvm
        D = diffusion_coef_num
        r = self.geometry.r_num
        volume = self.geometry.volume_num
        area = self.geometry.area_num
        gamma = self.geometry.gamma
        Q = generation_rate_num
        w = self.crank_nicolson_time_weight
        
        # 燃料元件的边界条件
        if self.category == 'element':
            boundary_transport_coef = boundary_transport_coef*adsorp_coef
            
            x_env = x_env / adsorp_coef
        
        concentration_num_new_fdm,concentration_num_new_fvm = MarsSolver.calculate_numerical_solution(solver,x_env,time_step,decay_constant,boundary_transport_coef,x_k_fdm,x_k_fvm,D,r,volume,Q,w,area,gamma)

        return concentration_num_new_fdm,concentration_num_new_fvm

class Particle(Sphere):
    '''
        完整颗粒类型
    '''
    class MaterialProperties(MarsMaterialProperties):

        def __init__(self,reader:MarsXMLReader,geometry:Sphere.Geometry):
            super().__init__(reader,'particle')
            # 颗粒网格信息
            self.mesh_number = geometry.mesh_number
            # 该颗粒的特殊属性
            self.molar_mass_kernel = reader.fuel_properties_dict['material_properties']['molar_mass_kernel']
            # 各材料的数值层索引区间
            self.kernel_idx = (geometry.material_idx[0][0],geometry.material_idx[0][1])
            self.buffer_idx = (geometry.material_idx[1][0],geometry.material_idx[1][1])
            self.ipyc_idx   = (geometry.material_idx[2][0],geometry.material_idx[2][1])
            self.sic_idx    = (geometry.material_idx[3][0],geometry.material_idx[3][1])
            self.opyc_idx   = (geometry.material_idx[4][0],geometry.material_idx[4][1])
            # 属性
            self.properties_dict = {
                'thermal_conductivity':self._update_thermal_conductivity_particle,
                'heat_capacity':self._update_heat_capacity_particle,
                'density':self._update_density_particle,
                'heat_diffusion_coef':self._update_heat_diffusion_coef_particle,
                'diffusion_coef':self._update_diffusion_coef_particle,
            }
            
        def _update_thermal_conductivity_particle(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            # 更新导热系数 [W]/[m][K]
            thermal_conductivity_kernel_num = self.calculate_thermal_conductivity_kernel(temperature_num[self.kernel_idx[0]:self.kernel_idx[1]])
            thermal_conductivity_buffer_num = self.calculate_thermal_conductivity_buffer(temperature_num[self.buffer_idx[0]:self.buffer_idx[1]])
            thermal_conductivity_ipyc_num = self.calculate_thermal_conductivity_pyc(temperature_num[self.ipyc_idx[0]:self.ipyc_idx[1]])
            thermal_conductivity_sic_num = self.calculate_thermal_conductivity_sic(temperature_num[self.sic_idx[0]:self.sic_idx[1]])
            thermal_conductivity_opyc_num = self.calculate_thermal_conductivity_pyc(temperature_num[self.opyc_idx[0]:self.opyc_idx[1]])
            thermal_conductivity_kernel = np.mean(thermal_conductivity_kernel_num)
            thermal_conductivity_buffer = np.mean(thermal_conductivity_buffer_num)
            thermal_conductivity_ipyc = np.mean(thermal_conductivity_ipyc_num)
            thermal_conductivity_sic = np.mean(thermal_conductivity_sic_num)
            thermal_conductivity_opyc = np.mean(thermal_conductivity_opyc_num)
            self.thermal_conductivity_particle_list = np.array([thermal_conductivity_kernel,thermal_conductivity_buffer,thermal_conductivity_ipyc,thermal_conductivity_sic,thermal_conductivity_opyc])
            self.thermal_conductivity_particle_num = np.repeat(self.thermal_conductivity_particle_list,self.mesh_number)
            
        def _update_heat_capacity_particle(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            # 更新热容 [J]/[kg][K]
            heat_capacity_kernel_num = self.calculate_heat_capacity_kernel(temperature_num[self.kernel_idx[0]:self.kernel_idx[1]],self.molar_mass_kernel)
            heat_capacity_buffer_num = self.calculate_heat_capacity_buffer(temperature_num[self.buffer_idx[0]:self.buffer_idx[1]])
            heat_capacity_ipyc_num = self.calculate_heat_capacity_pyc(temperature_num[self.ipyc_idx[0]:self.ipyc_idx[1]])
            heat_capacity_sic_num = self.calculate_heat_capacity_sic(temperature_num[self.sic_idx[0]:self.sic_idx[1]])
            heat_capacity_opyc_num = self.calculate_heat_capacity_pyc(temperature_num[self.opyc_idx[0]:self.opyc_idx[1]])
            heat_capacity_kernel = np.mean(heat_capacity_kernel_num)
            heat_capacity_buffer = np.mean(heat_capacity_buffer_num)
            heat_capacity_ipyc = np.mean(heat_capacity_ipyc_num)
            heat_capacity_sic = np.mean(heat_capacity_sic_num)
            heat_capacity_opyc = np.mean(heat_capacity_opyc_num)
            self.heat_capacity_particle_list = np.array([heat_capacity_kernel,heat_capacity_buffer,heat_capacity_ipyc,heat_capacity_sic,heat_capacity_opyc])
            self.heat_capacity_particle_num = np.repeat(self.heat_capacity_particle_list,self.mesh_number)

        def _update_density_particle(self,**kwargs):
            # 更新密度 [g]/[cm3]
            density_kernel = self.calculate_density_kernel()
            density_buffer = self.calculate_density_buffer()
            density_ipyc = self.calculate_density_pyc()
            density_sic = self.calculate_density_sic()
            density_opyc = self.calculate_density_pyc()
            self.density_particle_list = np.array([density_kernel,density_buffer,density_ipyc,density_sic,density_opyc])
            self.density_particle_num = np.repeat(self.density_particle_list,self.mesh_number)

        def _update_heat_diffusion_coef_particle(self,**kwargs):
            # 更新热扩散系数 [m2]/[s]
            density_num = self.density_particle_num * 1000.0
            density_list = self.density_particle_list * 1000.0
            self.heat_diffusion_coef_list = self.thermal_conductivity_particle_list / density_list / self.heat_capacity_particle_list
            self.heat_diffusion_coef_particle_num = self.thermal_conductivity_particle_num / density_num / self.heat_capacity_particle_num 
        
        def _update_diffusion_coef_particle(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            time = kwargs['time']
            # 更新核素扩散系数 [cm2]/[s]
            self.diffusion_coef_particle_num = self.calculate_nuclide_diffusion_coef(temperature_num,self.mesh_number,time)
            diffusion_coef_kernel = np.mean(self.diffusion_coef_particle_num[self.kernel_idx[0]:self.kernel_idx[1]])
            diffusion_coef_buffer = np.mean(self.diffusion_coef_particle_num[self.buffer_idx[0]:self.buffer_idx[1]])
            diffusion_coef_ipyc = np.mean(self.diffusion_coef_particle_num[self.ipyc_idx[0]:self.ipyc_idx[1]])
            diffusion_coef_sic = np.mean(self.diffusion_coef_particle_num[self.sic_idx[0]:self.sic_idx[1]])
            diffusion_coef_opyc = np.mean(self.diffusion_coef_particle_num[self.opyc_idx[0]:self.opyc_idx[1]])
            self.diffusion_coef_list = np.array([diffusion_coef_kernel,diffusion_coef_buffer,diffusion_coef_ipyc,diffusion_coef_sic,diffusion_coef_opyc])

        def update_particle_material_properties(self,updated_properties,**kwargs):
            '''
                更新颗粒属性
            '''
            for prop in updated_properties:
                self.properties_dict[prop](**kwargs)

    class FuelPerformance(MarsFuelPerformance):
        '''
            完整颗粒的燃料性能
        '''
        @classmethod
        def init_shared_properties(cls,reader:MarsXMLReader):
            '''
                初始化各区颗粒相同属性
            '''
            # 初始化子类共同属性
            cls.Pressure.init_shared_properties(reader)
            # 事故时间
            cls.time_accident = reader.external_conditions_dict['accident_time']
            # 颗粒几何
            cls.geometry = Sphere.Geometry('particle',reader)
            cls.geometry.generate_mesh()
            cls.volume_kernel_m3 = cls.geometry.volume_real[0] / 1e6
            cls.buffer_porosity = reader.fuel_properties_dict['material_properties']['porosity_buffer']
            cls.volume_buffer_pore_m3 = cls.geometry.volume_real[1] / 1e6 * cls.buffer_porosity
            cls.molar_volume_kernel = reader.fuel_properties_dict['material_properties']['molar_volume_kernel'] / 1e6
            cls.sic_thickness_raw = (cls.geometry.r_real[4] - cls.geometry.r_real[3]) / 1e2
            cls.r_real = cls.geometry.r_real / 1e2
            cls.r_average_sic = np.power((cls.r_real[3]**3 + cls.r_real[4]**3)/2,1/3)
            # 求解收敛条件
            cls.steady_gases_release_fraction_max_iteration = reader.solver_dict['steady_gases_release_fraction_max_iteration']
            # 模型
            cls.pressure_model = reader.models_dict['fuel_performance']['pressure_model']
            cls.corrosion_model = reader.models_dict['fuel_performance']['fp_corrosion_model']
            cls.intergranular_corrosion = reader.models_dict['fuel_performance']['intergranular_corrosion_model']
            cls.stress_model = reader.models_dict['fuel_performance']['sic_stress_model']
            cls.sic_thermal_decomposition_model = reader.models_dict['fuel_performance']['sic_thermal_decomposition_model']
            cls.failure_model = reader.models_dict['fuel_performance']['failure_model']
            # 参数
            cls.yield_Xe = reader.models_dict['fuel_performance']['yield_Xe']
            cls.yield_Kr = reader.models_dict['fuel_performance']['yield_Kr']
            cls.fast_neutron_share = reader.models_dict['fuel_performance']['fast_neutron_share']
            cls.external_pyc_creep_coef = reader.fuel_properties_dict['material_properties']['creep_coef_pyc']
            cls.sic_tensile_strength_raw = reader.fuel_properties_dict['material_properties']['tensile_strength_sic']
            cls.sic_weibull_raw = reader.fuel_properties_dict['material_properties']['weibull_sic']
            cls.pyc_creep_poisson_ratio = reader.fuel_properties_dict['material_properties']['creep_poisson_ratio_pyc']
            cls.pyc_poisson_ratio = reader.fuel_properties_dict['material_properties']['poisson_ratio_pyc']
            cls.sic_thermal_decomposition_alpha = reader.fuel_properties_dict['material_properties']['sic_thermal_decomposition_alpha']
            cls.sic_thermal_decomposition_beta = reader.fuel_properties_dict['material_properties']['sic_thermal_decomposition_beta']
            cls.sic_manufacturing_failure_fraction = reader.fuel_properties_dict['material_properties']['sic_manufacturing_failure_fraction']
            cls.failure_threshold = reader.models_dict['fuel_performance']['failure_threshold']
            cls.sic_output_failure_fraction_threshold = cls.sic_manufacturing_failure_fraction + cls.failure_threshold
            cls.step_failure_fraction_increase = reader.models_dict['fuel_performance']['step_failure_fraction_increment']
            cls.step_failure_fraction_table = np.cumsum(cls.step_failure_fraction_increase) 
            cls.step_failure_fraction_time_table = reader.models_dict['fuel_performance']['step_failure_fraction_time']
        
            # 时间
            cls.time_mid_history = [] # 燃料性能计算不得不单独给一个时间点列表和时间步列表,这是因为control类是按照输出点记录数据的
            cls.time_step_history = [] # 燃料性能计算不得不单独给一个时间点列表和时间步列表,这是因为control类是按照输出点记录数据的
            cls.fast_neutron_flux_cumulant = 0.0

        @classmethod
        def set_shared_properties(cls,time,time_step,external_burnup_gwd_t,external_neutron_flux):
            '''
                更新相同属性
            '''
            # 记录时间
            cls.time = time + time_step # 主函数在最后才会推进时间步
            cls.time_step = time_step
            if time <= cls.time_accident:
                cls.time_mid_history.append(time+time_step/2) # 主函数在最后才会推进时间步
                cls.time_step_history.append(time_step)
            # 燃耗
            cls.burnup_gwd_t = external_burnup_gwd_t
            # 燃耗单位转化[G][W][d]/[t]->[FIMA]
            cls.burnup_fima = cls.Pressure.convert_gwd_t_to_fima(cls.burnup_gwd_t)
            cls.neutron_flux = external_neutron_flux
            cls.fast_neutron_flux = cls.neutron_flux * cls.fast_neutron_share
            cls.fast_neutron_flux_cumulant += cls.fast_neutron_flux * cls.time_step

        def __init__(self,particle_temperature_list):
            super().__init__()
            # 颗粒温度
            self.temperature_list_new = copy.deepcopy(particle_temperature_list)
            self.temperature_list_old = copy.deepcopy(particle_temperature_list)
            self.temperature_mid_list_history = []
            # 开启注册表
            self.register = Registry()
            self.pressure = self.Pressure(self.register)
            self.fp_corrosion = self.FPCorrosion(self.register)
            self.stress = self.Stress(self.register)
            self.sic_thermal_decomposition = self.SiCThermalDecomposition(self.register)
            # 模型参数
            self.tao_a = 0.0
            self.pressure_old = 0.0
            self.pressure_new = 0.0
            self.corrosion_coef = 1.0
            self.sr_pyc_cumulant = 0.0
            self.st_pyc_cumulant = 0.0
            self.activate_intgral = 0.0
            self.failure_fraction = 0.0
            self.inter_accident = True
        
        def update_failure_rate(self,particle_temperature_list,particle_material):
            '''
                更新破损率
            '''
            failure_fraction_old = self.failure_fraction
            if self.failure_model == 'WeibullFailure':
                self.failure_fraction = self._update_weibull_failure(particle_temperature_list,particle_material)
            elif self.failure_model == 'StepFailure':
                self.failure_fraction = self._update_stepwise_failure()
            self.failure_fraction_increment = self.failure_fraction - failure_fraction_old
            
        def update_rupture_release_rate(self,particle_diffusion_field:DiffusionField,particle_number,time_step):
            '''
                更新破裂释放率
            '''
            if self.failure_fraction_increment > 0.0:
                particle_diffusion_field.rupture_release_rate =  sum(particle_diffusion_field.inventory_material_layer[1:]) / time_step
            else:
                particle_diffusion_field.rupture_release_rate = 0.0
            particle_diffusion_field.rupture_release_cumulant *= np.exp(-particle_diffusion_field.decay_constant*time_step)
            particle_diffusion_field.rupture_release_cumulant += particle_diffusion_field.rupture_release_rate * time_step
            
        def _update_stepwise_failure(self):
            '''
                阶梯颗粒失效率
            '''
            if len(self.step_failure_fraction_time_table) == 0 or len(self.step_failure_fraction_time_table) == 0:
                return self.sic_manufacturing_failure_fraction
            idx = bisect.bisect_right(self.step_failure_fraction_time_table,self.time)-1
            self.step_failure_fraction = self.step_failure_fraction_table[idx]
            failure_fraction = self.step_failure_fraction + self.sic_manufacturing_failure_fraction
            
            return failure_fraction

        def _update_weibull_failure(self,particle_temperature_list,particle_material):
            '''
                Weibull颗粒失效率
            '''
            # 更新温度
            self.temperature_list_old = copy.deepcopy(self.temperature_list_new)
            self.temperature_list_new = particle_temperature_list
            self.temperature_mid_list = (self.temperature_list_old + self.temperature_list_new) / 2
            if self.time <= self.time_accident:
                self.temperature_mid_list_history.append(self.temperature_mid_list)
            
            # 计算压力
            if self.time <= self.time_accident:
                self.co_release_fraction,self.temperature_mid_list_irradiation = self.pressure.calculate_transient_co_release_fraction(self.time,np.array(self.time_mid_history),np.array(self.time_step_history),np.array(self.temperature_mid_list_history))
                self.fp_release_fraction,self.tao_a = self.pressure.calculate_fp_release_fraction(self.time_step,self.temperature_mid_list,self.temperature_mid_list_irradiation,self.steady_gases_release_fraction_max_iteration,self.tao_a,time=self.time,time_accident=None)
                self.pressure_old = self.pressure_new
                self.pressure_new = self._calculate_pressure(self.temperature_mid_list_irradiation[1],self.co_release_fraction,self.fp_release_fraction,self.burnup_fima)
            else:
                fp_release_fraction_old = self.fp_release_fraction
                self.co_release_fraction = self.pressure.calculate_steady_co_release_fraction(self.temperature_mid_list,self.temperature_mid_list_irradiation,self.time_accident)
                self.fp_release_fraction,self.tao_a = self.pressure.calculate_fp_release_fraction(self.time_step,self.temperature_mid_list,self.temperature_mid_list_irradiation,self.steady_gases_release_fraction_max_iteration,self.tao_a,time=None,time_accident=self.time_accident)
                temperature_buffer_old = self.temperature_list_old[1]
                temperature_buffer = particle_temperature_list[1]
                if temperature_buffer_old > temperature_buffer:
                    temperature_buffer = (self.temperature_list_old[1]+particle_temperature_list[1]) / 2
                    temperature_buffer_old = temperature_buffer
                self.pressure_old = self._calculate_pressure(temperature_buffer_old,self.co_release_fraction,fp_release_fraction_old,self.burnup_fima)
                self.pressure_new = self._calculate_pressure(temperature_buffer,self.co_release_fraction,self.fp_release_fraction,self.burnup_fima)
            
            # 计算SiC金属腐蚀
            corrosion_coef_old = self.corrosion_coef
            self.corrosion_coef = self._calculate_corrosion(self.temperature_mid_list,corrosion_coef_old)
            self.sic_thickness_old = self.sic_thickness_raw / corrosion_coef_old
            self.sic_thickness_new = self.sic_thickness_raw / self.corrosion_coef

            # 更新燃料性能相关的参数
            self.sr_pyc_cumulant,self.st_pyc_cumulant,self.pyc_creep_coef,self.sic_weibull,self.sic_tensile_strength,self.sr,self.st = self._calculate_fuel_performance_material_properties(particle_material)

            # 计算应力分布
            self.stress_t_sic_new,self.stress_t_ipyc_new,self.stress_t_opyc_new = self._calculate_stress(self.pressure_new,self.sic_thickness_new,self.sr,self.st,self.pyc_creep_coef)
            self.stress_t_sic_old,self.stress_t_ipyc_old,self.stress_t_opyc_old = self._calculate_stress(self.pressure_old,self.sic_thickness_old,self.sr,self.st,self.pyc_creep_coef)
            
            # 计算承压失效率
            if self.time <= self.time_accident:
                self.sic_pressure_failure_fraction_new = self._calculate_weibull_pressure_failure(self.stress_t_sic_new,self.sic_tensile_strength,self.sic_weibull)
                self.sic_pressure_failure_fraction = self.sic_pressure_failure_fraction_new
            else:
                self.sic_pressure_failure_fraction_old = self._calculate_weibull_pressure_failure(self.stress_t_sic_old,self.sic_tensile_strength,self.sic_weibull)
                self.sic_pressure_failure_fraction_new = self._calculate_weibull_pressure_failure(self.stress_t_sic_new,self.sic_tensile_strength,self.sic_weibull)
                sic_pressure_failure_fraction_change = self.sic_pressure_failure_fraction_new - self.sic_pressure_failure_fraction_old
                if sic_pressure_failure_fraction_change > 0:
                    self.sic_pressure_failure_fraction += sic_pressure_failure_fraction_change
            
            # 计算热分解
            self.sic_thermal_decomposition_failure_fraction,self.activate_intgral = self._calculate_thermal_decomposition_failure_rate(self.temperature_mid_list[3],self.time_step,self.activate_intgral)
            
            # 总失效率
            self.sic_total_failure_fraction = 1 - (1-self.sic_manufacturing_failure_fraction)*(1-self.sic_pressure_failure_fraction)*(1-self.sic_thermal_decomposition_failure_fraction)
            
            # 输出失效率
            if self.sic_total_failure_fraction >= self.sic_output_failure_fraction_threshold:
                sic_output_failure_fraction = self.sic_total_failure_fraction
                while True:
                    self.sic_output_failure_fraction_threshold += self.failure_threshold
                    if self.sic_total_failure_fraction < self.sic_output_failure_fraction_threshold:
                        break
            else:
                if self.sic_output_failure_fraction_threshold == (self.failure_threshold + self.sic_manufacturing_failure_fraction):
                    sic_output_failure_fraction = self.sic_manufacturing_failure_fraction
                else:
                    sic_output_failure_fraction = self.sic_output_failure_fraction_threshold
            
            return sic_output_failure_fraction
        
        def _calculate_weibull_pressure_failure(self,stress_t_sic,sic_tensile_strength,sic_weibull):
            '''
                计算承压失效率
            '''
            stress_t_sic = np.longdouble(stress_t_sic) / 1e6 #[Pa]->[MPa]
            if stress_t_sic <= 0 :
                return 0.0
            mask = np.log(2)*np.exp(sic_weibull * np.log(stress_t_sic / sic_tensile_strength))
            if mask >= 150:
                mask = 150
            elif mask < 1e-8:
                return mask
            else:
                failure_rate = 1 - np.exp(-mask)
                return failure_rate

        def _calculate_fuel_performance_material_properties(self,particle_material):
            '''
                更新与燃料性能相关的参数
            '''
            # PyC热膨胀率
            sr_pyc_cumulant_new,st_pyc_cumulant_new,sr_new,st_new = particle_material.calculate_swelling_rate_pyc(self.time_step,self.fast_neutron_flux,self.fast_neutron_flux_cumulant,self.sr_pyc_cumulant,self.st_pyc_cumulant)
            # PyC蠕变系数
            pyc_creep_coef = particle_material.calculate_creep_coef_pyc(self.external_pyc_creep_coef,self.temperature_mid_list[2],self.temperature_mid_list[4],particle_material.density_particle_list[2],particle_material.density_particle_list[4])
            # SiC Weibull参数
            sic_weibull,sic_tensile_strength = particle_material.calculate_weibull_coef_sic(self.temperature_mid_list_irradiation[3],self.time,self.sic_tensile_strength_raw,self.sic_weibull_raw,self.fast_neutron_flux_cumulant,self.intergranular_corrosion)
            
            return sr_pyc_cumulant_new,st_pyc_cumulant_new,pyc_creep_coef,sic_weibull,sic_tensile_strength,sr_new,st_new

        def _calculate_pressure(self,temperature_buffer,CO_release_fraction,fp_release_fraction,burnup_fima):
            '''
                计算内压
            '''
            pressure = self.register.get('pressure',self.pressure_model)(temperature_buffer,self.yield_Xe,self.yield_Kr,CO_release_fraction,fp_release_fraction,self.volume_kernel_m3,self.volume_buffer_pore_m3,self.molar_volume_kernel,burnup_fima)

            return pressure

        def _calculate_corrosion(self,temperature_mid_list,corrosion_coef_old):
            '''
                SiC金属腐蚀
            '''
            corrosion_coef_new = self.register.get('sic_thickness',self.corrosion_model)(self.time,self.time_accident,temperature_mid_list,self.time_step,corrosion_coef_old,self.sic_thickness_raw)
            
            return corrosion_coef_new

        def _calculate_stress(self,pressure,sic_thickness,sr,st,pyc_creep_coef):
            '''
                计算颗粒应力
            '''
            if self.stress_model == 'Bubble':
                return self.register.get('stress',self.stress_model)(pressure,sic_thickness,self.r_average_sic)
            elif self.stress_model == 'RigidSiC':
                return self.register.get('stress',self.stress_model)(pressure,self.pyc_creep_poisson_ratio,self.pyc_poisson_ratio,sr,st,pyc_creep_coef,self.r_real)

        def _calculate_thermal_decomposition_failure_rate(self,temperature_sic,time_step,activate_intgral):
            '''
                SiC热分解失效率
            '''
            return self.register.get('thermal_decomposition',self.sic_thermal_decomposition_model)(self.time,self.time_accident,self.sic_thermal_decomposition_alpha,self.sic_thermal_decomposition_beta,self.sic_thickness_raw,temperature_sic,time_step,activate_intgral)
    
    @classmethod
    def init_shared_properties(cls,reader:MarsXMLReader):
        '''
            初始化共享参数
        '''
        # 总颗粒数
        cls.particle_number_total = reader.fuel_properties_dict['particle_number']
        # 温度份额/颗粒数
        cls.temperature_zone_number = reader.models_dict['intra_pebble_temperature']['temperature_zone_number']
        cls.temperature_subzone_volume_share = cls._temperature_share(reader)
        # 燃料性能
        cls.FuelPerformance.init_shared_properties(reader)
        
    @classmethod
    def _temperature_share(cls,reader):
        '''
            此温度颗粒占全部颗粒的份额
        '''
        element_geometry = Sphere.Geometry('element',reader)
        element_geometry.generate_mesh()
        r_span = np.linspace(0, element_geometry.r_real[1], cls.temperature_zone_number + 1)
        temperature_subzone_number = np.zeros(cls.temperature_zone_number if cls.temperature_zone_number != 0 else 1).astype(int)
        for i in range(element_geometry.mesh_number[0]):
            r_center = (element_geometry.r_num[i]+element_geometry.r_num[i+1]) / 2
            idx = np.digitize(r_center, r_span) - 1
            temperature_subzone_number[idx] += 1
        cum = np.concatenate(([0], np.cumsum(temperature_subzone_number)))
        temperature_subzone_idx = list(zip(cum[:-1], cum[1:]))
        temperature_subzone_volume = np.array([sum(element_geometry.volume_num[start:end]) for start,end in temperature_subzone_idx])
        temperature_subzone_volume_share = temperature_subzone_volume / element_geometry.volume_real[0]
        
        return temperature_subzone_volume_share

    def __init__(self,reader:MarsXMLReader,temperature_zone_idx):
        super().__init__('particle',reader)
        
        self.temperature_zone_idx = temperature_zone_idx
        self.temperature_share = self.temperature_subzone_volume_share[temperature_zone_idx]
        self.particle_number = self.temperature_share * self.particle_number_total
        # 材料属性
        self.material = self.MaterialProperties(reader,self.geometry)
        # 颗粒温度场/初始化材料属性
        self.temperature_field = ParticleTemperatureField()
        self.material.update_particle_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef','diffusion_coef'],temperature_num = self.temperature_field.temperature_num,time=0.0)
        # 颗粒核素扩散场/初始化浓度分布
        self.diffusion_field = DiffusionField('particle',reader,self.geometry,self.material.diffusion_coef_particle_num)
        self.diffusion_field.rupture_release_cumulant = 0.0
        # 颗粒的破损率
        self.fuel_performance = self.FuelPerformance(self.temperature_field.temperature_list)

class Kernel(Sphere):
    '''
        破损核芯
    '''
    class MaterialProperties(MarsMaterialProperties):
        
        def __init__(self,reader:MarsXMLReader,mesh_number):
            super().__init__(reader,'kernel')
            # 破损核芯网格信息
            self.mesh_number = mesh_number
            # 材料数值层索引区间
            self.kernel_idx = (0,self.mesh_number[0])
            # 属性
            self.properties_dict = {
                'diffusion_coef':self._update_diffusion_coef_kernel,
            }
        
        def _update_diffusion_coef_kernel(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            time = kwargs['time']
            # 更新核素扩散系数 [cm2]/[s]
            self.diffusion_coef_kernel_num = self.calculate_nuclide_diffusion_coef(temperature_num,self.mesh_number,time)
            diffusion_coef_kernel = np.mean(self.diffusion_coef_kernel_num[self.kernel_idx[0]:self.kernel_idx[1]])
            self.diffusion_coef_list = np.array([diffusion_coef_kernel])
        
        def update_material_properties(self,updated_properties,**kwargs):
            '''
                更新破损核芯属性
            '''
            for prop in updated_properties:
                self.properties_dict[prop](**kwargs)

    class DiffusionField(DiffusionField):
        
        def __init__(self, category, reader, geometry, diffusion_coef, particle:Particle):
            super().__init__(category, reader, geometry, diffusion_coef)
            # 覆盖为完整颗粒浓度
            self.concentration_num_fdm = particle.diffusion_field.concentration_num_fdm[0:sum(self.geometry.mesh_number)]
            self.concentration_num_fvm = particle.diffusion_field.concentration_num_fvm[0:sum(self.geometry.mesh_number)+1]
            self.inventory = particle.diffusion_field.inventory
            self.inventory_material_layer = [particle.diffusion_field.inventory_material_layer[0]]
            # 新破损颗粒调整时间步
            self.failure_adjustment_model = reader.solver_dict['failure_adjustment_model']
            self.failure_adjust_split_number = reader.solver_dict['failure_adjust_split_number'] if self.failure_adjustment_model == 'on' else 1
            self.failure_adjust_time_step_number = reader.solver_dict['failure_adjust_time_step_number'] if self.failure_adjustment_model == 'on' else 0
        
        def update_concentration_field(self, time_step, diffusion_coef_num):
            '''
                更新破损核芯的浓度场
            '''
            if self.failure_adjust_time_step_number > 0:
                self._update_adjust_concentration_field(time_step,diffusion_coef_num)
                self.failure_adjust_time_step_number -= 1
            else:
                if self.diffusion_model == 'Numerical':
                    self._fick_diffusion(time_step,diffusion_coef_num)
                elif self.diffusion_model == 'Booth':
                    self._booth(time_step,diffusion_coef_num)

        def _update_adjust_concentration_field(self,time_step,diffusion_coef_num):
            '''
                更新破损核芯的调整浓度场
            '''
            sub_time_step = time_step / self.failure_adjust_split_number
            concentration_num_fdm_new = self.concentration_num_fdm
            concentration_num_fvm_new = self.concentration_num_fvm
            inventory_new = self.inventory
            inventory_material_layer_new = self.inventory_material_layer
            generation_rate_num = self.generation_rate_num
            if self.diffusion_model == 'Numerical':
                for sub_step_idx in range(self.failure_adjust_time_step_number):
                    # 记录原始浓度场/盘存量
                    concentration_num_old_fdm = concentration_num_fdm_new
                    concentration_num_old_fvm = concentration_num_fvm_new
                    inventory_old = inventory_new
                    inventory_material_layer_old = inventory_material_layer_new
                    # 扩散计算/浓度场
                    concentration_num_fdm_new,concentration_num_fvm_new = self._calculate_diffusion(sub_time_step,concentration_num_old_fdm,concentration_num_old_fvm,diffusion_coef_num,generation_rate_num)
                    # 盘存量
                    inventory_material_layer_new,inventory_new = self._calculate_inventory(concentration_num_fdm_new,concentration_num_fvm_new)
                # 更新浓度与盘存量
                self.concentration_num_fdm = concentration_num_fdm_new
                self.concentration_num_fvm = concentration_num_fvm_new
                self.inventory = self.inventory
                self.inventory_material_layer = inventory_material_layer_new
                # 释放率计算
                self.diffusion_release_rate = self._calculate_diffusion_release_rate(self.concentration_num_fdm,self.concentration_num_fvm,diffusion_coef_num,self.geometry.r_num)
                self.diffusion_release_cumulant = self._calculate_release_cumulant(self.diffusion_release_rate,self.diffusion_release_cumulant,time_step)
                self.recoil_release_cumulant = self._calculate_release_cumulant(self.recoil_release_rate,self.recoil_release_cumulant,time_step)
                self.release_cumulant = self.diffusion_release_cumulant + self.recoil_release_cumulant
                self.release_rate = self.diffusion_release_rate + self.recoil_release_rate
            elif self.diffusion_model == 'Booth':
                self._booth(time_step,diffusion_coef_num)

    def __init__(self,reader:MarsXMLReader,particle:Particle,failure_particle_number):
        super().__init__('kernel',reader)
        # 温度份额/颗粒数
        self.temperature_zone_idx = particle.temperature_zone_idx
        self.particle_number = failure_particle_number
        # 材料属性
        self.material = self.MaterialProperties(reader,self.geometry.mesh_number)
        # 破损核芯温度场/初始化材料属性
        self.temperature_field = KernelTemperatureField(self.geometry,particle)
        self.material.update_material_properties(['diffusion_coef'],temperature_num = self.temperature_field.temperature_num,time=0.0)
        # 核素扩散场/初始化核素浓度
        self.diffusion_field = self.DiffusionField('kernel',reader,self.geometry,self.material.diffusion_coef_kernel_num,particle)
        
class GraphiteGrain(Sphere):
    '''
        石墨晶粒类型
    '''
    class MaterialProperties(MarsMaterialProperties):
        
        def __init__(self,reader:MarsXMLReader,mesh_number):
            super().__init__(reader,'graphite_grain')
            # 石墨晶粒网格信息
            self.mesh_number = mesh_number
            # 材料数值层索引区间
            self.graphite_grain_idx = (0,self.mesh_number[0])
            # 属性
            self.properties_dict = {
                'diffusion_coef':self._update_diffusion_coef_graphite_grain,
            }
        
        def _update_diffusion_coef_graphite_grain(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            time = kwargs['time']
            # 更新核素扩散系数 [cm2]/[s]
            self.diffusion_coef_graphite_grain_num = self.calculate_nuclide_diffusion_coef(temperature_num,self.mesh_number,time)
            diffusion_coef_graphite_grain = np.mean(self.diffusion_coef_graphite_grain_num[self.graphite_grain_idx[0]:self.graphite_grain_idx[1]])
            self.diffusion_coef_list = np.array([diffusion_coef_graphite_grain])
        
        def update_material_properties(self,updated_properties,**kwargs):
            '''
                更新石墨晶粒属性
            '''
            for prop in updated_properties:
                self.properties_dict[prop](**kwargs)

    def __init__(self,reader:MarsXMLReader):
        super().__init__('graphite_grain',reader)
        # 材料属性
        self.material = self.MaterialProperties(reader,self.geometry.mesh_number)
        # 晶粒温度场/初始化材料属性
        self.temperature_field = GraphiteGrainTemperatureField(self.geometry)
        self.material.update_material_properties(['diffusion_coef'],temperature_num = self.temperature_field.temperature_num,time=0.0)
        # 核素扩散场/初始化核素浓度
        self.diffusion_field = DiffusionField('graphite_grain',reader,self.geometry,self.material.diffusion_coef_graphite_grain_num)

class Element(Sphere):
    '''
        燃料元件类型
    '''
    class MaterialProperties(MarsMaterialProperties):

        def __init__(self,reader:MarsXMLReader,category,geometry:Sphere.Geometry,particle:Particle):
            super().__init__(reader,category)
            # 元件网格信息
            self.mesh_number = geometry.mesh_number
            self.pebble_diameter = geometry.r_num[-1]*2
            # 各材料的数值层索引区间
            self.fuel_zone_idx = (geometry.material_idx[0][0],geometry.material_idx[0][1])
            self.non_fuel_zone_idx = (geometry.material_idx[1][0],geometry.material_idx[1][1])
            # 颗粒的体积占比
            volume_fuel_zone = geometry.volume_real[0]
            self.fuel_zone_particle_material_volume_share = particle.particle_number_total * particle.geometry.volume_real / volume_fuel_zone
            self.fuel_zone_graphite_volume_share = 1 - sum(self.fuel_zone_particle_material_volume_share)
            # 属性
            self.properties_dict = {
                'thermal_conductivity':self._update_thermal_conductivity_element,
                'heat_capacity':self._update_heat_capacity_element,
                'density':self._update_density_element,
                'heat_diffusion_coef':self._update_heat_diffusion_coef_element,
                'diffusion_coef':self._update_diffusion_coef_element,
                'adsorp_coef':self._update_adsorp_coef_element,
                'dynamic_viscosity':self._update_dynamic_viscosity,
                'binary_diffusion_coef':self._update_binary_diffusion_coef,
                'coolant_density':self._update_coolant_density,
            }

        def _update_thermal_conductivity_element(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            particle_list = kwargs['particle_list']
            # 更新导热系数 [W]/[m][K]
            thermal_conductivity_non_fuel_zone_num = self.calculate_thermal_conductivity_graphite_matrix(temperature_num[self.non_fuel_zone_idx[0]:self.non_fuel_zone_idx[1]])
            thermal_conductivity_fuel_zone_num = self.calculate_thermal_conductivity_graphite_matrix(temperature_num[self.fuel_zone_idx[0]:self.fuel_zone_idx[1]])
            thermal_conductivity_non_fuel_zone = np.mean(thermal_conductivity_non_fuel_zone_num)
            thermal_conductivity_fuel_zone = np.mean(thermal_conductivity_fuel_zone_num) * self.fuel_zone_graphite_volume_share
            for particle in particle_list:
                thermal_conductivity_fuel_zone += particle.temperature_share * sum(particle.material.thermal_conductivity_particle_list * self.fuel_zone_particle_material_volume_share)
            self.thermal_conductivity_element_list = np.array([thermal_conductivity_fuel_zone,thermal_conductivity_non_fuel_zone])
            self.thermal_conductivity_element_num = np.repeat(self.thermal_conductivity_element_list,self.mesh_number)

        def _update_heat_capacity_element(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            particle_list = kwargs['particle_list']
            # 更新热容 [J]/[kg][K]
            heat_capacity_non_fuel_zone_num = self.calculate_heat_capacity_graphite_matrix(temperature_num[self.non_fuel_zone_idx[0]:self.non_fuel_zone_idx[1]])
            heat_capacity_fuel_zone_num = self.calculate_heat_capacity_graphite_matrix(temperature_num[self.fuel_zone_idx[0]:self.fuel_zone_idx[1]])
            heat_capacity_non_fuel_zone = np.mean(heat_capacity_non_fuel_zone_num)
            heat_capacity_fuel_zone = np.mean(heat_capacity_fuel_zone_num) * self.fuel_zone_graphite_volume_share
            for particle in particle_list:
                heat_capacity_fuel_zone += particle.temperature_share * sum(particle.material.heat_capacity_particle_list * self.fuel_zone_particle_material_volume_share)
            self.heat_capacity_element_list = np.array([heat_capacity_fuel_zone,heat_capacity_non_fuel_zone])
            self.heat_capacity_element_num = np.repeat(self.heat_capacity_element_list,self.mesh_number)
        
        def _update_density_element(self,**kwargs):
            particle_list = kwargs['particle_list']
            # 更新密度 [g]/[cm3]
            density_non_fuel_zone = self.calculate_density_graphite_matrix()
            density_fuel_zone = self.calculate_density_graphite_matrix() * self.fuel_zone_graphite_volume_share
            for particle in particle_list:
                density_fuel_zone += particle.temperature_share * sum(particle.material.density_particle_list * self.fuel_zone_particle_material_volume_share)
            self.density_element_list = np.array([density_fuel_zone,density_non_fuel_zone])
            self.density_element_num = np.repeat(self.density_element_list,self.mesh_number)
            
        def _update_heat_diffusion_coef_element(self,**kwargs):
            # 更新热扩散系数 [m2]/[s]
            density_num = self.density_element_num * 1000.0
            density_list = self.density_element_list * 1000.0
            self.heat_diffusion_coef_list = self.thermal_conductivity_element_list / density_list / self.heat_capacity_element_list
            self.heat_diffusion_coef_element_num = self.thermal_conductivity_element_num / density_num / self.heat_capacity_element_num 

        def _update_diffusion_coef_element(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            time = kwargs['time']
            # 更新核素扩散系数 [cm2]/[s]
            self.diffusion_coef_element_num = self.calculate_nuclide_diffusion_coef(temperature_num,self.mesh_number,time)
            diffusion_coef_fuel_zone = np.mean(self.diffusion_coef_element_num[self.fuel_zone_idx[0]:self.fuel_zone_idx[1]])
            diffusion_coef_non_fuel_zone = np.mean(self.diffusion_coef_element_num[self.non_fuel_zone_idx[0]:self.non_fuel_zone_idx[1]])
            self.diffusion_coef_list = np.array([diffusion_coef_fuel_zone,diffusion_coef_non_fuel_zone])

        def _update_adsorp_coef_element(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            c_fvm = kwargs['c_fvm']
            # 更新吸附分配系数
            self.adsorb_coef = self.calculate_adsorb_coef(temperature_num[-1],c_fvm)

        def _update_dynamic_viscosity(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            # 更新动力粘度
            self.dynamic_viscosity = self._calculate_dynamic_viscosity(temperature_num[-1])
        
        def _update_binary_diffusion_coef(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            # 更新二元扩散系数
            self.binary_diffusion_coef = self._calculate_binary_diffusion_coef(temperature_num[-1])
        
        def _update_coolant_density(self,**kwargs):
            temperature_num = kwargs['temperature_num']
            # 更新二元扩散系数
            self.coolant_density = self._calculate_coolant_density(temperature_num[-1])

        def update_mass_transfer_coef(self,temperature_num):
            '''
                更新边界层到主流的质量传递系数
            '''
            self.update_element_material_properties(['dynamic_viscosity','binary_diffusion_coef','coolant_density'],temperature_num=temperature_num)
            Sc = self.dynamic_viscosity/(self.coolant_density*self.binary_diffusion_coef)
            Re = self.coolant_density*self.coolant_velocity*(self.pebble_diameter*(1e-2))/self.dynamic_viscosity
            part_1 = 1.27*np.power(Sc,1/3)/np.power(self.epsilon,1.18)*np.power(Re,0.36)
            part_2 = 0.033*np.power(Sc,1/2)/np.power(self.epsilon,1.07)*np.power(Re,0.86)
            Sh = part_1 + part_2
            self.mass_transfer_coef = Sh*self.binary_diffusion_coef/(self.pebble_diameter*(1e-2)) * 1e2
            
        def update_element_material_properties(self,updated_properties,**kwargs):
            '''
                更新颗粒属性
            '''
            for prop in updated_properties:
                self.properties_dict[prop](**kwargs)

    class DiffusionField(DiffusionField):
        '''
            重写产生率方法与扩散方法
        '''
        def __init__(self, category, reader, geometry, diffusion_coef):
            super().__init__(category, reader, geometry, diffusion_coef)

        def update_generation_rate(self,element_generation_rate,temperature_subzone_idx,temperature_subzone_volume,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]],graphite_grain:GraphiteGrain):
            '''
                更新球体因裂变导致的核素产生率分布
            '''
            # 原始裂变产生率分布
            generation_rate_num_uranium_contamination_raw = self._calculate_uranium_contamination_generation_rate_raw(element_generation_rate)
           
            # 反冲修正
            self.generation_rate_num_uranium_contamination,self.recoil_release_rate = self._calculate_recoil_correction(generation_rate_num_uranium_contamination_raw)
            self.generation_rate_from_fission = np.sum(self.generation_rate_num_uranium_contamination * self.geometry.volume_num)
            # 颗粒的释放
            self.generation_rate_num_total_particle = np.zeros_like(self.generation_rate_num_uranium_contamination)
            for idx,particle in enumerate(particle_list):
                kernel_batch = kernel_batch_list[idx] 
                particle_release_rate = particle.diffusion_field.release_rate * particle.particle_number + particle.diffusion_field.rupture_release_rate * particle.particle_number * particle.fuel_performance.failure_fraction_increment
                failed_particle_release_rate = sum([kernel.diffusion_field.release_rate*kernel.particle_number for kernel in kernel_batch])
                # 颗粒的总释放
                region_total_particle_release_rate = particle_release_rate + failed_particle_release_rate
                # 子区体积
                region_volume = temperature_subzone_volume[idx]
                # 体积产生率
                volumetric_generation_rate_total_particle = region_total_particle_release_rate / region_volume
                self.generation_rate_num_total_particle[temperature_subzone_idx[idx][0]:temperature_subzone_idx[idx][1]] = volumetric_generation_rate_total_particle
            self.generation_rate_from_particle = np.sum(self.generation_rate_num_total_particle * self.geometry.volume_num)
            # 晶粒的释放
            self.generation_rate_num_graphite_grain = graphite_grain.diffusion_field.release_rate / self.geometry.volume_total * np.ones_like(self.generation_rate_num_uranium_contamination)
            self.generation_rate_from_graphite_grain = np.sum(self.generation_rate_num_graphite_grain * self.geometry.volume_num)
            # 总产生率分布
            self.generation_rate_num = self.generation_rate_num_uranium_contamination + self.generation_rate_num_total_particle + self.generation_rate_num_graphite_grain
            self.generation_rate = np.sum(self.generation_rate_num * self.geometry.volume_num)

    def __init__(self,reader:MarsXMLReader,particle_list:List[Particle]):
        super().__init__('element',reader)
        # 温度子区网格索引
        self.temperature_zone_number = reader.models_dict['intra_pebble_temperature']['temperature_zone_number']
        self._temperature_zone_idx()
        # 材料属性
        self.material = self.MaterialProperties(reader,'element',self.geometry,particle_list[0])
        # 时间依赖强制属性
        self.external_element_power_table = TimeSeries(times=reader.external_conditions_dict['element_power'][1], values=reader.external_conditions_dict['element_power'][0])
        self.external_temperature_table = TimeSeries(times=reader.external_conditions_dict['temperature'][1], values=reader.external_conditions_dict['temperature'][0])
        self.external_total_inventory_table = TimeSeries(times=reader.external_conditions_dict['inventory'][1],values=reader.external_conditions_dict['inventory'][0])
        self.external_burnup_table = TimeSeries(times=reader.external_conditions_dict['burnup'][1],values=reader.external_conditions_dict['burnup'][0])
        self.external_neutron_flux_table = TimeSeries(times=reader.external_conditions_dict['neutron_flux'][1],values=reader.external_conditions_dict['neutron_flux'][0])
        # 元件温度场/初始化材料属性
        self.temperature_field = ElementTemperatureField(self.geometry)
        self.material.update_element_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef','diffusion_coef'],temperature_num=self.temperature_field.temperature_num,particle_list=particle_list,time=0.0)
        # 核素扩散场/初始化核素浓度
        self.diffusion_field = self.DiffusionField('element',reader,self.geometry,self.material.diffusion_coef_element_num)

    def update_time_dependent_properties(self,time):
        '''
            更新与时间相关的强制外部条件属性
        '''
        # 燃料元件的功率
        self.element_power = self.external_element_power_table.at(time)
        self.external_temperature = self.external_temperature_table.at(time)
        self.total_inventory = self.external_total_inventory_table.at(time)
        self.burnup = self.external_burnup_table.at(time)
        self.neutron_flux = self.external_neutron_flux_table.at(time)
        self.element_generation_rate = self._calculate_element_generation_rate(time)

    def _calculate_element_generation_rate(self,time,tiny=1e-15):
        '''
            更新元件的总裂变核素产生率
        '''
        inventory_sequence = self.external_total_inventory_table._values
        time_sequence = self.external_total_inventory_table._times
        if len(inventory_sequence) <= 1:
            return 0.0
        idx = bisect.bisect_left(time_sequence, time)
        if idx >= len(inventory_sequence) or idx == 0:
            return 0.0
        lam = float(self.diffusion_field.decay_constant)
        if lam == 0.0:
            # lam=0 时模型退化为 dN/dt = q，可用差分近似
            dt = float(time_sequence[idx] - time_sequence[idx - 1])
            if abs(dt) < tiny:
                return 0.0
            return float(inventory_sequence[idx] - inventory_sequence[idx - 1]) / dt
        dt = float(time_sequence[idx] - time_sequence[idx - 1])
        if abs(dt) < tiny:
            return 0.0
        N0 = float(inventory_sequence[idx - 1])
        N1 = float(inventory_sequence[idx])
        x = lam * dt  
        denom = -np.expm1(-x)
        if abs(denom) < tiny:
            return (N1 - N0) / dt

        e = np.exp(-x)
        q = lam * (N1 - N0 * e) / denom
        return float(q)

    def _temperature_zone_idx(self):
        '''
            各温度子区的网格索引
        '''
        r_span = np.linspace(0, self.geometry.r_real[1], self.temperature_zone_number + 1)
        self.temperature_subzone_number = np.zeros(self.temperature_zone_number if self.temperature_zone_number != 0 else 1).astype(int)
        for i in range(self.geometry.mesh_number[0]):
            r_center = (self.geometry.r_num[i]+self.geometry.r_num[i+1]) / 2
            idx = np.digitize(r_center, r_span) - 1
            self.temperature_subzone_number[idx] += 1
        cum = np.concatenate(([0], np.cumsum(self.temperature_subzone_number)))
        self.temperature_subzone_idx = list(zip(cum[:-1], cum[1:]))
        self.temperature_subzone_volume = np.array([sum(self.geometry.volume_num[start:end]) for start,end in self.temperature_subzone_idx])
        self.temperature_subzone_volume_share = self.temperature_subzone_volume / self.geometry.volume_real[0]
      
class TemperatureField:
    '''
        温度场
    '''
    @classmethod
    def set_shared_properties(cls,reader:MarsXMLReader):
        '''
            初始化温度场共有属性
        '''
        cls.max_iterations_steady = reader.solver_dict['steady_temperature_field_max_iterations']
        cls.residual_steady = reader.solver_dict['steady_temperature_field_residual']
        cls.max_iterations_transient = reader.solver_dict['transient_temperature_field_max_iterations']
        cls.residual_transient = reader.solver_dict['transient_temperature_field_residual']
        cls.temperature_field_model = reader.models_dict['intra_pebble_temperature']['temperature_field_model']
        cls.transient_temerature_begin = reader.models_dict['intra_pebble_temperature']['transient_temperature_begin']
        cls.initial_temperature = reader.conditions_dict['initial_temperature']
        cls.temperature_zone_number = reader.models_dict['intra_pebble_temperature']['temperature_zone_number']

    @classmethod
    def _update_non_uniform_steady_temperature_field(cls,particle_list:List[Particle],element:Element,graphite_grain:GraphiteGrain,kernel_batch_list:List[List[Kernel]]):
        '''
            更新稳态非均匀温度场
        '''
        # 求解迭代收敛标准
        max_iterations = cls.max_iterations_steady
        residual = cls.residual_steady
        # 边界条件
        element_power = element.element_power
        element_surface_temperature = element.external_temperature
        # 准备数据
        temperature_num_particle_new_k_2d = [None for particle in particle_list]
        temperature_num_particle_new_k_1_2d = [None for particle in particle_list]
        subzone_idx = element.temperature_subzone_idx
        # 初始化热物性参数
        for particle in particle_list:
            particle.material.update_particle_material_properties(['thermal_conductivity'],temperature_num=np.ones_like(particle.geometry.volume_num)*(element_surface_temperature+20))
        element.material.update_element_material_properties(['thermal_conductivity'],temperature_num=np.ones_like(element.geometry.volume_num)*(element_surface_temperature+10),particle_list=particle_list)
        
        # 迭代稳态颗粒与石墨基体的温度场与导热系数
        for step in range(max_iterations):
            
            # 更新石墨基体的温度
            if step == 0:
                temperature_num_element_new_k = element.temperature_field._calculate_non_uniform_steady_temperature(element_power,element_surface_temperature,element.material)
                temperature_num_element_new_k_1 = temperature_num_element_new_k.copy()
            else:
                temperature_num_element_new_k = temperature_num_element_new_k_1.copy()
                temperature_num_element_new_k_1 = element.temperature_field._calculate_non_uniform_steady_temperature(element_power,element_surface_temperature,element.material)
                
            for idx,particle in enumerate(particle_list):
                # 更新各颗粒的温度
                particle_surface_temperature = np.mean(temperature_num_element_new_k_1[subzone_idx[idx][0]:subzone_idx[idx][1]])
                if cls.temperature_zone_number == 0:
                    particle_surface_temperature = temperature_num_element_new_k_1[0]
                temperature_num_particle_new = particle.temperature_field._calculate_non_uniform_steady_temperature(element_power,particle_surface_temperature,particle.material)
                if step == 0:
                    temperature_num_particle_new_k_2d[idx] = temperature_num_particle_new
                    temperature_num_particle_new_k_1_2d[idx] = temperature_num_particle_new
                else:
                    temperature_num_particle_new_k_2d[idx] = temperature_num_particle_new_k_1_2d[idx]
                    temperature_num_particle_new_k_1_2d[idx] = temperature_num_particle_new
                #更新各颗粒的导热系数
                particle.material.update_particle_material_properties(['thermal_conductivity'],temperature_num=temperature_num_particle_new)
               
            # 更新基体的导热系数
            element.material.update_element_material_properties(['thermal_conductivity'],temperature_num=temperature_num_element_new_k_1,particle_list=particle_list)
            if step == 0:
                temperature_num_particle_new_k_2d = np.vstack(temperature_num_particle_new_k_2d)
                temperature_num_particle_new_k_1_2d = np.vstack(temperature_num_particle_new_k_1_2d)
                continue
            else:
                if (np.linalg.norm(temperature_num_particle_new_k_1_2d-temperature_num_particle_new_k_2d,ord=np.inf) < residual) and (np.linalg.norm(temperature_num_element_new_k-temperature_num_element_new_k_1,ord=np.inf) < residual):
                    break

        # 更新石墨基体温度
        element.temperature_field.temperature_num = temperature_num_element_new_k_1
        # 更新颗粒温度
        for idx,particle in enumerate(particle_list):
            particle.temperature_field.temperature_num = temperature_num_particle_new_k_1_2d[idx]
        # 更新石墨晶粒温度
        graphite_grain.temperature_field._calculate_uniform_temperature(np.mean(element.temperature_field.temperature_num))
        # 更新破损颗粒温度
        for kernel_batch,particle in zip(kernel_batch_list,particle_list):
            for kernel in kernel_batch:
                kernel.temperature_field.temperature_num = kernel.temperature_field._give_kernel_temperature(kernel,particle)
        # 更新颗粒温度依赖材料信息
        for particle in particle_list:
            particle.material.update_particle_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=particle.temperature_field.temperature_num)
        # 更新石墨基体温度依赖材料信息
        element.material.update_element_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=element.temperature_field.temperature_num,particle_list=particle_list)
        # 更新石墨晶粒温度依赖材料信息
        graphite_grain.material.update_material_properties([],temperature_num=graphite_grain.temperature_field.temperature_num)
        # 更新破损核芯温度依赖材料信息
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.material.update_material_properties([],temperature_num=kernel.temperature_field.temperature_num)

    @classmethod
    def _update_derived_material_mean_temperature(cls,particle_list:List[Particle],element:Element,graphite_grain:GraphiteGrain,kernel_batch_list:List[List[Kernel]]):
        '''
            计算各球体各材料的平均温度
        '''
        # 基体两区温度
        element.temperature_field.temperature_list = np.array([np.mean(element.temperature_field.temperature_num[start:end]) for start, end in element.geometry.material_idx])
        # 颗粒五区温度
        for idx,particle in enumerate(particle_list):
            particle.temperature_field.temperature_list = np.array([np.mean(particle.temperature_field.temperature_num[start:end]) for start, end in particle.geometry.material_idx])
        # 核芯温度
        for kernel_batch,particle in zip(kernel_batch_list,particle_list):
            for kernel in kernel_batch:
                kernel.temperature_field.temperature_list = np.array([np.mean(kernel.temperature_field.temperature_num)])
        # 晶粒温度
        graphite_grain.temperature_field.temperature_list = np.array([np.mean(graphite_grain.temperature_field.temperature_num)])

    @classmethod
    def _update_non_uniform_transient_temperature_field(cls,time_step,particle_list:List[Particle],element:Element,graphite_grain:GraphiteGrain,kernel_batch_list:List[List[Kernel]]):
        '''
            更新瞬态非均匀温度场
        '''
        # 收敛标准
        max_iterations = cls.max_iterations_transient
        residual = cls.residual_transient
        # 边界条件
        element_power = element.element_power
        element_surface_temperature = element.external_temperature
        # 初始条件
        temperature_num_element_old = element.temperature_field.temperature_num
        # 准备数据
        temperature_num_particle_new_k_2d = [None for particle in particle_list]
        temperature_num_particle_new_k_1_2d = [None for particle in particle_list]
        subzone_idx = element.temperature_subzone_idx
        
        for step in range(max_iterations):    
            # 更新基体温度
            if step == 0:
                temperature_num_element_new_k = element.temperature_field._calculate_non_uniform_transient_temperature(time_step,temperature_num_element_old,element_power,element_surface_temperature,element.material)
                temperature_num_element_new_k_1 = temperature_num_element_new_k.copy()
            else:
                temperature_num_element_new_k = temperature_num_element_new_k_1.copy()
                temperature_num_element_new_k_1 = element.temperature_field._calculate_non_uniform_transient_temperature(time_step,temperature_num_element_old,element_power,element_surface_temperature,element.material)
            # 更新颗粒温度
            for idx,particle in enumerate(particle_list):
                particle_surface_temperature = np.mean(temperature_num_element_new_k_1[subzone_idx[idx][0]:subzone_idx[idx][1]])
                if cls.temperature_zone_number == 0:
                    particle_surface_temperature = temperature_num_element_new_k_1[0]
                if step == 0:
                    temperature_num_particle_new = particle.temperature_field._calculate_non_uniform_steady_temperature(element_power,particle_surface_temperature,particle.material)
                    temperature_num_particle_new_k_2d[idx] = temperature_num_particle_new
                    temperature_num_particle_new_k_1_2d[idx] = temperature_num_particle_new.copy()
                else:
                    temperature_num_particle_new_k_2d[idx] = temperature_num_particle_new_k_1_2d[idx]
                    temperature_num_particle_new_k_1_2d[idx] = particle.temperature_field._calculate_non_uniform_steady_temperature(element_power,particle_surface_temperature,particle.material)
            
            # 更新颗粒属性
            for idx,particle in enumerate(particle_list):
                particle.material.update_particle_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=temperature_num_particle_new_k_1_2d[idx])
            # 更新基体属性
            element.material.update_element_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=temperature_num_element_new_k_1,particle_list=particle_list)
            
            if step == 0:
                temperature_num_particle_new_k_2d = np.vstack(temperature_num_particle_new_k_2d)
                temperature_num_particle_new_k_1_2d = np.vstack(temperature_num_particle_new_k_1_2d)
                continue
            if (np.linalg.norm(np.vstack(temperature_num_particle_new_k_1_2d)-np.vstack(temperature_num_particle_new_k_2d) ,ord=np.inf) < residual) and (np.linalg.norm(temperature_num_element_new_k-temperature_num_element_new_k_1 ,ord=np.inf) < residual):
                break
        
        # 更新石墨基体温度
        element.temperature_field.temperature_num = temperature_num_element_new_k_1
        # 更新颗粒温度
        for idx,particle in enumerate(particle_list):
            particle.temperature_field.temperature_num = temperature_num_particle_new_k_1_2d[idx]
        # 更新石墨晶粒温度
        graphite_grain.temperature_field._calculate_uniform_temperature(np.mean(element.temperature_field.temperature_num))
        # 更新破损颗粒温度
        for kernel_batch,particle in zip(kernel_batch_list,particle_list):
            for kernel in kernel_batch:
                kernel.temperature_field.temperature_num = kernel.temperature_field._give_kernel_temperature(kernel,particle)

        # 更新颗粒温度依赖材料信息
        for particle in particle_list:
            particle.material.update_particle_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=particle.temperature_field.temperature_num)
        # 更新石墨基体温度依赖材料信息
        element.material.update_element_material_properties(['thermal_conductivity','heat_capacity','density','heat_diffusion_coef'],temperature_num=element.temperature_field.temperature_num,particle_list=particle_list)
        # 更新石墨晶粒温度依赖材料信息
        graphite_grain.material.update_material_properties([],temperature_num=graphite_grain.temperature_field.temperature_num)
        # 更新破损核芯温度依赖材料信息
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.material.update_material_properties([],temperature_num=kernel.temperature_field.temperature_num)

    @classmethod
    def _update_uniform_temperature_field(cls,particle_list:List[Particle],element:Element,graphite_grain:GraphiteGrain,kernel_batch_list:List[List[Kernel]]):
        '''
            更新均匀温度场
        '''
        uniform_temperature = element.external_temperature
        # 更新石墨基体温度
        element.temperature_field.temperature_num = element.temperature_field._calculate_uniform_temperature(uniform_temperature)
        # 更新颗粒温度
        for particle in particle_list:
            particle.temperature_field.temperature_num = particle.temperature_field._calculate_uniform_temperature(uniform_temperature)
        # 更新石墨晶粒温度
        graphite_grain.temperature_field.temperature_num = graphite_grain.temperature_field._calculate_uniform_temperature(uniform_temperature)
        # 更新破损核芯温度
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.temperature_field.temperature_num = kernel.temperature_field._calculate_uniform_temperature(uniform_temperature)

    @classmethod
    def update_temperature_field(cls,time_step,time,reader:MarsXMLReader,particle_list:List[Particle],element:Element,graphite_grain:GraphiteGrain,kernel_batch_list:List[List[Kernel]]):
        '''
            更新各球体的温度场
        '''
        if reader.models_dict['intra_pebble_temperature']['temperature_field_model'] == 'uniform':
            cls._update_uniform_temperature_field(particle_list,element,graphite_grain,kernel_batch_list)
        elif reader.models_dict['intra_pebble_temperature']['temperature_field_model'] == 'non_uniform':
            if time < reader.models_dict['intra_pebble_temperature']['transient_temperature_begin']:
                cls._update_non_uniform_steady_temperature_field(particle_list,element,graphite_grain,kernel_batch_list)
            else:
                cls._update_non_uniform_transient_temperature_field(time_step,particle_list,element,graphite_grain,kernel_batch_list)
        # 更新材料平均温度
        cls._update_derived_material_mean_temperature(particle_list,element,graphite_grain,kernel_batch_list)

class ParticleTemperatureField(TemperatureField):
    '''
        颗粒温度场
    '''
    @classmethod
    def set_particle_shared_properties(cls,reader:MarsXMLReader):
        '''
            设置颗粒温度场的共享属性
        '''
        particle_number_total = reader.fuel_properties_dict['particle_number']
        particle_geometry = Sphere.Geometry('particle',reader)
        particle_geometry.generate_mesh()
        # 颗粒几何
        cls.geometry = particle_geometry
        # 网格信息
        cls.mesh_number = particle_geometry.mesh_number
        # 颗粒总数
        cls.particle_number_total = particle_number_total
        # 用于温度场计算的几何半径([cm]->[m])
        cls.r_kernel = cls.geometry.r_real[1] / 100
        cls.r_buffer = cls.geometry.r_real[2] / 100
        cls.r_ipyc = cls.geometry.r_real[3] / 100
        cls.r_sic = cls.geometry.r_real[4] / 100
        cls.r_opyc = cls.geometry.r_real[5] / 100
        cls.r_num = cls.geometry.r_num / 100
        # 用于计算的核芯体积
        cls.kernel_volume = cls.geometry.volume_real[0]/(1e6)
        # 各材料的数值层索引区间
        cum = np.cumsum(cls.mesh_number)
        cls.kernel_idx = (0, cum[0])
        cls.buffer_idx = (cum[0], cum[1])
        cls.ipyc_idx   = (cum[1], cum[2])
        cls.sic_idx    = (cum[2], cum[3])
        cls.opyc_idx   = (cum[3], cum[4])

    def __init__(self):
        super().__init__()
        # 初始化温度场
        self.temperature_num = np.ones(sum(self.mesh_number)) * (self.initial_temperature)
        self.temperature_list = np.ones_like(self.mesh_number) * (self.initial_temperature)

    def _calculate_non_uniform_steady_temperature(self,element_power,particle_surface_temperature,particle_material:Particle.MaterialProperties):
        '''
            计算完整颗粒内部的温度场
        '''
        power_density = element_power / self.particle_number_total / self.kernel_volume
        k_kernel = particle_material.thermal_conductivity_particle_list[0]
        k_buffer = particle_material.thermal_conductivity_particle_list[1]
        k_ipyc = particle_material.thermal_conductivity_particle_list[2]
        k_sic = particle_material.thermal_conductivity_particle_list[3]
        k_opyc = particle_material.thermal_conductivity_particle_list[4]
        temperature_num = np.zeros_like(self.geometry.volume_num)
        
        # 核芯温度
        def _kernel_temperature(i):
            part_1 = (1/self.r_sic - 1/self.r_opyc) / k_opyc + (1/self.r_ipyc - 1/self.r_sic) / k_sic + (1/self.r_buffer - 1/self.r_ipyc) / k_ipyc + (1/self.r_kernel - 1/self.r_buffer) / k_buffer
            part_2 = (1/self.r_kernel - ((self.r_num[i]/2+self.r_num[i+1]/2)**2)/(self.r_kernel**3) ) / (2*k_kernel)
            temperature = particle_surface_temperature + power_density * (self.r_kernel**3) / 3 * (part_1+part_2)
            return temperature
        # 缓冲层温度
        def _buffer_temperature(i):
            part_1 = (1/self.r_sic - 1/self.r_opyc) / k_opyc + (1/self.r_ipyc - 1/self.r_sic) / k_sic + (1/self.r_buffer - 1/self.r_ipyc) / k_ipyc + (1/(self.r_num[i]/2+self.r_num[i+1]/2) - 1/self.r_buffer) / k_buffer
            part_2 = power_density*(self.r_kernel**3) / 3
            temperature = particle_surface_temperature + part_1 * part_2
            return temperature
        # IPyC温度
        def _ipyc_temperature(i):
            part_1 = (1/self.r_sic - 1/self.r_opyc) / k_opyc + (1/self.r_ipyc - 1/self.r_sic) / k_sic + (1/(self.r_num[i]/2+self.r_num[i+1]/2) - 1/self.r_ipyc) / k_ipyc
            part_2 = power_density*(self.r_kernel**3) / 3
            temperature = particle_surface_temperature + part_1 * part_2
            return temperature
        # SiC温度
        def _sic_temperature(i):
            part_1 = (1/self.r_sic - 1/self.r_opyc) / k_opyc + (1/(self.r_num[i]/2+self.r_num[i+1]/2) - 1/self.r_sic) / k_sic
            part_2 = power_density*(self.r_kernel**3) / 3
            temperature = particle_surface_temperature + part_1 * part_2
            return temperature
        # OPyC温度
        def _opyc_temperature(i):
            part_1 = (1/(self.r_num[i]/2+self.r_num[i+1]/2) - 1/self.r_opyc) / k_opyc
            part_2 = power_density*(self.r_kernel**3) / 3
            temperature = particle_surface_temperature + part_1 * part_2
            return temperature
        
        # 核芯温度
        temperature_num[self.kernel_idx[0]:self.kernel_idx[1]] = [_kernel_temperature(i) for i in range(self.kernel_idx[0],self.kernel_idx[1])]
        # buffer温度
        temperature_num[self.buffer_idx[0]:self.buffer_idx[1]] = [_buffer_temperature(i) for i in range(self.buffer_idx[0],self.buffer_idx[1])]
        # IPyC温度
        temperature_num[self.ipyc_idx[0]:self.ipyc_idx[1]] = [_ipyc_temperature(i) for i in range(self.ipyc_idx[0],self.ipyc_idx[1])]
        # SiC温度
        temperature_num[self.sic_idx[0]:self.sic_idx[1]] = [_sic_temperature(i) for i in range(self.sic_idx[0],self.sic_idx[1])]
        # OPyC温度
        temperature_num[self.opyc_idx[0]:self.opyc_idx[1]] = [_opyc_temperature(i) for i in range(self.opyc_idx[0],self.opyc_idx[1])] 

        return temperature_num

    def _calculate_uniform_temperature(self,uniform_temperature):
        '''
            颗粒的均匀温度
        '''
        temperature_num = np.ones_like(self.geometry.volume_num) * uniform_temperature

        return temperature_num

class ElementTemperatureField(TemperatureField):
    '''
        石墨基体温度场
    '''
    def __init__(self,element_geometry:Element.Geometry):
        super().__init__()
        # 元件几何
        self.geometry = element_geometry
        # 网格
        self.mesh_number = self.geometry.mesh_number
        # 用于温度计算的体积([cm3]->[m3])
        self.volume_fuel_zone = self.geometry.volume_real[0] / (1e6)
        self.volume_num = self.geometry.volume_num / (1e6)
        # 用于温度计算的面积([cm2]->[m2])
        self.area_num = self.geometry.area_num / (1e4)
        # 用于温度计算的半径([cm]->[m])
        self.r_fuel_zone = self.geometry.r_real[1] / 100
        self.r_non_fuel_zone = self.geometry.r_real[2] / 100
        self.r_num = self.geometry.r_num / 100
        # 各材料的数值层索引区间
        cum = np.cumsum(self.mesh_number)
        self.fuel_zone_idx = (0, cum[0])
        self.non_fuel_zone_idx = (cum[0], cum[1])
        # 初始化温度场
        self.temperature_num = np.ones(sum(self.mesh_number)) * (self.initial_temperature)
        self.temperature_list = np.ones_like(self.mesh_number) * (self.initial_temperature)

    def _calculate_non_uniform_steady_temperature(self,element_power,element_surface_temperature,element_material:Element.MaterialProperties):
        '''
            计算石墨基体内部温度场       
        '''
        power_density = element_power / self.volume_fuel_zone
        k_fuel_zone = element_material.thermal_conductivity_element_list[0]
        k_non_fuel_zone = element_material.thermal_conductivity_element_list[1]
        temperature_num = np.zeros_like(self.geometry.volume_num)

        # 燃料区温度
        def _fuel_zone_temperature(i):
            r_center = (self.r_num[i] + self.r_num[i+1]) / 2
            part_1 = (1/self.r_fuel_zone - 1/self.r_non_fuel_zone) / k_non_fuel_zone + (1/self.r_fuel_zone - (r_center**2)/(self.r_fuel_zone**3)) / (2*k_fuel_zone)
            part_2 = power_density * (self.r_fuel_zone**3) / 3
            temperature = element_surface_temperature + part_1 * part_2
            return temperature
        # 非燃料区
        def _non_fuel_zone_temperature(i):
            r_center = (self.r_num[i] + self.r_num[i+1]) / 2
            part_1 = (1/r_center - 1/self.r_non_fuel_zone) / k_non_fuel_zone
            part_2 = power_density * (self.r_fuel_zone**3) / 3
            temperature = element_surface_temperature + part_1 * part_2
            return temperature
        
        # 燃料区温度
        temperature_num[self.fuel_zone_idx[0]:self.fuel_zone_idx[1]] = [_fuel_zone_temperature(i) for i in range(self.fuel_zone_idx[0],self.fuel_zone_idx[1])]
        # 非燃料区温度
        temperature_num[self.non_fuel_zone_idx[0]:self.non_fuel_zone_idx[1]] = [_non_fuel_zone_temperature(i) for i in range(self.non_fuel_zone_idx[0],self.non_fuel_zone_idx[1])]

        return temperature_num
    
    def _calculate_non_uniform_transient_temperature(self,time_step,temperature_num_element_old,element_power,element_surface_temperature,element_material:Element.MaterialProperties):
        '''
            计算瞬态温度场
        '''
        x_env = element_surface_temperature
        time_step = time_step
        decay_constant = 0
        boundary_transport_coef = None
        x_k = temperature_num_element_old
        x_k = np.append(x_k[0],x_k)
        D = element_material.heat_diffusion_coef_element_num
        r = self.r_num
        volume = self.volume_num
        area = self.area_num
        gamma = self.geometry.gamma
        Q = element_power / self.volume_fuel_zone
        Q = Q / ((element_material.density_element_num*1000)*element_material.heat_capacity_element_num)
        Q[self.non_fuel_zone_idx[0]:self.non_fuel_zone_idx[1]] = 0.0
        temperature_num_next = MarsSolver._fvm_euler_thomas(x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,None,area,gamma)
        
        return temperature_num_next[0]
    
    def _calculate_uniform_temperature(self,uniform_temperature):
        '''
            石墨基体的均匀温度场
        '''
        temperature_num = np.ones_like(self.geometry.volume_num) * uniform_temperature

        return temperature_num
    
class GraphiteGrainTemperatureField(TemperatureField):
    '''
        石墨晶粒的温度场
    '''
    def __init__(self,graphite_grain_geometry:GraphiteGrain.Geometry):
        super().__init__()
        # 颗粒几何 
        self.geometry = graphite_grain_geometry
        # 初始化温度场
        self.temperature_num = np.ones(sum(self.geometry.mesh_number)) * (self.initial_temperature)
        self.temperature_list = np.ones_like(self.geometry.mesh_number) * (self.initial_temperature)

    def _calculate_uniform_temperature(self,uniform_temperature):
        '''
            晶粒的均匀温度场
        '''
        temperature_num = np.ones_like(self.geometry.volume_num) * uniform_temperature

        return temperature_num

class KernelTemperatureField(TemperatureField):
    '''
        破损核芯的温度场
    '''
    def __init__(self,kernel_geometry:Kernel.Geometry,particle:Particle):
        super().__init__()
        # 核芯几何
        self.geometry = kernel_geometry
        # 初始化温度场
        self.temperature_num = particle.temperature_field.temperature_num[0:sum(self.geometry.mesh_number)]
        self.temperature_list = np.ones_like(kernel_geometry.mesh_number) * np.mean(self.temperature_num)

    def _calculate_uniform_temperature(self,uniform_temperature):
        '''
            破损核芯的均匀温度场
        '''
        temperature_num = np.ones_like(self.geometry.volume_num) * uniform_temperature

        return temperature_num

    def _give_kernel_temperature(self,kernel:Kernel,particle:Particle):
        '''
            赋值破损核芯温度场
        '''
        if kernel.temperature_zone_idx == particle.temperature_zone_idx:
            temperature_num = particle.temperature_field.temperature_num[0:kernel.geometry.mesh_number[0]]
            return temperature_num
        else:
            print('破损核芯与完整颗粒温度区间不匹配')
            sys.exit()


 
