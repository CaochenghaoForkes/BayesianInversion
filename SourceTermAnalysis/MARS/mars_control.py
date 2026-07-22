from MARS.mars_xml_reader import MarsXMLReader
from MARS.mars_spherical import Particle,Element,GraphiteGrain,Kernel,Sphere
from typing import List
from TOOL.cosmos_general_function import round_sig,write_header,formatting_line
import os
import shutil

class MarsControl:
    '''
        控制程序的时间步进,输出
    '''
    def __init__(self,reader:MarsXMLReader):
        # 基本信息
        self.time_list = reader.control_dict['time']
        self.time_step_list = reader.control_dict['time_step']
        self.output_dir = reader.control_dict['output_path']
        self.write_interval_list = reader.control_dict['write_interval']
        self.output_type = reader.control_dict['output_type']
        self.simulation_mode = reader.control_dict['simulation_mode']
        self.nuclide = reader.external_conditions_dict['nuclide']
        self.finish_time = round_sig(sum(self.time_list),sig=9)
        self.time_step = round_sig(self.time_step_list[0],sig=9)
        self.time_span = round_sig(self.time_list[0],sig=9)
        self.write_interval = round_sig(self.write_interval_list[0],sig=9)
        self.distribution_type = reader.control_dict['distribution_type']
        self.distribution_time = list(reader.control_dict['distribution_time'])
        self.output_distribution_time = self.distribution_time[0] if len(self.distribution_time)>=1 else 1e30
        self.finish_swtich = False
        self.write_time = self.write_interval
        self.time_tot = 0.0
        self.phase = 0
        self.step_tot = 0
        self.release_rate_list = []
        self.output_time_list = []
        # 创建输出目录
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        self._output_txt_path()
        
    def _output_txt_path(self):
        '''
            输出txt文件的路径
        '''
        if 'ReleaseRate' in self.output_type:
            self.rupture_release_rate_path = os.path.join(self.output_dir,'RuptureReleaseRate.txt')
            self.recoil_release_rate_path = os.path.join(self.output_dir,'RecoilReleaseRate.txt')
            self.diffusion_release_rate_path = os.path.join(self.output_dir,'DiffusionReleaseRate.txt')
            self.release_rate_path = os.path.join(self.output_dir,'ReleaseRate.txt')
            self.f_rupture_release_rate = open(self.rupture_release_rate_path,'a',buffering=1)
            self.f_recoil_release_rate = open(self.recoil_release_rate_path,'a',buffering=1)
            self.f_diffusion_release_rate = open(self.diffusion_release_rate_path,'a',buffering=1)
            self.f_release_rate = open(self.release_rate_path,'a',buffering=1)
        if 'ReleaseCumulant' in self.output_type:
            self.rupture_release_cumulant_path = os.path.join(self.output_dir,'RuptureReleaseCumulant.txt')
            self.recoil_release_cumulant_path = os.path.join(self.output_dir,'RecoilReleaseCumulant.txt')
            self.diffusion_release_cumulant_path = os.path.join(self.output_dir,'DiffusionReleaseCumulant.txt')
            self.release_cumulant_path = os.path.join(self.output_dir,'ReleaseCumulant.txt')
            self.f_rupture_release_cumulant = open(self.rupture_release_cumulant_path,'a',buffering=1)
            self.f_recoil_release_cumulant = open(self.recoil_release_cumulant_path,'a',buffering=1)
            self.f_diffusion_release_cumulant = open(self.diffusion_release_cumulant_path,'a',buffering=1)
            self.f_release_cumulant = open(self.release_cumulant_path,'a',buffering=1)
        if 'Inventory' in self.output_type:
            self.inventory_path = os.path.join(self.output_dir,'Inventory.txt')
            self.inventory_material_layer_path = os.path.join(self.output_dir,'InventoryMaterialLayer')
            self.f_inventory = open(self.inventory_path,'a',buffering=1)
            self.f_inventory_material_layer = open(self.inventory_material_layer_path,'a',buffering=1)
        if 'Temperature' in self.output_type:
            self.temperature_path = os.path.join(self.output_dir,'Temperature.txt')
            self.f_temperature = open(self.temperature_path,'a',buffering=1)
        if 'DiffusionCoef' in self.output_type:
            self.diffusion_coef_path = os.path.join(self.output_dir,'DiffusionCoef.txt')
            self.f_diffusion_coef = open(self.diffusion_coef_path,'a',buffering=1)
        if 'FuelPerformance' in self.output_type:
            self.performance_path = os.path.join(self.output_dir,'Performance.txt')
            self.f_performance = open(self.performance_path,'a',buffering=1)
        if 'FPGeneration' in self.output_type:
            self.fp_generation_path = os.path.join(self.output_dir,'FPGeneration.txt')
            self.f_fp_generation = open(self.fp_generation_path,'a',buffering=1)
        
        if 'Concentration' in self.distribution_type:
            self.concentration_folder = os.path.join(self.output_dir,'ConcentrationDistribution')
            os.mkdir(self.concentration_folder)
        if 'Temperature' in self.distribution_type:
            self.temperature_folder = os.path.join(self.output_dir,'TemperatureDistribution')
            os.mkdir(self.temperature_folder)
        if 'DiffusionCoef' in self.distribution_type:
            self.diffusion_coef_folder = os.path.join(self.output_dir,'DiffusionCoefDistribution')
            os.mkdir(self.diffusion_coef_folder)
        if 'HeatDiffusionCoef' in self.distribution_type:
            self.heat_diffusion_coef_folder = os.path.join(self.output_dir,'HeatDiffusionCoefDistribution')
            os.mkdir(self.heat_diffusion_coef_folder)
        if 'ThermalConductivity' in self.distribution_type:
            self.thermal_conductivity_folder = os.path.join(self.output_dir,'ThermalConductivityDistribution')
            os.mkdir(self.thermal_conductivity_folder)
        if 'FPGeneration' in self.distribution_type:
            self.fp_generation_folder = os.path.join(self.output_dir,'FPGenerationDistribution')
            os.mkdir(self.fp_generation_folder)
            
    def procced(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            推进时间步并记录数据
        '''
        
        if self.time_tot >= self.time_span:
            self.phase += 1
            if self.phase >= len(self.time_list):
                # 模拟结束
                self.finish_swtich = True
            else:
                # 进入下一阶段
                self.write_interval = self.write_interval_list[self.phase]
                self.time_span = self.time_span+self.time_list[self.phase]
                self.time_step = self.time_step_list[self.phase]
                if self.time_step >= self.time_list[self.phase]:
                    self.time_step = self.time_list[self.phase]/2
        if self.simulation_mode == 'independent':
            output_str = f'Cosmos[Mars] running: {self.time_tot/self.finish_time*100:.3e}% time: {self.time_tot:.3e}s'
        elif 'tracer' in self.simulation_mode:
            output_str = f'Cosmos[Mars] running: {self.time_tot/self.finish_time*100:.3e}% {self.simulation_mode} {self.nuclide}'
        elif 'fsar' == self.simulation_mode:
            output_str = f'Cosmos[Mars] running: {self.time_tot/self.finish_time*100:.3e}% FSAR mode {self.nuclide}'
        if self.write_interval < 0:
            if self.step_tot % -self.write_interval == 0:
                #print(output_str)
                # 记录当前信息
                self._record(element,graphite_grain,particle_list,kernel_batch_list)
        elif self.write_interval > 0:
            if self.time_tot >= self.write_time:
                self.write_time += self.write_interval
                #print(output_str)
                # 记录当前信息
                self._record(element,graphite_grain,particle_list,kernel_batch_list)
        # 推进时间步
        if round_sig(self.time_tot + self.time_step,sig=9) <= self.time_span:
            self.time_tot = self.time_tot + self.time_step
        else:
            self.time_step = self.time_span - self.time_tot
            self.time_tot = self.time_span
        self.step_tot = self.step_tot + 1
        if self.finish_swtich:
            self._close_and_write_header()
        
    def _record(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录数据
        '''
        # 记录时间与元件释放率
        self.output_time_list.append(self.time_tot)
        self.release_rate_list.append(element.diffusion_field.release_rate)
        # 将球体展成列表/名称
        sphere_list = [element,graphite_grain] + particle_list
        self.sphere_name_list = ['Element','GraphiteGrain'] + [f'Intact(T:{particle.temperature_zone_idx})' for particle in particle_list]
        self.sphere_name_material_list = ['FuelZone','NonFuelZone','GraphiteGrain']
        self.particle_name_list = [f'Intact(T:{particle.temperature_zone_idx})' for particle in particle_list]
        for particle in particle_list:
            self.sphere_name_material_list += [f'Kernel(T:{particle.temperature_zone_idx})',f'Buffer(T:{particle.temperature_zone_idx})',f'IPyC(T:{particle.temperature_zone_idx})',f'SiC(T:{particle.temperature_zone_idx})',f'OPyC(T:{particle.temperature_zone_idx})']
        for temperature_zone_idx,kernel_batch in enumerate(kernel_batch_list):
            sphere_list += kernel_batch
            self.sphere_name_list += [f'Failed(T:{temperature_zone_idx}B:{batch})' for batch in range(len(kernel_batch))]
            self.sphere_name_material_list += [f'Failed(T:{temperature_zone_idx}B:{batch})' for batch in range(len(kernel_batch))]

        # 输出
        if 'ReleaseRate' in self.output_type:
            self._record_release_rate(sphere_list,particle_list,kernel_batch_list)
        if 'ReleaseCumulant' in self.output_type:
            self._record_release_cumulant(sphere_list,particle_list,kernel_batch_list)
        if 'Inventory' in self.output_type:
            self._record_inventory(sphere_list,particle_list,kernel_batch_list,element,graphite_grain)
        if 'Temperature' in self.output_type:
            self._record_temperature(sphere_list,particle_list,kernel_batch_list,element)
        if 'DiffusionCoef' in self.output_type:
            self._record_diffusion_coef(sphere_list,particle_list,kernel_batch_list)
        if 'FuelPerformance' in self.output_type:
            self._record_performance(sphere_list,particle_list,kernel_batch_list)
        if 'FPGeneration' in self.output_type:
            if len(kernel_batch_list[0]) != 0:
                kernel = kernel_batch_list[0][0]
            else:
                kernel = None
            self._record_fp_generation(element,kernel,particle_list[0],graphite_grain)
        
        # 输出分布
        if self.time_tot >= self.output_distribution_time:
            idx = self.distribution_time.index(self.output_distribution_time)
            self.output_distribution_time = self.distribution_time[idx+1] if idx+1 < len(self.distribution_time) else 1e30
            if 'Concentration' in self.distribution_type:
                self._record_concentration_distribution(element,graphite_grain,particle_list,kernel_batch_list)
            if 'Temperature' in self.distribution_type:
                self._record_temperature_distribution(element,graphite_grain,particle_list,kernel_batch_list)
            if 'DiffusionCoef' in self.distribution_type:
                self._record_diffusion_coef_distribution(element,graphite_grain,particle_list,kernel_batch_list)
            if 'HeatDiffusionCoef' in self.distribution_type:
                self._record_heat_diffusion_coef_distribution(element,graphite_grain,particle_list,kernel_batch_list)
            if 'ThermalConductivity' in self.distribution_type:
                self._record_thermal_conductivity_distribution(element,graphite_grain,particle_list,kernel_batch_list)
            if 'FPGeneration' in self.distribution_type:
                self._record_fp_generation_distribution(element,graphite_grain,particle_list,kernel_batch_list)

    def _close_and_write_header(self):
        '''
            写列头并关闭IO
        '''
        if 'ReleaseRate' in self.output_type:
            self.f_rupture_release_rate.close()
            self.f_recoil_release_rate.close()
            self.f_diffusion_release_rate.close()
            self.f_release_rate.close()
            write_header(self.rupture_release_rate_path,'RuptureReleaseRate([Bq]/[s])[破裂释放率的输出可能与累积释放率,燃料性能,基体产生率不一致,这是因为该释放率不连续,输出时间步可能会跳过破裂时刻]\n',formatting_line(['Time(s)']+self.particle_name_list))
            write_header(self.recoil_release_rate_path,'RecoilReleaseRate([Bq]/[s])\n',formatting_line(['Time(s)']+self.sphere_name_list))
            write_header(self.diffusion_release_rate_path,'DiffusionReleaseRate([Bq]/[s])\n',formatting_line(['Time(s)']+self.sphere_name_list))
            write_header(self.release_rate_path,'ReleaseRate([Bq]/[s])\n',formatting_line(['Time(s)']+self.sphere_name_list))
        if 'ReleaseCumulant' in self.output_type:
            self.f_rupture_release_cumulant.close()
            self.f_recoil_release_cumulant.close()
            self.f_diffusion_release_cumulant.close()
            self.f_release_cumulant.close()
            write_header(self.rupture_release_cumulant_path,'RuptureReleaseCumulant([Bq])\n',formatting_line(['Time(s)']+self.particle_name_list))
            write_header(self.recoil_release_cumulant_path,'RecoilReleaseCumulant([Bq])\n',formatting_line(['Time(s)']+self.sphere_name_list))
            write_header(self.diffusion_release_cumulant_path,'DiffusionReleaseCumulant([Bq])\n',formatting_line(['Time(s)']+self.sphere_name_list))
            write_header(self.release_cumulant_path,'ReleaseCumulant([Bq])\n',formatting_line(['Time(s)']+self.sphere_name_list))
        if 'Inventory' in self.output_type:
            self.f_inventory.close()
            self.f_inventory_material_layer.close()
            write_header(self.inventory_path,'Inventory([Bq])\n',formatting_line(['Time(s)']+self.sphere_name_list+['TotalInventory']+['TotalInventoryCalByNum']))
            write_header(self.inventory_material_layer_path,'InventoryMaterialLyer([Bq])\n',formatting_line(['Time(s)']+self.sphere_name_material_list))
        if 'Temperature' in self.output_type:
            self.f_temperature.close()
            write_header(self.temperature_path,'Temperature([K])\n',formatting_line(['Time(s)']+self.sphere_name_material_list+['Power(W)']))
        if 'DiffusionCoef' in self.output_type:
            self.f_diffusion_coef.close()
            write_header(self.diffusion_coef_path,'DiffusionCoef([cm2]/[s])\n',formatting_line(['Time(s)']+self.sphere_name_material_list))
        if 'FuelPerformance' in self.output_type:
            self.f_performance.close()
            write_header(self.performance_path,'FuelPerformance\n',formatting_line(['Time(s)']+self.fuel_performance_name_list))
        if 'FPGeneration' in self.output_type:
            self.f_fp_generation.close()
            write_header(self.fp_generation_path,'FPGeneration([Bq]/[s])\n',formatting_line(['Time(s)']+['TotalGeneration']+['ElementTotal','ElementFission','ElementFromParticle','ElementFromGrain']+['Intact']+['Failed']+['Grain']))
    
    def _record_concentration_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录浓度分布
        '''
        txt_path = os.path.join(self.concentration_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('Concentration Distribution([Bq]/[cm3])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num)))
        f.write(formatting_line(['C']+list(element.diffusion_field.concentration_num_fvm)))
        f.write('-----Graphite Grain-----\n')
        f.write(formatting_line(['R([cm])']+list(graphite_grain.geometry.r_num)))
        f.write(formatting_line(['C']+list(graphite_grain.diffusion_field.concentration_num_fvm)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'C(T:{idx})']+list(particle.diffusion_field.concentration_num_fvm)))
        if len(kernel_batch_list) > 0 and len(kernel_batch_list[0]) > 0:
            f.write('-----Failed Particle-----\n')
            f.write(formatting_line(['R([cm])']+list(kernel_batch_list[0][0].geometry.r_num)))
        for idx,kernel_batch in enumerate(kernel_batch_list):
            for batch,kernel in enumerate(kernel_batch):
                f.write(formatting_line([f'C(T:{idx}B:{batch})']+list(kernel.diffusion_field.concentration_num_fvm)))
        f.close()

    def _record_temperature_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录温度分布
        '''
        txt_path = os.path.join(self.temperature_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('Temperature Distribution([K])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num[1:]/2+element.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['T']+list(element.temperature_field.temperature_num)))
        f.write('-----Graphite Grain-----\n')
        f.write(formatting_line(['R([cm])']+list(graphite_grain.geometry.r_num[1:]/2+graphite_grain.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['T']+list(graphite_grain.temperature_field.temperature_num)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num[1:]/2+particle_list[0].geometry.r_num[0:-1]/2)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'T:{idx}']+list(particle.temperature_field.temperature_num)))
        if len(kernel_batch_list) > 0 and len(kernel_batch_list[0]) > 0:
            f.write('-----Failed Particle-----\n')
            f.write(formatting_line(['R([cm])']+list(kernel_batch_list[0][0].geometry.r_num[1:]/2+kernel_batch_list[0][0].geometry.r_num[0:-1]/2)))
        for idx,kernel_batch in enumerate(kernel_batch_list):
            for batch,kernel in enumerate(kernel_batch):
                f.write(formatting_line([f'T:{idx}B:{batch}']+list(kernel.temperature_field.temperature_num)))
        f.close()
    
    def _record_diffusion_coef_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录扩散系数分布
        '''
        txt_path = os.path.join(self.diffusion_coef_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('DiffusionCoef Distribution([cm2]/[s])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num[1:]/2+element.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['D']+list(element.material.diffusion_coef_element_num)))
        f.write('-----Graphite Grain-----\n')
        f.write(formatting_line(['R([cm])']+list(graphite_grain.geometry.r_num[1:]/2+graphite_grain.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['D']+list(graphite_grain.material.diffusion_coef_graphite_grain_num)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num[1:]/2+particle_list[0].geometry.r_num[0:-1]/2)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'D(T:{idx})']+list(particle.material.diffusion_coef_particle_num)))
        f.write('-----Failed Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(kernel_batch_list[0][0].geometry.r_num[1:]/2+kernel_batch_list[0][0].geometry.r_num[0:-1]/2)))
        for idx,kernel_batch in enumerate(kernel_batch_list):
            for batch,kernel in enumerate(kernel_batch):
                f.write(formatting_line([f'D(T:{idx}B:{batch})']+list(kernel.material.diffusion_coef_kernel_num)))
        f.close()
    
    def _record_heat_diffusion_coef_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录热扩散系数分布
        '''
        txt_path = os.path.join(self.heat_diffusion_coef_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('HeatDiffusionCoef Distribution([cm2]/[s])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num[1:]/2+element.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['alpha']+list(element.material.heat_diffusion_coef_element_num)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num[1:]/2+particle_list[0].geometry.r_num[0:-1]/2)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'alpha(T:{idx})']+list(particle.material.heat_diffusion_coef_particle_num)))
        f.close()
    
    def _record_thermal_conductivity_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录导热系数分布
        '''
        txt_path = os.path.join(self.thermal_conductivity_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('ThermalConductivity Distribution([W]/[m][s])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num[1:]/2+element.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['k']+list(element.material.thermal_conductivity_element_num)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num[1:]/2+particle_list[0].geometry.r_num[0:-1]/2)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'k(T:{idx})']+list(particle.material.thermal_conductivity_particle_num)))
        f.close()

    def _record_fp_generation_distribution(self,element:Element,graphite_grain:GraphiteGrain,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            记录产生率分布
        '''
        txt_path = os.path.join(self.fp_generation_folder,f'{self.time_tot:.6e}.txt')
        f = open(txt_path,'a',buffering=1)
        f.write('FP Generation Distribution([Bq]/[cm3][s])\n')
        f.write('-----Element-----\n')
        f.write(formatting_line(['R([cm])']+list(element.geometry.r_num[1:]/2+element.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['Q']+list(element.diffusion_field.generation_rate_num)))
        f.write('-----Graphite Grain-----\n')
        f.write(formatting_line(['R([cm])']+list(graphite_grain.geometry.r_num[1:]/2+graphite_grain.geometry.r_num[0:-1]/2)))
        f.write(formatting_line(['Q']+list(graphite_grain.diffusion_field.generation_rate_num)))
        f.write('-----Intact Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(particle_list[0].geometry.r_num[1:]/2+particle_list[0].geometry.r_num[0:-1]/2)))
        for idx,particle in enumerate(particle_list):
            f.write(formatting_line([f'Q(T:{idx})']+list(particle.diffusion_field.generation_rate_num)))
        f.write('-----Failed Particle-----\n')
        f.write(formatting_line(['R([cm])']+list(kernel_batch_list[0][0].geometry.r_num[1:]/2+kernel_batch_list[0][0].geometry.r_num[0:-1]/2)))
        for idx,kernel_batch in enumerate(kernel_batch_list):
            for batch,kernel in enumerate(kernel_batch):
                f.write(formatting_line([f'Q(T:{idx}B:{batch})']+list(kernel.diffusion_field.generation_rate_num)))
        f.close()

    def _record_release_rate(self,sphere_list,particle_list,kernel_batch_list):
        '''
            输出释放率
        '''
        rupture_release_rate_list = self._get_sphere_property('rupture_release_rate',particle_list,particle_list,kernel_batch_list)
        recoil_release_rate_list = self._get_sphere_property('recoil_release_rate',sphere_list,particle_list,kernel_batch_list)
        diffusion_release_rate_list = self._get_sphere_property('diffusion_release_rate',sphere_list,particle_list,kernel_batch_list)
        release_rate_list = self._get_sphere_property('release_rate',sphere_list,particle_list,kernel_batch_list)
        self.f_rupture_release_rate.write(formatting_line([self.time_tot] + rupture_release_rate_list))
        self.f_recoil_release_rate.write(formatting_line([self.time_tot] + recoil_release_rate_list))
        self.f_diffusion_release_rate.write(formatting_line([self.time_tot] + diffusion_release_rate_list))
        self.f_release_rate.write(formatting_line([self.time_tot] + release_rate_list))

    def _record_release_cumulant(self,sphere_list,particle_list,kernel_batch_list):
        '''
            输出释放量
        '''
        rupture_release_cumulant_list = self._get_sphere_property('rupture_release_cumulant',particle_list,particle_list,kernel_batch_list)
        recoil_release_cumulant_list = self._get_sphere_property('recoil_release_cumulant',sphere_list,particle_list,kernel_batch_list)
        diffusion_release_cumulant_list = self._get_sphere_property('diffusion_release_cumulant',sphere_list,particle_list,kernel_batch_list)
        release_cumulant_list = self._get_sphere_property('release_cumulant',sphere_list,particle_list,kernel_batch_list)
        self.f_rupture_release_cumulant.write(formatting_line([self.time_tot] + rupture_release_cumulant_list))
        self.f_recoil_release_cumulant.write(formatting_line([self.time_tot] + recoil_release_cumulant_list))
        self.f_diffusion_release_cumulant.write(formatting_line([self.time_tot] + diffusion_release_cumulant_list))
        self.f_release_cumulant.write(formatting_line([self.time_tot] + release_cumulant_list))

    def _record_inventory(self,sphere_list,particle_list:List[Particle],kernel_batch_list:List[List[Kernel]],element:Element,graphite_grain:GraphiteGrain):
        '''
            输出盘存量
        '''
        inventory_list = self._get_sphere_property('inventory',sphere_list,particle_list,kernel_batch_list)
        inventory_material_layer_list = self._get_sphere_property('inventory_material_layer',sphere_list,particle_list,kernel_batch_list)
        total_inventory_real = element.diffusion_field.inventory + graphite_grain.diffusion_field.inventory
        for particle in particle_list:
            total_inventory_real += particle.diffusion_field.inventory*particle.particle_number
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                total_inventory_real += kernel.diffusion_field.inventory*kernel.particle_number
        self.f_inventory.write(formatting_line([self.time_tot] + inventory_list+[element.total_inventory]+[total_inventory_real]))
        self.f_inventory_material_layer.write(formatting_line([self.time_tot] + inventory_material_layer_list))

    def _record_temperature(self,sphere_list,particle_list,kernel_batch_list,element):
        '''
            输出温度
        '''
        temperature_list = self._get_sphere_property('temperature_list',sphere_list,particle_list,kernel_batch_list)
        self.f_temperature.write(formatting_line([self.time_tot] + temperature_list + [element.element_power]))
    
    def _record_diffusion_coef(self,sphere_list,particle_list,kernel_batch_list):
        '''
            输出扩散系数
        '''
        diffusion_coef_list = self._get_sphere_property('diffusion_coef_list',sphere_list,particle_list,kernel_batch_list)
        self.f_diffusion_coef.write(formatting_line([self.time_tot] + diffusion_coef_list))
    
    def _record_performance(self,sphere_list,particle_list,kernel_batch_list):
        '''
            输出燃料性能
        '''
        fuel_performance_list,self.fuel_performance_name_list = self._get_sphere_property('fuel_performance',sphere_list,particle_list,kernel_batch_list)
        self.f_performance.write(formatting_line([self.time_tot] + fuel_performance_list))
    
    def _record_fp_generation(self,element:Element,kernel,particle:Particle,graphite_grain:GraphiteGrain):
        '''
            输出裂变产物产生率
        '''
        kernel_part = [kernel.diffusion_field.generation_rate] if kernel is not None else []
        self.f_fp_generation.write(formatting_line([self.time_tot] + [element.element_generation_rate] + [element.diffusion_field.generation_rate,element.diffusion_field.generation_rate_from_fission,element.diffusion_field.generation_rate_from_particle,element.diffusion_field.generation_rate_from_graphite_grain]+[particle.diffusion_field.generation_rate]+kernel_part+[graphite_grain.diffusion_field.generation_rate]))

    def _get_sphere_property(self,property_name,sphere_list:List[Sphere],particle_list:List[Particle],kernel_batch_list:List[List[Kernel]]):
        '''
            得到属性列表
        '''
        if ('release' in property_name) or ('inventory' == property_name):
            property_list = [getattr(sphere.diffusion_field,property_name) for sphere in sphere_list]
            return property_list
        elif property_name in ['temperature_list','inventory_material_layer','diffusion_coef_list']:
            property_list = []
            for sphere in sphere_list:
                if 'temperature_list' == property_name:
                    property_list += list(sphere.temperature_field.temperature_list) 
                elif 'diffusion_coef_list' == property_name:
                    property_list += list(sphere.material.diffusion_coef_list)
                elif 'inventory_material_layer' == property_name:
                    property_list += list(sphere.diffusion_field.inventory_material_layer)
            return property_list
        # 特殊属性
        elif property_name == 'fuel_performance':
            property_list = []
            property_name_list = []
            for idx,particle in enumerate(particle_list):
                particle_number = particle.particle_number
                failed_particle_number = sum([kernel.particle_number for kernel in kernel_batch_list[idx]])
                failure_fraction_simulation = failed_particle_number / particle.particle_number
                if particle.fuel_performance.failure_model == 'WeibullFailure':
                    pressure = particle.fuel_performance.pressure_new
                    sic_thickness = particle.fuel_performance.sic_thickness_new
                    sic_weibull = particle.fuel_performance.sic_weibull
                    sic_tensile = particle.fuel_performance.sic_tensile_strength
                    sic_stress = particle.fuel_performance.stress_t_sic_new
                    failure_fraction_pressure = particle.fuel_performance.sic_pressure_failure_fraction
                    failure_fraction_thermal = particle.fuel_performance.sic_thermal_decomposition_failure_fraction
                    property_list += [pressure,sic_thickness,sic_weibull,sic_tensile,sic_stress,failure_fraction_pressure,failure_fraction_thermal,failure_fraction_simulation,particle_number,failed_particle_number]
                    property_name_list += [f'P(MPa)(T:{idx})',f'SiC-d(m)(T:{idx})',f'Weibull(MPa)(T:{idx})',f'Strength(MPa)(T:{idx})',f'Stress(MPa)(T:{idx})',f'P-Failure(T:{idx})',f'T-Failure(T:{idx})',f'Failure(T:{idx})',f'Intact-N(T:{idx})',f'Failed-N(T:{idx})']
                else:
                    property_name_list += [f'Failure(T:{idx})',f'Intact-N(T:{idx})',f'Failed-N(T:{idx})']
                    property_list += [failure_fraction_simulation,particle_number,failed_particle_number]
            return property_list,property_name_list
        
