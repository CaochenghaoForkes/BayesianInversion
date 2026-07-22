from MARS.mars_xml_reader import MarsXMLReader
from MARS.mars_spherical import Particle,Element,GraphiteGrain,Kernel,TemperatureField,ParticleTemperatureField,DiffusionField
from MARS.mars_control import MarsControl
from TOOL.cosmos_general_function import extract_txt
import os
import numpy as np
def mars(mars_xml_path,tracer=None,target_nuclide=None,core_control=None):

    # 从xml读取信息
    reader = MarsXMLReader(mars_xml_path)
    
    # 读取信息
    reader.read_configuration(tracer,target_nuclide,core_control)

    # 如果是独立计算或联合Core计算,则直接填好字典进行一次扩散计算
    if reader.control_dict['simulation_mode'] == 'independent' or 'tracer' in reader.control_dict['simulation_mode']:
        diffusion_calculation(reader)
    # 如果是安分报告计算模式,则根据输入进行批量计算
    elif reader.control_dict['simulation_mode'] == 'fsar':
        for nuclide in reader.fsar_mode_dict['nuclide_list']:
            core_release_rate = 0.0
            inventory_total = reader.fsar_mode_dict['inventory_dict'][nuclide]
            decay_constant = reader.fsar_mode_dict['decay_constant_dict'][nuclide]
            diffusion_coef = reader.fsar_mode_dict['diffusion_coef_dict'][nuclide]
            reader.external_conditions_dict['nuclide'] = nuclide
            reader.external_conditions_dict['decay_constant'] = decay_constant
            D_graphite_matrix,D_graphite_grain,D_kernel,D_buffer,D_pyc,D_sic,A_graphite_matrix,A_graphite_grain,A_kernel,A_buffer,A_pyc,A_sic = diffusion_coef
            reader.external_conditions_dict['diffusion_coef']['D_element'] = [D_graphite_matrix,D_graphite_matrix]
            reader.external_conditions_dict['diffusion_coef']['D_graphite_grain'] = [D_graphite_grain]
            reader.external_conditions_dict['diffusion_coef']['D_particle'] = [D_kernel,D_buffer,D_pyc,D_sic,D_pyc]
            reader.external_conditions_dict['diffusion_coef']['D_buffer'] = [D_buffer]
            reader.external_conditions_dict['diffusion_coef']['D_pyc'] = [D_pyc]
            reader.external_conditions_dict['diffusion_coef']['D_sic'] = [D_sic]
            reader.external_conditions_dict['diffusion_coef']['D_kernel'] = [D_kernel]
            reader.external_conditions_dict['diffusion_coef']['A_element'] = [A_graphite_matrix,A_graphite_matrix]
            reader.external_conditions_dict['diffusion_coef']['A_graphite_grain'] = [A_graphite_grain]
            reader.external_conditions_dict['diffusion_coef']['A_particle'] = [A_kernel,A_buffer,A_pyc,A_sic,A_pyc]
            reader.external_conditions_dict['diffusion_coef']['A_buffer'] = [A_buffer]
            reader.external_conditions_dict['diffusion_coef']['A_pyc'] = [A_pyc]
            reader.external_conditions_dict['diffusion_coef']['A_sic'] = [A_sic]
            reader.external_conditions_dict['diffusion_coef']['A_kernel'] = [A_kernel]
            for share,temperature in zip(reader.fsar_mode_dict['temperature_share_list'],reader.fsar_mode_dict['temperature_list']):
                inventory = inventory_total * share
                reader.external_conditions_dict['temperature'] = ([temperature,temperature],[0.0,1e30])
                reader.external_conditions_dict['inventory'] = ([0.0,inventory],[0.0,reader.fsar_mode_dict['time']])
                reader.control_dict['output_path'] = os.path.join(reader.fsar_mode_dict['output_path'],f'{nuclide}_{temperature:.1f}K')
                # 扩散计算
                diffusion_calculation(reader)
                core_release_rate += extract_txt(os.path.join(reader.control_dict['output_path'],'ReleaseRate.txt'))['Element'][-1]
            with open(os.path.join(reader.fsar_mode_dict['output_path'],f'CoreReleaseRate.txt'),'a') as f:
                f.write(f'{nuclide}: {core_release_rate:.4e} [Bq]/[s]\n')
        
def diffusion_calculation(reader):

    # 初始化
    Particle.init_shared_properties(reader)
    TemperatureField.set_shared_properties(reader)
    ParticleTemperatureField.set_particle_shared_properties(reader)
    DiffusionField.set_shared_properties(reader)
    
    # 生成控制对象
    control = MarsControl(reader)

    # 生成颗粒对象
    particle_list = [Particle(reader, temperature_zone_idx) for temperature_zone_idx in range(reader.models_dict['intra_pebble_temperature']['temperature_zone_number']if reader.models_dict['intra_pebble_temperature']['temperature_zone_number'] != 0 else 1)]
    # 破损颗粒对象列表
    kernel_batch_list = [[] for _ in range(reader.models_dict['intra_pebble_temperature']['temperature_zone_number'] if reader.models_dict['intra_pebble_temperature']['temperature_zone_number'] != 0 else 1)]
    # 生成燃料元件对象
    element = Element(reader,particle_list)
    # 生成石墨晶粒对象
    graphite_grain = GraphiteGrain(reader)
    import matplotlib.pyplot as plt
    plt.ion()
    # 开启时间循环
    while not control.finish_swtich:

        # 更新外部强制条件
        element.update_time_dependent_properties(control.time_tot)

        # 更新类共有属性
        Particle.FuelPerformance.set_shared_properties(control.time_tot,control.time_step,element.burnup,element.neutron_flux)

        # 更新温度场
        TemperatureField.update_temperature_field(control.time_step,control.time_tot,reader,particle_list,element,graphite_grain,kernel_batch_list)
        element.material.update_element_material_properties(['binary_diffusion_coef'],temperature_num=element.temperature_field.temperature_num)
        # 更新扩散系数
        for idx,particle in enumerate(particle_list):
            particle.material.update_particle_material_properties(['diffusion_coef'],temperature_num=particle.temperature_field.temperature_num,time=control.time_tot)
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.material.update_material_properties(['diffusion_coef'],temperature_num=kernel.temperature_field.temperature_num,time=control.time_tot)
        graphite_grain.material.update_material_properties(['diffusion_coef'],temperature_num=graphite_grain.temperature_field.temperature_num,time=control.time_tot)
        element.material.update_element_material_properties(['diffusion_coef'],temperature_num=element.temperature_field.temperature_num,time=control.time_tot)
        
        # 更新完整颗粒的破损率/产生新破损颗粒
        for idx,particle in enumerate(particle_list):
            particle.fuel_performance.update_failure_rate(particle.temperature_field.temperature_list,particle.material)
            if particle.fuel_performance.failure_fraction_increment > 0.0:
                kernel_batch_list[idx].append(Kernel(reader,particle,particle.particle_number*particle.fuel_performance.failure_fraction_increment))

        # 添加完整颗粒的破裂释放率
        for particle in particle_list:
            particle.fuel_performance.update_rupture_release_rate(particle.diffusion_field,particle.particle_number,control.time_step)
        
        # 更新完整颗粒/破损核芯/石墨晶粒的核素产生率分布
        for particle in particle_list:
            particle.diffusion_field.update_generation_rate(element.element_generation_rate)
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.diffusion_field.update_generation_rate(element.element_generation_rate)
        graphite_grain.diffusion_field.update_generation_rate(element.element_generation_rate)
        
        # 完整颗粒/破损核芯/石墨晶粒进行扩散计算
        for particle in particle_list:
            particle.diffusion_field.update_concentration_field(control.time_step,particle.material.diffusion_coef_particle_num)
        for kernel_batch in kernel_batch_list:
            for kernel in kernel_batch:
                kernel.diffusion_field.update_concentration_field(control.time_step,kernel.material.diffusion_coef_kernel_num)
        graphite_grain.diffusion_field.update_concentration_field(control.time_step,graphite_grain.material.diffusion_coef_graphite_grain_num)
        
        # 更新石墨基体的产生率
        element.diffusion_field.update_generation_rate(element.element_generation_rate,element.temperature_subzone_idx,element.temperature_subzone_volume,particle_list,kernel_batch_list,graphite_grain)
        # 石墨基体进行吸附计算
        element.material.update_element_material_properties(['adsorp_coef'],temperature_num=element.temperature_field.temperature_num,c_fvm=element.diffusion_field.concentration_num_fvm)
        # 边界层质量传递系数
        if element.diffusion_field.mass_transfer_mode == 'Theory':
            element.material.update_mass_transfer_coef(temperature_num=element.temperature_field.temperature_num)
            element.diffusion_field.mass_transfer_coef = element.material.mass_transfer_coef
        # 石墨基体进行扩散计算
        element.diffusion_field.update_concentration_field(control.time_step,element.material.diffusion_coef_element_num,adsorp_coef=element.material.adsorb_coef)
    
        # 推进时间步
        control.procced(element,graphite_grain,particle_list,kernel_batch_list)
        

    
  