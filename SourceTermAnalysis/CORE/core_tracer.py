import numpy as np
import random
from itertools import accumulate
from typing import List
import bisect
from CORE.core_xml_reader import CoreXMLReader
from TOOL.cosmos_general_function import TimeSeries,round_sig

sig = 12

class Tracer:
    '''
        示踪球
    '''
    def __init__(self,tracer_idx,initial_channel,initial_layer,add_time,initial_material,channel_range,reader):
        # 基本属性
        self.idx = tracer_idx
        self.initial_material = initial_material
        self.accident_begin = reader.control_dict['accident_begin']
        # 初始化
        self.grid_stay_time = 0.0
        self.batch = 0
        self.layer_location = initial_layer
        self.channel_location = initial_channel
        self.channel_range = channel_range
        self.time_sequence = [round_sig(add_time)]
        self.time_step_history = []
        self.batch_history = []
        self.layer_location_history = []
        self.channel_location_history = []
        self.temperature_history = None
        self.neutron_flux_history = None
        self.power_history = None
        self.burnup_history = None
        self.inventory_history_dict = None
        self.discharge_inventory_history_dict = {}
        self.storage_inventory_history_dict = {}
        self.isotope_history_dict = None
        self.discharge_isotope_history_dict = {}
        self.storage_isotope_history_dict = {}
        self.decay_heat_history = None
        self.gamma_spectrum_history = {}
        self.discharge_gamma_spectrum_history = {}
        self.storage_gamma_spectrum_history = {}
        self.release_history_dict = {}
        self.is_alive = True
        self.dead_time = 1e30

class Tracers:
    '''
        示踪球列表
    '''
    def __init__(self,reader:CoreXMLReader):
        
        # 产生示踪球
        self.tracer_list = self._generate_tracers(reader)

    def _generate_tracers(self,reader:CoreXMLReader):
        '''
            产生示踪球
        '''
        channel_number = reader.core_properties_dict['channel_number']
        layer_number = reader.core_properties_dict['layer_number']
        tracer_info_list = reader.tracer_list
        tracer_idx = 0
        tracer_list = []
        for tracer_info in tracer_info_list:
            tracer_number = tracer_info['tracer_number']
            each_time_add_number = tracer_info['each_time_add_number']
            initial_channel = tracer_info['initial_channel']
            channel_range = tracer_info['channel_range']
            if type(channel_range) == str:
                if channel_range == 'all':
                    channel_range = list(range(channel_number))
    
            initial_layer = tracer_info['initial_layer']
            add_start = tracer_info['add_start']
            add_end = tracer_info['add_end']
            initial_material = tracer_info['material']

            # 添加的次数
            add_times = (tracer_number + each_time_add_number - 1) // each_time_add_number
            # 添加的时刻
            if add_times == 1:
                add_time_list = [add_start]
            else:
                step = (add_end - add_start) / (add_times - 1)
                add_time_list = [add_start + i * step for i in range(add_times)]
            add_time_idx = 0
            for batch_idx in range(tracer_number):
                # 该球被添加的时刻
                add_time = add_time_list[add_time_idx]
                # 该球初始位置
                this_tracer_initial_channel = random.randint(0,channel_number-1) if initial_channel == 'random' else int(initial_channel)
                this_tracer_initial_layer = random.randint(0,layer_number-1) if initial_layer == 'random' else int(initial_layer)
                # 创建示踪球对象
                tracer = Tracer(tracer_idx,this_tracer_initial_channel,this_tracer_initial_layer,add_time,initial_material,channel_range,reader)
                tracer_list.append(tracer)
                # 更新索引
                tracer_idx += 1
                if (batch_idx+1) % each_time_add_number == 0:
                    add_time_idx += 1
        
        return tracer_list

class Core:
    '''
        堆芯对象
    '''
    def __init__(self,reader:CoreXMLReader,control):
        
        # 基本属性
        self.reader = reader
        self.target_burnup = reader.core_properties_dict['target_burnup']
        self.channel_number = reader.core_properties_dict['channel_number']
        self.layer_number = reader.core_properties_dict['layer_number']
        self.discharge_mesh = reader.core_properties_dict['discharge_mesh']
        self.element_number = reader.core_properties_dict['element_number']
        self.maximum_batch = reader.core_properties_dict['maximum_batch']
        self.accident_begin = reader.control_dict['accident_begin']
        self.time_span_list = reader.control_dict['time']
        self.end_list = list(accumulate(self.time_span_list))
        self.time_step_list = reader.control_dict['time_step']
        self.finish_time = round(sum(self.time_span_list),10)
        self.control = control
       
        # Matrix[几何层][通道] 
        # 温度场
        if self.control.read_temperature_swtich:
            self.temperature_field_steady = reader.core_properties_dict['temperature_field_steady']
            temperature_field_transient_list = reader.core_properties_dict['temperature_field_transient_list']
            temperature_field_transient_time_list = reader.core_properties_dict['temperature_field_transient_time_list']
            temperature_field_transient_list = [self.temperature_field_steady,self.temperature_field_steady] + list(temperature_field_transient_list)
            steady_end = temperature_field_transient_time_list[0] if len(temperature_field_transient_time_list) > 0 else 1e30
            temperature_field_transient_time_list = [0.0,steady_end] + list(temperature_field_transient_time_list)
            self.temperature_field_transient_table = TimeSeries(temperature_field_transient_time_list,temperature_field_transient_list) if self.accident_begin < self.finish_time else None

        # 中子注量率场
        if self.control.read_neutron_flux_swtich:
            self.neutron_flux_field_steady = reader.core_properties_dict['neutron_flux_field_steady']
            neutron_flux_field_transient_list = reader.core_properties_dict['neutron_flux_field_transient_list']
            neutron_flux_field_transient_time_list = reader.core_properties_dict['neutron_flux_field_transient_time_list']
            steady_end = neutron_flux_field_transient_time_list[0] if len(neutron_flux_field_transient_time_list) > 0 else 1e30
            neutron_flux_field_transient_time_list = [0.0,steady_end] + list(neutron_flux_field_transient_time_list)
            self.neutron_flux_field_transient_table_batch = []
            for batch in range(self.maximum_batch+1):
                neutron_flux_field_transient_batch = [neutron_flux_field_transient_list[i][batch] for i in range(len(neutron_flux_field_transient_time_list)-2)]
                neutron_flux_field_transient_batch = [self.neutron_flux_field_steady[batch],self.neutron_flux_field_steady[batch]] + list(neutron_flux_field_transient_batch)
                self.neutron_flux_field_transient_table_batch.append(TimeSeries(neutron_flux_field_transient_time_list,neutron_flux_field_transient_batch)) if self.accident_begin < self.finish_time else None
        # 功率场
        if self.control.read_power_swtich:
            self.power_field_steady = reader.core_properties_dict['power_field_steady']
            power_field_transient_list = reader.core_properties_dict['power_field_transient_list']
            power_field_transient_time_list = reader.core_properties_dict['power_field_transient_time_list']
            steady_end = power_field_transient_time_list[0] if len(power_field_transient_time_list) > 0 else 1e30
            power_field_transient_time_list = [0.0,steady_end] + list(power_field_transient_time_list)
            self.power_field_transient_table_batch = []
            for batch in range(self.maximum_batch):
                power_field_transient_batch = [power_field_transient_list[i][batch] for i in range(len(power_field_transient_time_list)-2)]
                power_field_transient_batch = [self.power_field_steady[batch],self.power_field_steady[batch]] + list(power_field_transient_batch)
                self.power_field_transient_table_batch.append(TimeSeries(power_field_transient_time_list,power_field_transient_batch)) if self.accident_begin < self.finish_time else None
        # 滞留时间场
        self.grid_stay_time_field_steady = reader.core_properties_dict['grid_stay_time_field_steady']
        grid_stay_time_field_transient_list = reader.core_properties_dict['grid_stay_time_field_transient_list']
        grid_stay_time_field_transient_list = [self.grid_stay_time_field_steady,self.grid_stay_time_field_steady] + list(grid_stay_time_field_transient_list)
        grid_stay_time_field_transient_time_list = reader.core_properties_dict['grid_stay_time_field_transient_time_list']
        steady_end = grid_stay_time_field_transient_time_list[0] if len(grid_stay_time_field_transient_time_list) > 0 else 1e30
        grid_stay_time_field_transient_time_list = [0.0,steady_end] + list(grid_stay_time_field_transient_time_list)
        self.grid_stay_time_field_transient_table = TimeSeries(grid_stay_time_field_transient_time_list,grid_stay_time_field_transient_list) if self.accident_begin < self.finish_time else None

    def generate_location_history(self,tracer:Tracer):
        '''
            产生位置历史
        '''
        # 初始时刻
        time = tracer.time_sequence[0]
        while round(time,10) <= self.finish_time and tracer.is_alive:
            # 当前位置
            layer_idx_old = tracer.layer_location
            channel_idx_old = tracer.channel_location
            # 网格滞留时间 
            grid_stay_time_field = self.grid_stay_time_field_steady if time < self.accident_begin else self.grid_stay_time_field_transient_table.at(time,mode='step')
            grid_stay_time = grid_stay_time_field[layer_idx_old][channel_idx_old]
            # 处于第几个阶段
            phase = bisect.bisect_right(self.end_list,time)
            if phase >= len(self.time_span_list):
                phase = len(self.time_span_list) - 1
            
            # 获得时间步
            if round(time,10) >= round(self.accident_begin,10):
                time_step = self.time_step_list[phase]
            else:
                time_step = (self.accident_begin - time) if time + grid_stay_time > self.accident_begin else grid_stay_time
            # 推进时间
            time += time_step
            # 记录时间
            tracer.time_sequence.append(time)
            # 记录时间步
            tracer.time_step_history.append(time_step)
            # 记录这个时间步中球的批次
            tracer.batch_history.append(tracer.batch)
            # 记录这个时间步中球的位置
            tracer.layer_location_history.append(layer_idx_old)
            tracer.channel_location_history.append(channel_idx_old)
            # 推进滞留时间
            tracer.grid_stay_time += time_step

            # 判断位置
            # 超出旧网格的滞留时间
            if round(tracer.grid_stay_time,8) >= round(grid_stay_time,8):
                # 更新滞留时间
                tracer.grid_stay_time -= grid_stay_time
                # 如果不是最底的几何层则向下流动
                if layer_idx_old < self.layer_number + self.discharge_mesh - 1:
                    layer_idx_new = layer_idx_old + 1
                    channel_idx_new = channel_idx_old
                # 最底几何层则判断是否达到最后批次
                elif layer_idx_old == self.layer_number + self.discharge_mesh - 1:
                    # 如果不是最大批次则返回堆芯顶部流道
                    if tracer.batch < self.maximum_batch:
                        tracer.batch += 1
                        layer_idx_new = 0
                        channel_idx_new = random.choice(tracer.channel_range)       
                    # 如果是最大批次则卸出堆芯
                    elif tracer.batch == self.maximum_batch:
                        tracer.is_alive = False
                        tracer.dead_time = time
            else:
                layer_idx_new = layer_idx_old
                channel_idx_new = channel_idx_old
            # 更新新位置
            tracer.layer_location = layer_idx_new
            tracer.channel_location = channel_idx_new

    def generate_operation_history(self,tracer:Tracer):
        '''
            产生运行历史
        '''
        time_mid = np.array(tracer.time_sequence[1:])/2+np.array(tracer.time_sequence[0:-1])/2
        if self.control.read_temperature_swtich:
            tracer.temperature_history = np.zeros_like(tracer.time_step_history)
        if self.control.read_neutron_flux_swtich:
            tracer.neutron_flux_history = np.zeros_like(tracer.time_step_history)
        if self.control.read_power_swtich:
            tracer.power_history = np.zeros_like(tracer.time_step_history)

        for step,time in enumerate(time_mid):
            layer_idx = tracer.layer_location_history[step]
            channel_idx = tracer.channel_location_history[step]
            batch = tracer.batch_history[step]
            if time < self.accident_begin:
                if self.control.read_temperature_swtich:
                    temperature = self.temperature_field_steady[layer_idx][channel_idx]
                if self.control.read_neutron_flux_swtich:
                    neutron_flux = self.neutron_flux_field_steady[batch][layer_idx][channel_idx]
                if self.control.read_power_swtich:
                    power = self.power_field_steady[batch][layer_idx][channel_idx]
            else:
                if self.control.read_temperature_swtich:
                    temperature = self.temperature_field_transient_table.at(time)[layer_idx][channel_idx]
                if self.control.read_neutron_flux_swtich:
                    neutron_flux = self.neutron_flux_field_transient_table_batch[batch].at(time)[layer_idx][channel_idx]
                if self.control.read_power_swtich:
                    power = self.power_field_transient_table_batch[batch].at(time)[layer_idx][channel_idx]
       
            if self.control.read_temperature_swtich:
                tracer.temperature_history[step] = temperature
            if self.control.read_neutron_flux_swtich:
                tracer.neutron_flux_history[step] = neutron_flux
            if self.control.read_power_swtich:
                tracer.power_history[step] = power

    def extract_effective_history(self,tracer:Tracer):
        '''
            截取未超过目标燃耗的部分:注意最后一个燃耗点是大于等于目标燃耗的
        '''
        if tracer.burnup_history[-1] <= self.target_burnup:
            return
        for batch in range(self.maximum_batch):
            burnup_idx = (batch+1)*(self.layer_number+self.discharge_mesh)-self.discharge_mesh
            storage_idx = burnup_idx + self.discharge_mesh
            if burnup_idx >= len(tracer.time_sequence):
                break
            examined_burnup = tracer.burnup_history[burnup_idx]
            if examined_burnup >= self.target_burnup:
                final_idx = min(storage_idx,len(tracer.time_sequence)-1)
                tracer.time_sequence = tracer.time_sequence[0:final_idx+1]
                tracer.burnup_history = tracer.burnup_history[0:final_idx+1]
                tracer.time_step_history = tracer.time_step_history[0:final_idx]
                tracer.batch_history = tracer.batch_history[0:final_idx]
                tracer.layer_location_history = tracer.layer_location_history[0:final_idx]
                tracer.channel_location_history = tracer.channel_location_history[0:final_idx]
                tracer.neutron_flux_history = tracer.neutron_flux_history[0:final_idx]
                tracer.power_history = tracer.power_history[0:final_idx]
                for nuclide,inventory_history in tracer.inventory_history_dict.items():
                    tracer.inventory_history_dict[nuclide] = inventory_history[0:final_idx+1]
                    tracer.discharge_inventory_history_dict[nuclide] = tracer.discharge_inventory_history_dict[nuclide][0:batch+1]
                    tracer.storage_inventory_history_dict[nuclide] = tracer.storage_inventory_history_dict[nuclide][0:batch+1]
                for nuclide,isotope_history in tracer.isotope_history_dict.items():
                    tracer.isotope_history_dict[nuclide] = isotope_history[0:final_idx+1]
                    tracer.discharge_isotope_history_dict[nuclide] = tracer.discharge_isotope_history_dict[nuclide][0:batch+1]
                    tracer.storage_isotope_history_dict[nuclide] = tracer.storage_isotope_history_dict[nuclide][0:batch+1]
                if self.control.decay_heat_swtich:
                    tracer.decay_heat_history = tracer.decay_heat_history[0:final_idx]
                if self.control.read_temperature_swtich:
                    tracer.temperature_history = tracer.temperature_history[0:final_idx]
                if self.control.gamma_spectrum_swtich:
                    for energy,gamma_history in tracer.gamma_spectrum_history.items():
                        tracer.gamma_spectrum_history[energy] = gamma_history[0:final_idx]
                        tracer.discharge_gamma_spectrum_history[energy] = tracer.discharge_gamma_spectrum_history[energy][0:batch+1]
                        tracer.storage_gamma_spectrum_history[energy] = tracer.storage_gamma_spectrum_history[energy][0:batch+1]
                break



        

    
