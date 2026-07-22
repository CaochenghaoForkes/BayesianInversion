from CORE.core_xml_reader import CoreXMLReader
from CORE.core_tracer import Core,Tracers
from CORE.core_to_nuit import Nuit
from CORE.core_to_mars import Mars
from CORE.core_control import CoreControl
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import numpy as np
def core(core_xml_path,n_job=20):
    
    # 读取对象
    reader = CoreXMLReader(core_xml_path)

    # 读取信息
    reader.read_configuration()
    
    # 控制对象
    control = CoreControl(reader)

    # 示踪球对象
    tracers = Tracers(reader)

    # 堆芯对象
    core = Core(reader,control)

    # 燃耗计算对象
    burnup = Nuit(reader,control)

    # 燃料源项计算对象
    diffusion = Mars(control)

    # 批次处理
    control.split_to_batch(tracers.tracer_list)

    for batch,tracer_batch in enumerate(control.tracer_batch_list):
      
        control.make_dir(batch)
    
        # 生成示踪球的轨迹
        for tracer in tracer_batch:
            core.generate_location_history(tracer)
        
        # 生成示踪球的运行历史
        for tracer in tracer_batch:
            core.generate_operation_history(tracer)
     
        # 生成示踪球的燃耗历史
        Parallel(n_jobs=n_job, backend="threading", prefer="threads")(delayed(burnup.burnup_calculation)(tracer)for tracer in tracer_batch)
        
        # 截取目标燃耗以内历史
        for tracer in tracer_batch:
            core.extract_effective_history(tracer)
   
        # 生成示踪球的释放率历史
        if control.diffusion_swtich:
            diffusion.diffusion_parallel(core_xml_path,tracer_batch,batch,n_job=n_job)
        
        # 输出
        control.time_loop(tracer_batch)

    # 合并批次
    control.merge_batch(tracer)
 