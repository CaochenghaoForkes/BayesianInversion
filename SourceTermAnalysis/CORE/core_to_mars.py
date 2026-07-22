from CORE.core_xml_reader import CoreXMLReader
from CORE.core_tracer import Tracer
from CORE.core_control import CoreControl
from MARS.mars_main import mars
from joblib import Parallel, delayed
from typing import List
import re
import numpy as np
from collections import defaultdict
import os
import shutil

class Mars:
    '''
        调用燃料源项模块
    '''
    def __init__(self,control:CoreControl):
        
        self.control = control
        
    def _diffusion_calculation(self,tracer:Tracer) -> Tracer:
        '''
            调用Mars计算燃料源项
        '''
        for target_nuclide in self.control.output_nuclide:
            mars(self.control.mars_input_path,tracer,target_nuclide,self.control)
        return tracer

    @staticmethod
    def _worker(core_xml_path,batch,tracer:Tracer) -> Tracer:
        '''
            loky 子进程 worker:子进程里重建 reader 和 Mars:避免 pickle reader/self。
        '''
        reader = CoreXMLReader(core_xml_path)
        reader.read_configuration()
        control = CoreControl(reader,main_control=False)
        control.make_dir(batch)
        diffusion = Mars(control)
        return diffusion._diffusion_calculation(tracer)

    def diffusion_parallel(self,core_xml_path,tracer_list:List[Tracer],batch,n_job=8):
        '''
            多进程并行：返回更新后的 tracers
        '''
        # 并行计算
        results = Parallel(n_jobs=n_job, backend="loky", prefer="processes")(
            delayed(Mars._worker)(core_xml_path,batch, tracer)
            for tracer in tracer_list
        )
        
        # 提取释放率目录
        release_dict,dir_dict = self._extract_mars_output(self.control.mars_intermediate_dir)
        for nuclide in self.control.output_nuclide:
            for tracer in tracer_list:
                time_mid = np.array(tracer.time_sequence[1:])/2 + np.array(tracer.time_sequence[0:-1])/2
                born_time = tracer.time_sequence[0]
                path = release_dict[tracer.idx][nuclide]
                data = self.extract_txt(path)
                time = data['Time(s)'] + born_time
                release_rate = data['Element']
                release_history = np.interp(time_mid,time,release_rate)
                tracer.release_history_dict[nuclide] = release_history
        
                # 删除中间文件
                if self.control.delete_mars_files:
                    os.remove(os.path.join(self.control.mars_intermediate_dir,f'Tracer{tracer.idx}_{nuclide}.xml'))
                    shutil.rmtree(dir_dict[tracer.idx][nuclide])
    
    def _extract_mars_output(self,parent_dir):
        '''
            提取Mars的输出
        '''
        def add_file(tracer_map,filename):
            return {
                idx: {nuclide: os.path.join(dir_path, filename) for nuclide, dir_path in nuc_dict.items()}
                for idx, nuc_dict in tracer_map.items()
            }
            
        pattern = re.compile(r"^Tracer(\d+)_([A-Za-z]+[0-9]+(?:m[0-9]*)?)$")
        d = defaultdict(dict)

        for name in os.listdir(parent_dir):
            full_path = os.path.join(parent_dir, name)
            if not os.path.isdir(full_path):
                continue
            m = pattern.match(name)
            if not m:
                continue
            tracer_idx = int(m.group(1))
            nuclide = m.group(2)
            d[tracer_idx][nuclide] = full_path
        dir_dict = {k: v for k, v in d.items()}
        
        release_dict = add_file(dir_dict,'ReleaseRate.txt')
        
        return release_dict,dir_dict
    
    def extract_txt(self,file_path):
        '''
            提取txt文件
        '''
        with open(file_path, "r", encoding="utf-8") as f:
            title_line = f.readline()      
            header_line = f.readline()     
            headers = header_line.split()  

        data = np.loadtxt(file_path, skiprows=2) 
        result = {name: data[:, i] for i, name in enumerate(headers)}
        return result
        