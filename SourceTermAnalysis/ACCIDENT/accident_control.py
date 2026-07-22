import os
import shutil
from ACCIDENT.accident_xml_reader import AccidentXMLReader
from ACCIDENT.accident_matrix import AccidentMatrix
from TOOL.cosmos_general_function import formatting_line,write_header,round_sig

class AccidentControl:
    '''
        控制程序的时间步进,输出
    '''
    def __init__(self,reader:AccidentXMLReader):
        # 基本信息
        self.output_dir = reader.control_dict['output_path']
        self.write_interval_list = reader.control_dict['write_interval']
        self.integral_swtich = reader.control_dict['integral_swtich']
        self.cv_labels = reader.cv_labels
        self.nuclides = reader.nuclides
        self.n_nuclide = reader.n_nuclide
        self.n_cv = reader.n_cv
        self.simulation_time_list = reader.simulation_time_list
        self.simulation_time_step_list = reader.simulation_time_step_list
        self.finish_time = self.simulation_time_list[-1]
        # 初始化
        self.write_interval = round_sig(self.write_interval_list[0]) if self.write_interval_list[0] > 0 else int(self.write_interval_list[0])
        self.finish_swtich = False
        self.write_time = self.write_interval
        self.step_tot = 0
        self.time_tot = self.simulation_time_list[self.step_tot] # 表示该时间步的左时间点
        self.time_step = self.simulation_time_step_list[self.step_tot] # 表示该时间步的时间长度
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
        self.inventory_path_list = []
        self.inventory_f_list = []
        for label in self.cv_labels:
            path = os.path.join(self.output_dir,f'{label}.txt')
            self.inventory_path_list.append(path)
            f = open(path,'a',buffering=1)
            self.inventory_f_list.append(f)

    def _close_and_write_header(self,accident_matrix):
        '''
            写列头并关闭IO
        '''
        cv_idx = 0
        for cv_label,path,f in zip(self.cv_labels,self.inventory_path_list,self.inventory_f_list):
            f.close()
            if self.integral_swtich == 'off':
                write_header(path,cv_label+': nuclides inventory([atom])\n',formatting_line(['Time(s)']+self.nuclides))
            elif self.integral_swtich == 'on':
                integral_list = self._extract_integral(accident_matrix,cv_idx,header=True)
                write_header(path,cv_label+': nuclides inventory([atom])\n',formatting_line(['Time(s)']+self.nuclides+integral_list))
            cv_idx += 1
    def _record(self,accident_matrix:AccidentMatrix):
        '''
            记录数据
        '''
        # 记录时间
        self.output_time_list.append(self.time_tot)
        # 记录各控制体的核素盘存量
        for idx,cv_label in enumerate(self.cv_labels):
            nucdlie_inventory_list = list(accident_matrix.inventory_list[idx*self.n_nuclide:(idx+1)*self.n_nuclide])
            if self.integral_swtich == 'off':
                self.inventory_f_list[idx].write(formatting_line([self.time_tot]+nucdlie_inventory_list))
            elif self.integral_swtich == 'on':
                integral_list = self._extract_integral(accident_matrix,idx)
                self.inventory_f_list[idx].write(formatting_line([self.time_tot]+nucdlie_inventory_list+integral_list))

    def _extract_integral(self,accident_matrix:AccidentMatrix,cv_idx,header=False):
        '''
            提取积分值
        '''
        if not header:
            if cv_idx >= len(accident_matrix.transfer.transfer_rate_detailed_cumulant_matrix):
                flat_transfer = [] 
                flat_transfer_pured = []
            else:
                transfer = accident_matrix.transfer.transfer_rate_detailed_cumulant_matrix[cv_idx]
                flat_transfer = [x for arr in transfer for x in arr]
                transfer_pured = accident_matrix.transfer.transfer_rate_pured_detailed_cumulant_matrix[cv_idx]
                flat_transfer_pured = [x for arr in transfer_pured for x in arr]
            remove = accident_matrix.remove.remove_rate_detailed_cumulant_matrix[cv_idx]
            flat_remove = [x for arr in remove for x in arr]
            release = accident_matrix.release.release_rate_detailed_cumulant_matrix[cv_idx]
            flat_release = [x for arr in release for x in arr]
            flat_decay = list(accident_matrix.decay_detailed_cumulant_matrix[cv_idx])
            return flat_transfer + flat_transfer_pured + flat_remove + flat_release + flat_decay
        else:
            if cv_idx >= len(accident_matrix.transfer.transfer_rate_detailed_cumulant_matrix):
                transfer_header_list = []
                transfer_pured_header_list = []
            else:
                transfer = accident_matrix.transfer.transfer_info[cv_idx]
                transfer_header_list = []
                transfer_pured_header_list = []
                for nuclide_idx in range(self.n_nuclide):
                    for transfer_idx,transfer_info in enumerate(transfer[nuclide_idx]):
                        transfer_header_list.append(transfer_info['label'])
                        transfer_pured_header_list.append('pured_'+transfer_info['label'])
            remove = accident_matrix.remove.remove_info[cv_idx]
            remove_header_list = []
            for nuclide_idx in range(self.n_nuclide):
                for remove_idx,remove_info in enumerate(remove[nuclide_idx]):
                    remove_header_list.append(remove_info['label'])
            release = accident_matrix.release.release_info[cv_idx]
            release_header_list = []
            for nuclide_idx in range(self.n_nuclide):
                for release_idx,release_info in enumerate(release[nuclide_idx]):
                    release_header_list.append(release_info['label'])
            decay_header_list = []
            for nuclide_idx in range(self.n_nuclide):
                decay_header_list.append(self.nuclides[nuclide_idx]+'_decay')
            return transfer_header_list + transfer_pured_header_list + remove_header_list + release_header_list + decay_header_list
                    
    def procced(self,accident_matrix):
        '''
            推进时间步并记录数据
        '''
        if self.step_tot >= len(self.simulation_time_step_list)-1:
            self._close_and_write_header(accident_matrix)
            self.finish_swtich = True
        else:
            output_str = f'Cosmos[Accident] running: {self.time_tot/self.finish_time*100:.3e}% time: {self.time_tot:.3e}s'
            if self.write_interval < 0:
                if self.step_tot % -self.write_interval == 0:
                    print(output_str)
                    self._record(accident_matrix)
            elif self.write_interval > 0:
                if self.time_tot >= self.write_time:
                    self.write_time = round_sig(self.write_time+self.write_interval)
                    print(output_str)
                    self._record(accident_matrix)
            # 推进时间步
            self.step_tot += 1
            self.time_tot = self.simulation_time_list[self.step_tot]
            self.time_step = self.simulation_time_step_list[self.step_tot]