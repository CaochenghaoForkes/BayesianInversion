from CORE.core_xml_reader import CoreXMLReader
from CORE.core_tracer import Tracer
from CORE.core_control import CoreControl
import os
import xml.etree.ElementTree as ET
import subprocess
import copy
import numpy as np

class Nuit:
    '''
        调用Nuit
    '''
    def __init__(self,reader:CoreXMLReader,control:CoreControl):
        
        self.nuit_mode = reader.solver_dict['nuit_mode']
        self.nuit_solver = reader.solver_dict['nuit_solver']
        self.layer_number = reader.core_properties_dict['layer_number']
        self.discharge_number = reader.core_properties_dict['discharge_mesh']
        self.control = control
        self.output_history_dict = {}
        for nuclide in self.control.output_nuclide:
            self.output_history_dict[nuclide] = None
        
    def _extract_nuclide_id(self,nuclide,decay_constant_path=r'SourceTermAnalysis/TOOL/DecayLib.dat'):
        '''
            通过核素名找到其id
        '''
        with open(decay_constant_path,'r') as f:
            lines = iter(f)
            for line in lines:
                parts = line.split()
                if len(parts) < 2:
                    continue
                part = parts[1]
                if part == nuclide:
                    nuclide_id = parts[0]
                    break
            return nuclide_id

    def _generate_nuit_files(self,tracer:Tracer,library_path=r'TOOL/NUITLib_HTGR_900k',decay_constant_path=r'SourceTermAnalysis/TOOL/DecayLib.dat'):
        '''
            生成Nuit输入文件
        '''
        root = ET.Element("input")
        root.text = "\n\n\t"

        # 依赖库
        library = ET.SubElement(root, "library")
        library.set("lib_path", library_path)
        library.tail = "\n\n\t"

        # 材料
        material_data_raw = tracer.initial_material
        material = ET.SubElement(root, "material", {"unit": "gram"})
        material.text = "\n\t\t"
        material.tail = "\n\n\t"
        items = list(material_data_raw.items())  
        for i, (nuclide, mass) in enumerate(items):
            idx = self._extract_nuclide_id(nuclide,decay_constant_path)
            nuc_elem = ET.SubElement(material, "nuc", {
                "id": str(idx),
                "den": str(mass),
            })
            nuc_elem.tail = "\n\t" if i == len(items) - 1 else "\n\t\t"
        
        # 燃耗
        mode = self.nuit_mode
        mode_cfg = {
            "constpower": {
                "attr_name": "power",
                "history": tracer.power_history,
            },
            "constflux": {
                "attr_name": "flux",
                "history": tracer.neutron_flux_history,
            },
        }
        time_hist = tracer.time_step_history
        val_hist = mode_cfg[mode]["history"]
        burnup = ET.SubElement(root, "burnup")
        burnup.text = "\n\t\t"
        burnup.tail = "\n\n\t"
        attr_name = mode_cfg[mode]["attr_name"]
        last_burn_elem = None
        for t, v in zip(time_hist, val_hist):
            burn_elem = ET.SubElement(burnup, "burn", {
                "mode": mode,
                "time": str(t),
                "unit": "second",
                attr_name: str(v),
            })
            burn_elem.tail = "\n\t\t"
            last_burn_elem = burn_elem
        if last_burn_elem is not None:
            last_burn_elem.tail = "\n\t"

        # 输出
        output = ET.SubElement(root, "output")
        output.text = "\n\t\t"
        output.tail = "\n\n\t"
        table_types = ["radioactivity"]
        if self.control.decay_heat_swtich:
            table_types.append("decayheat")
        if self.control.gamma_spectrum_swtich:
            table_types.append("gammaspectra")
        if self.control.isotope_swtich:
            table_types.append("isotope")
        for t in table_types:
            tbl = ET.SubElement(output, "table", {
                "type": t,
                "print_all_step": "1",
                "integral": "0",
            })
            tbl.tail = "\n\t"
        
        # solver
        solver = ET.SubElement(root, "solver")
        solver.set("method",self.nuit_solver)
        library.tail = "\n\n\t"
        root[-1].tail = "\n\n"
        xml_str = ET.tostring(root, encoding="utf8", method="xml", xml_declaration=False).decode()

        # 输出NUIT识别xml
        file_path = os.path.join(self.control.nuit_intermediate_dir, "Tracer" + str(tracer.idx) + ".xml")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write('<?xml version="1.0"?>\n\n')
            file.write(xml_str)
        
        return file_path
        
    def _nuit(self,xml_path,nuit_path=r'TOOL/NUIT.exe'):
        '''
            调用Nuit
        '''
        proc = subprocess.Popen(nuit_path,stdin=subprocess.PIPE,text=True)
        command = xml_path
        proc.stdin.write(command + "\n")
        proc.stdin.flush()
        proc.wait()        

    def _extract_discharge_and_storage_info(self,value_list:list):
        '''
            获得排放瞬间和存储后瞬间的信息
        '''
        layer_number = self.layer_number
        discharge_number = self.discharge_number
        discharge_value_list = []
        storage_value_list = []
        break_swtich = False
        for batch in range(999): # 最大999次通过堆芯
            discharge_idx = (batch+1) * (layer_number+discharge_number) - discharge_number
            storage_idx = discharge_idx + discharge_number
            if discharge_idx >= len(value_list):
                break_swtich = True
            else:
                discharge_value_list.append(value_list[discharge_idx])
            if storage_idx >= len(value_list):
                break_swtich = True
            else:
                storage_value_list.append(value_list[storage_idx])
            if break_swtich:
                break
        
        return discharge_value_list,storage_value_list
    
    def _extract_nuit_output(self,xml_path,tracer:Tracer):
        '''
            提取Nuit的输出
        '''
        decay_heat_history = None
        power_history = None
        burnup_history = None
        gamma_spectrum_history = {}
        inventory_swtich = False
        isotope_swtich = False
        decay_heat_swtich = False
        gamma_spectrum_swtich = False
        inventory_history_dict = copy.deepcopy(self.output_history_dict)
        isotope_history_dict = copy.deepcopy(self.output_history_dict)
        with open(xml_path+'.out','r') as f:
            for line in f:
                if 'Instantaneous Radioactivity (Curies)' in line:
                    inventory_swtich = True
                if 'Instantaneous Decay Heat (Watts)' in line:
                    decay_heat_swtich = True
                if 'Gamma-ray emission rate (n/s)' in line:
                    gamma_spectrum_swtich = True
                if 'Nuclide Density(n/barn/cm)' in line:
                    isotope_swtich = True
                parts = line.split()
                if len(parts) < 2:
                    continue
                if '**************' in line and inventory_swtich and 'Instantaneous Radioactivity (Curies)' not in line:
                    inventory_swtich = False
                if '**************' in line and decay_heat_swtich and 'Instantaneous Decay Heat (Watts)' not in line:
                    decay_heat_swtich = False
                if '**************' in line and isotope_swtich and 'Nuclide Density(n/barn/cm)' not in line:
                    isotope_swtich = False
                if '> Bound' in line and gamma_spectrum_swtich and 'Gamma-ray emission rate (n/s)' not in line:
                    gamma_spectrum_swtich = False
                if parts[0] == 'power(MW):':
                    power_history = [float(x) for x in parts[1:]]
                if parts[0] == 'Flux(n/cm^2/s):':
                    neutron_flux_history = [float(x) for x in parts[1:]]
                if parts[0] == 'BU(MWd/kgHM):':
                    burnup_history = [float(x) for x in parts[1:]] # [MW][d]/[kg]->[GW][d]/[t]
                if inventory_swtich:
                    if parts[1] in self.control.output_nuclide:
                        inventory_history_dict[parts[1]] = [float(x)*3.7e10 for x in parts[2:]]
                    if 'Non-Actinide' in self.control.output_nuclide and parts[0]=='Non-Actinide':
                        inventory_history_dict[parts[0]] = [float(x)*3.7e10 for x in parts[1:]]
                    if 'Actinide' in self.control.output_nuclide and parts[0]=='Actinide':
                        inventory_history_dict[parts[0]] = [float(x)*3.7e10 for x in parts[1:]]
                    if 'Total' in self.control.output_nuclide and parts[0]=='Total' and 'Time' not in line:
                        inventory_history_dict[parts[0]] = [float(x)*3.7e10 for x in parts[1:]]
                if isotope_swtich:
                    if parts[1] in self.control.output_nuclide:
                        isotope_history_dict[parts[1]] = [float(x)*1e24 for x in parts[2:]]
                    if 'Non-Actinide' in self.control.output_nuclide and parts[0]=='Non-Actinide':
                        isotope_history_dict[parts[0]] = [float(x)*1e24 for x in parts[1:]]
                    if 'Actinide' in self.control.output_nuclide and parts[0]=='Actinide':
                        isotope_history_dict[parts[0]] = [float(x)*1e24 for x in parts[1:]]
                    if 'Total' in self.control.output_nuclide and parts[0]=='Total' and 'Time' not in line:
                        isotope_history_dict[parts[0]] = [float(x)*1e24 for x in parts[1:]]
                if decay_heat_swtich:
                    if parts[0] == 'Non-Actinide':
                        decay_heat_history = np.array([float(x)/1e6 for x in parts[1:]]) # [W]->[MW]
                    if parts[0] == 'Actinide':
                        decay_heat_history += np.array([float(x)/1e6 for x in parts[1:]]) # [W]->[MW]
                if gamma_spectrum_swtich:
                    try:
                        energy = float(parts[0])
                        is_energy = True
                    except ValueError:
                        is_energy = False
                    if is_energy:
                        gamma_spectrum_history[parts[0]] = [float(x) for x in parts[1:]]
                        
        if tracer.power_history is None:
            tracer.power_history = np.array(power_history[1:])/2 + np.array(power_history[0:-1])/2
        if tracer.neutron_flux_history is None:
            tracer.neutron_flux_history = np.array(neutron_flux_history[1:])/2 + np.array(neutron_flux_history[0:-1])/2
        tracer.decay_heat_history = np.array(decay_heat_history[1:])/2 + np.array(decay_heat_history[0:-1])/2 if decay_heat_history is not None else None
        for energy,gamma_history in gamma_spectrum_history.items():
            tracer.discharge_gamma_spectrum_history[energy],tracer.storage_gamma_spectrum_history[energy] = self._extract_discharge_and_storage_info(gamma_history)
            tracer.gamma_spectrum_history[energy] = np.array(gamma_history[1:])/2+np.array(gamma_history[0:-1])/2
        tracer.burnup_history = burnup_history
        tracer.inventory_history_dict = inventory_history_dict
        for nuclide in self.control.output_nuclide:
            tracer.discharge_inventory_history_dict[nuclide],tracer.storage_inventory_history_dict[nuclide] = self._extract_discharge_and_storage_info(inventory_history_dict[nuclide])
        tracer.isotope_history_dict = isotope_history_dict
        for nuclide in self.control.output_nuclide:
            tracer.discharge_isotope_history_dict[nuclide],tracer.storage_isotope_history_dict[nuclide] = self._extract_discharge_and_storage_info(isotope_history_dict[nuclide])
            
    def burnup_calculation(self,tracer,library_path='TOOL/NUITLib_HTGR_900k',decay_constant_path='SourceTermAnalysis/TOOL/DecayLib.dat'):
        '''
            燃耗计算
        '''
        output_str = f'Cosmos[Core] running: generate burnup history tracer{tracer.idx}'
        print(output_str)

        # 生成Nuit文件
        xml_path = self._generate_nuit_files(tracer,library_path,decay_constant_path)
        
        # 调用Nuit
        self._nuit(xml_path)
        
        # 提取信息
        self._extract_nuit_output(xml_path,tracer)
        
        # 删除中间文件
        if self.control.delete_nuit_files:
            os.remove(xml_path)
            os.remove(xml_path+'.out')

