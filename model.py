from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

from config import InputCard

class Model:
    """正向计算模型"""

    class Mars:
        """调用MARS作为正向计算模型"""

        def __init__(self,mars_config:InputCard.Model.Mars):
            # 读入配置 
            self.case_map:dict[str,dict[str,str]] = mars_config.case_map
            # 初始化
            self.source_term_dir,self.procedure_dir,self.mars_calculator = self._get_work_dir(mars_config.work_dir)

        def _get_work_dir(self,mars_work_dir):
            """MARS工作路径"""

            work_dir = Path(mars_work_dir)
            if not work_dir.is_absolute():
                work_dir = PROJECT_DIR / work_dir
            work_dir = work_dir.resolve()

            source_term_dir = work_dir if "SourceTermAnalysis" in work_dir.parts else work_dir / "SourceTermAnalysis"
            procedure_dir = source_term_dir.parent / "ProcedureFiles"

            if not source_term_dir.exists():
                raise FileNotFoundError(f"没有找到 SourceTermAnalysis 目录: {source_term_dir}")

            if str(source_term_dir) not in sys.path:
                sys.path.insert(0, str(source_term_dir))

            from MARS.mars_main import mars # pyright: ignore[reportMissingImports]

            procedure_dir.mkdir(parents=True,exist_ok=True)

            return source_term_dir,procedure_dir,mars

        @staticmethod
        def _set_pre_exponential(root,val,name) -> None:

            element = root.find(".//"+name+"DiffusionPreExponential")
            element.set("value", str(val))

        @staticmethod
        def _set_activation_energy(root,val,name) -> None:

            element = root.find(".//"+name+"ActivationEnergy")
            element.set("value", str(val))

        @staticmethod
        def _set_output_path(root,val) -> None:

            element = root.find(".//OutputPath")
            if element is None:
                raise ValueError("没有找到 OutputPath")
            element.set("value", str(val))

        def _set_diffusion_coef(self,root,key,val) -> None:
            """设置扩散系数到xml输入卡"""

            key_lower = key.lower()
            if "kernel" in key_lower:
                material = "Kernel"
            elif "pyc" in key_lower:
                material = "PyC"
            elif "sic" in key_lower:
                material = "SiC"
            elif "buffer" in key_lower:
                material = "Buffer"

            if 'D' in key:
                prefactor = val
                if 'log' in key:
                    prefactor = 10 ** prefactor
                self._set_pre_exponential(root,prefactor,material)
            elif 'A' in key:
                activation_energy = val
                self._set_activation_energy(root,activation_energy,material)

        def _run_mars(self,xml_path:str | Path,procedure_file_label:str,case_label:str,theta:dict[str,float]) -> dict[str,np.ndarray]:
            """调用一次MARS"""

            # xml输入卡路径
            xml_path = Path(xml_path)
            xml_path = xml_path if xml_path.suffix == ".xml" else Path(f"{xml_path}.xml")
            if not xml_path.is_absolute():
                xml_path = self.source_term_dir / xml_path
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 反演参数空间 -> 物理参数空间 -> 写入xml
            for key,val in theta.items():
                self._set_diffusion_coef(root,key,val)

            # 写入xml输出路径与新xml路径
            target_xml_path = self.procedure_dir / f"{case_label}_{procedure_file_label}.xml"
            output_dir = self.procedure_dir / f"{case_label}_{procedure_file_label}_output"
            self._set_output_path(root,str(output_dir))

            try:
                tree.write(target_xml_path,encoding="utf-8",xml_declaration=True)
                # 调用MARS
                self.mars_calculator(target_xml_path)
                # 读取MARS输出
                target_output_path = output_dir / "DiffusionReleaseRate.txt"
                data_pred = np.loadtxt(target_output_path,skiprows=2)
                release_rate_pred = data_pred[:,-1]
                t_pred = data_pred[:,0]
            finally:
                # 无论计算是否成功都删除本次调用产生的中间文件
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                if target_xml_path.exists():
                    target_xml_path.unlink()

            return {
                "time": t_pred,
                "release_rate": release_rate_pred,
            }

        def mars(self,theta:dict[str,float],procedure_file_label:str) -> dict[str,np.ndarray]:
            """MARS作为正演模型"""

            data_pred_dict = {}
            for xml_path,data_map in self.case_map.items():
                case_label = Path(xml_path).stem
                case_pred_dict = self._run_mars(xml_path,procedure_file_label,case_label,theta)
                for output_name,data_name in data_map.items():
                    data_pred_dict[data_name] = case_pred_dict[output_name]

            return data_pred_dict

    def __init__(
        self,
        model_config: InputCard.Model

    ):
        # 读入配置
        self.model_config = model_config
        # 初始化
        self.mars = self.Mars(self.model_config.mars)
        self.model_forward_dict = {
            'mars':self.mars.mars
        }

    def model_forward(self,theta:dict[str,float],procedure_file_label:str) -> dict[str,np.ndarray]:
        """正向计算模型"""

        model_forward = self.model_forward_dict[self.model_config.type]
        data_pred_dict = model_forward(theta,procedure_file_label)

        return data_pred_dict
