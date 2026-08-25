from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

from config import InputCard


@dataclass
class ModelResult:
    """一次正演计算产生的全部命名预测数据集"""

    names: list[str]
    values: dict[str,np.ndarray]
    metadata: dict[str,Any] = field(default_factory=dict)

    def component(self,dataset_name: str,component_name: str) -> np.ndarray:
        """提取一个预测数据集中的指定分量"""

        component_index = self.metadata["sources"][dataset_name]["component_index"]
        return self.values[dataset_name][:,component_index[component_name]]

class Model:
    """正向计算模型"""

    @staticmethod
    def _dataset_values(
        raw_result: dict[str,np.ndarray],
        component_map: dict[str,str],
    ) -> np.ndarray:
        """按照分量映射组装一个二维预测数据集"""

        columns = [
            np.asarray(raw_result[raw_name],dtype=float)
            for raw_name in component_map.values()
        ]

        return np.column_stack(columns)

    @staticmethod
    def _assemble_result(
        raw_results: dict[str,dict[str,np.ndarray]],
        case_map: dict[str,Any],
    ) -> ModelResult:
        """将各算例原始输出组装成命名预测数据集"""

        values = {}
        source_metadata = {}

        for case_name,case_config in case_map.items():
            raw_result = raw_results[case_name]

            for dataset_name,dataset in case_config.datasets.items():
                component_order = list(dataset.component_map)
                dataset_values = Model._dataset_values(raw_result,dataset.component_map)

                values[dataset_name] = dataset_values
                source_metadata[dataset_name] = {
                    "case": case_name,
                    "component_order": component_order,
                    "component_index": {name: index for index,name in enumerate(component_order)},
                    "shape": dataset_values.shape,
                    "dtype": str(dataset_values.dtype),
                }

        return ModelResult(
            names=list(values),
            values=values,
            metadata={"sources": source_metadata},
        )

    class Mars:
        """调用MARS作为正向计算模型"""

        def __init__(
            self,
            parameter_names: list[str],
            mars_config: InputCard.Model.Mars,
        ):
            self.parameter_names = parameter_names
            self.case_map = mars_config.case_map
            self.source_term_dir,self.procedure_dir,self.mars_calculator = self._get_work_dir(mars_config.work_dir)
            self.input_setters = self._create_input_setters()

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
        def _required_element(root,path: str):
            """读取必须存在的XML节点"""

            element = root.find(path)
            if element is None:
                raise ValueError(f"MARS XML中没有找到节点: {path}")
            return element

        @classmethod
        def _set_pre_exponential(cls,root,val,name) -> None:
            """设置材料扩散前因子"""

            element = cls._required_element(root,f".//{name}DiffusionPreExponential")
            element.set("value",str(val))

        @classmethod
        def _set_activation_energy(cls,root,val,name) -> None:
            """设置材料扩散活化能"""

            element = cls._required_element(root,f".//{name}ActivationEnergy")
            element.set("value",str(val))

        @classmethod
        def _set_output_path(cls,root,val) -> None:
            """设置MARS输出目录"""

            element = cls._required_element(root,".//OutputPath")
            element.set("value",str(val))

        @staticmethod
        def _material_name(parameter_name: str) -> str:
            """从扩散参数名称识别材料"""

            name = parameter_name.lower()
            material_map = {
                "kernel": "Kernel",
                "buffer": "Buffer",
                "pyc": "PyC",
                "sic": "SiC",
            }
            for keyword,material in material_map.items():
                if keyword in name:
                    return material

            raise ValueError(f"无法从参数名称识别MARS材料: {parameter_name}")

        def _set_diffusion_coefficient(self,root,val,key) -> None:
            """设置一个材料扩散参数"""

            name = key.lower()
            material = self._material_name(key)
            if name.startswith("log_d_"):
                self._set_pre_exponential(root,10.0**float(val),material)
                return
            if name.startswith("d_"):
                self._set_pre_exponential(root,float(val),material)
                return
            if name.startswith("a_"):
                self._set_activation_energy(root,float(val),material)
                return

            raise ValueError(f"无法识别MARS扩散参数类型: {key}")

        def _create_input_setters(self):
            """创建反演参数名称到XML设置方法的映射"""

            return {
                name: partial(self._set_diffusion_coefficient,key=name)
                for name in self.parameter_names
            }

        def _set_inputs(self,root,theta: dict[str,float]) -> None:
            """将一组反演参数写入MARS XML"""

            for name,val in theta.items():
                setter = self.input_setters.get(name)
                if setter is None:
                    raise ValueError(f"没有为反演参数配置MARS输入方法: {name}")
                setter(root,float(val))

        def _run_mars(self,xml_path: str | Path,procedure_file_label: str,case_label: str,theta: dict[str,float]) -> dict[str,np.ndarray]:
            """调用一次MARS"""

            # xml输入卡路径
            xml_path = Path(xml_path)
            xml_path = xml_path if xml_path.suffix == ".xml" else Path(f"{xml_path}.xml")
            if not xml_path.is_absolute():
                xml_path = self.source_term_dir / xml_path
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 反演参数空间 -> 物理参数空间 -> 写入xml
            self._set_inputs(root,theta)

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
                raw_result = {
                    "time": data_pred[:,0],
                    "release_rate": data_pred[:,-1],
                }
            finally:
                # 无论计算是否成功都删除本次调用产生的中间文件
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                completed_xml_path = target_xml_path.with_name(f"completed_{target_xml_path.name}")
                if completed_xml_path.exists():
                    completed_xml_path.unlink()
                if target_xml_path.exists():
                    target_xml_path.unlink()

            return raw_result

        def mars(
            self,
            theta: dict[str,float],
            procedure_file_label: str,
        ) -> ModelResult:
            """依次运行MARS算例并组装预测数据集"""

            raw_results = {}
            for xml_path in self.case_map:
                case_label = Path(xml_path).stem
                raw_results[xml_path] = self._run_mars(
                    xml_path,
                    procedure_file_label,
                    case_label,
                    theta,
                )

            return Model._assemble_result(raw_results,self.case_map)

    def __init__(
        self,
        model_config: InputCard.Model,
        parameter_names: list[str],

    ):
        # 读入配置
        self.model_config = model_config
        # 初始化
        self.mars = self.Mars(parameter_names,self.model_config.mars)
        self.model_forward_dict = {
            'mars':self.mars.mars
        }

    def model_forward(
        self,
        theta: dict[str,float],
        procedure_file_label: str,
    ) -> ModelResult:
        """正向计算模型"""

        model_forward = self.model_forward_dict.get(self.model_config.type)
        if model_forward is None:
            raise NotImplementedError(
                f"未实现的正向模型: {self.model_config.type}"
            )

        return model_forward(theta,procedure_file_label)
