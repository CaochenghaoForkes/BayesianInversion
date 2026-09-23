"""Execute configured forward models and assemble named prediction datasets."""

from __future__ import annotations

from dataclasses import dataclass,field
from functools import partial
from pathlib import Path
from typing import Any
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np

from config import InputCard
from three_layer_spherical_shell import (
    ThreeLayerSphericalShellSolver as _ThreeLayerSphericalShellSolver,
    arrhenius_diffusivity as _arrhenius_diffusivity,
)


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelResult:
    """一次正演计算产生的全部命名预测数据集"""

    names: list[str]
    values: dict[str,np.ndarray]
    metadata: dict[str,Any] = field(default_factory=dict)

    def component(
        self,
        dataset_name: str,
        component_name: str,
    ) -> np.ndarray:
        """提取一个预测数据集中的指定分量"""

        component_index = self.metadata["sources"][dataset_name]["component_index"]
        return self.values[dataset_name][:,component_index[component_name]]

class Model:
    """正向计算模型"""

    @staticmethod
    def _dataset_values(
        raw_result: dict[str,np.ndarray | float],
        component_map: dict[str,str],
    ) -> np.ndarray:
        """按照分量映射组装一个二维预测数据集"""

        columns = [
            np.atleast_1d(
                np.asarray(raw_result[raw_name],dtype=np.float64)
            )
            for raw_name in component_map.values()
        ]

        return np.column_stack(columns)

    @staticmethod
    def _assemble_result(
        raw_results: dict[str,dict[str,np.ndarray | float]],
        case_map: dict[str,InputCard.Model.Mars.Case],
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
            self.source_term_dir,self.procedure_dir,self.mars_calculator = (
                self._get_work_dir(mars_config.work_dir)
            )
            self.input_setters = self._create_input_setters()

        def _get_work_dir(self,mars_work_dir: str):
            """定位MARS源码和临时算例目录"""

            work_dir = Path(mars_work_dir)
            if not work_dir.is_absolute():
                work_dir = PROJECT_DIR/work_dir
            work_dir = work_dir.resolve()

            if "SourceTermAnalysis" in work_dir.parts:
                source_term_dir = work_dir
            else:
                source_term_dir = work_dir/"SourceTermAnalysis"

            if not source_term_dir.is_dir():
                raise FileNotFoundError(
                    f"没有找到 SourceTermAnalysis 目录: {source_term_dir}"
                )

            procedure_dir = source_term_dir.parent/"ProcedureFiles"

            if str(source_term_dir) not in sys.path:
                sys.path.insert(0, str(source_term_dir))

            from MARS.mars_main import mars  # pyright: ignore[reportMissingImports]

            return source_term_dir,procedure_dir,mars

        @staticmethod
        def _required_element(root,path: str):
            """读取必须存在的XML节点"""

            element = root.find(path)
            if element is None:
                raise ValueError(f"MARS XML中没有找到节点: {path}")
            return element

        @classmethod
        def _set_pre_exponential(cls,root,value: float,material: str) -> None:
            """设置材料扩散前因子"""

            element = cls._required_element(
                root,
                f".//{material}DiffusionPreExponential",
            )
            element.set("value",str(value))

        @classmethod
        def _set_activation_energy(cls,root,value: float,material: str) -> None:
            """设置材料扩散活化能"""

            element = cls._required_element(
                root,
                f".//{material}ActivationEnergy",
            )
            element.set("value",str(value))

        @classmethod
        def _set_output_path(cls,root,value: str | Path) -> None:
            """设置MARS输出目录"""

            element = cls._required_element(root,".//OutputPath")
            element.set("value",str(value))

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

        def _set_diffusion_coefficient(
            self,
            root,
            value: float,
            parameter_name: str,
        ) -> None:
            """设置一个材料扩散参数"""

            name = parameter_name.lower()
            material = self._material_name(parameter_name)
            if name.startswith(("log_d_","d_log_")):
                self._set_pre_exponential(root,10.0**float(value),material)
                return
            if name.startswith("d_"):
                self._set_pre_exponential(root,float(value),material)
                return
            if name.startswith("a_"):
                self._set_activation_energy(root,float(value),material)
                return

            raise ValueError(
                f"无法识别MARS扩散参数类型: {parameter_name}"
            )

        def _create_input_setters(self):
            """创建反演参数名称到XML设置方法的映射"""

            return {
                name:partial(
                    self._set_diffusion_coefficient,
                    parameter_name=name,
                )
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
            file_label = f"{case_label}_{procedure_file_label}"
            target_xml_path = self.procedure_dir/f"{file_label}.xml"
            output_dir = self.procedure_dir/f"{file_label}_output"
            completed_xml_path = target_xml_path.with_name(
                f"completed_{target_xml_path.name}"
            )
            self._set_output_path(root,output_dir)

            self.procedure_dir.mkdir(parents=True,exist_ok=True)
            temporary_paths = (target_xml_path,output_dir,completed_xml_path)
            if any(path.exists() for path in temporary_paths):
                raise FileExistsError(f"MARS临时算例名称冲突: {file_label}")

            try:
                tree.write(target_xml_path,encoding="utf-8",xml_declaration=True)
                # 调用MARS
                self.mars_calculator(target_xml_path)
                # 读取MARS输出
                target_output_path = output_dir / "DiffusionReleaseRate.txt"
                data_pred = np.atleast_2d(
                    np.loadtxt(target_output_path,skiprows=2)
                )
                raw_result = {
                    "time": data_pred[:,0],
                    "release_rate": data_pred[:,-1],
                }
            finally:
                # 无论计算是否成功都删除本次调用产生的中间文件
                if output_dir.exists():
                    shutil.rmtree(output_dir)
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
                try:
                    raw_results[xml_path] = self._run_mars(
                        xml_path,
                        procedure_file_label,
                        case_label,
                        theta,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"MARS算例运行失败: {xml_path}"
                    ) from error

            return Model._assemble_result(raw_results,self.case_map)

    class ThreeLayerSphericalShell:
        """调用轻量三层球壳半解析模型计算多组恒温释放实验"""

        model_name = "three_layer_spherical_shell"

        def __init__(
            self,
            parameter_names: list[str],
            config: InputCard.Model.ThreeLayerSphericalShell,
        ):
            self.parameter_names = parameter_names
            self.times = np.asarray(config.times,dtype=np.float64)
            self.datasets = config.datasets

            if len(self.parameter_names) != 4:
                raise ValueError(
                    "三层球壳模型必须且只能接收四个扩散参数，"
                    f"当前为 {self.parameter_names}"
                )
            self._diffusion_parameters({name:1.0 for name in self.parameter_names})

        @staticmethod
        def _matching_inputs(
            inputs: dict[str,float],
            aliases: set[str],
        ) -> list[tuple[str,float]]:
            """按不区分大小写的别名查找输入"""

            return [
                (name,float(value))
                for name,value in inputs.items()
                if name.lower() in aliases
            ]

        @classmethod
        def _required_input(
            cls,
            inputs: dict[str,float],
            physical_name: str,
            aliases: set[str],
        ) -> float:
            """读取一个必须且只能出现一次的物理输入"""

            matches = cls._matching_inputs(inputs,aliases)
            if not matches:
                raise ValueError(
                    f"{cls.model_name} 缺少输入参数: {physical_name}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"{cls.model_name} 的 {physical_name} 输入重复: "
                    f"{[name for name,_ in matches]}"
                )
            return matches[0][1]

        @classmethod
        def _pre_exponential(
            cls,
            inputs: dict[str,float],
            material: str,
        ) -> float:
            """读取 D0 名称含 log 的输入按以 10 为底的对数还原。"""

            material = material.lower()
            direct = cls._matching_inputs(
                inputs,{f"d0_{material}",f"d_{material}"}
            )
            logarithmic = cls._matching_inputs(
                inputs,
                {
                    f"log_d0_{material}",f"log_d_{material}",
                    f"d0_log_{material}",f"d_log_{material}",
                },
            )
            matches = [*direct,*logarithmic]
            if not matches:
                raise ValueError(
                    f"{cls.model_name} 缺少 D0_{material} 输入"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"{cls.model_name} 的 D0_{material} 输入重复: "
                    f"{[name for name,_ in matches]}"
                )
            if logarithmic:
                return float(10.0**logarithmic[0][1])
            return direct[0][1]

        @classmethod
        def _diffusion_parameters(
            cls,
            inputs: dict[str,float],
        ) -> tuple[float,float,float,float]:
            """按 PyC D0、PyC A、SiC D0、SiC A 的顺序读取参数。"""

            return (
                cls._pre_exponential(inputs,"pyc"),
                cls._required_input(inputs,"A_PyC",{"a_pyc","activation_energy_pyc"}),
                cls._pre_exponential(inputs,"sic"),
                cls._required_input(inputs,"A_SiC",{"a_sic","activation_energy_sic"}),
            )

        def _release_rate(
            self,
            temperature: float,
            d0_pyc: float,
            activation_energy_pyc: float,
            d0_sic: float,
            activation_energy_sic: float,
        ) -> np.ndarray:
            """计算一个温度下的三层球壳外表面单位面积释放率。"""

            diffusivity_pyc = _arrhenius_diffusivity(d0_pyc,activation_energy_pyc,temperature)
            diffusivity_sic = _arrhenius_diffusivity(d0_sic,activation_energy_sic,temperature)
            solver = _ThreeLayerSphericalShellSolver(diffusivity_pyc,diffusivity_sic)

            return solver.release_rate(self.times)

        def forward(
            self,
            theta: dict[str,float],
            procedure_file_label: str,
        ) -> ModelResult:
            """返回各温度实验对应的命名释放率数据集。"""

            del procedure_file_label
            diffusion_parameters = self._diffusion_parameters(theta)
            values = {}
            sources = {}

            for dataset_name,dataset in self.datasets.items():
                raw_result = {
                    "time":self.times,
                    "release_rate":self._release_rate(dataset.temperature,*diffusion_parameters),
                }
                component_order = list(dataset.component_map)
                dataset_values = Model._dataset_values(raw_result,dataset.component_map)
                values[dataset_name] = dataset_values
                sources[dataset_name] = {
                    "case":dataset_name,
                    "temperature":float(dataset.temperature),
                    "component_order":component_order,
                    "component_index":{name:index for index,name in enumerate(component_order)},
                    "shape":dataset_values.shape,
                    "dtype":str(dataset_values.dtype),
                }

            return ModelResult(
                names=list(values),
                values=values,
                metadata={"sources":sources},
            )

    def __init__(
        self,
        model_config: InputCard.Model,
        parameter_names: list[str],

    ):
        self.model_config = model_config
        self.parameter_names = parameter_names
        self.forward = self._create_forward_model()

    def _create_forward_model(self):
        """只初始化当前启用的模型"""

        if self.model_config.type == "mars":
            mars = self.Mars(
                self.parameter_names,
                self.model_config.mars,
            )
            return mars.mars

        if self.model_config.type == "three_layer_spherical_shell":
            config = self.model_config.three_layer_spherical_shell
            if config is None:
                raise ValueError("缺少 model.three_layer_spherical_shell 配置")
            model = self.ThreeLayerSphericalShell(
                self.parameter_names,config
            )
            return model.forward

        raise NotImplementedError(
            f"未实现的正向模型: {self.model_config.type}"
        )

    def model_forward(
        self,
        theta: dict[str,float],
        procedure_file_label: str,
    ) -> ModelResult:
        """正向计算模型"""

        actual_names = set(theta)
        expected_names = set(self.parameter_names)
        if actual_names != expected_names:
            missing = sorted(expected_names-actual_names)
            extra = sorted(actual_names-expected_names)
            raise ValueError(
                f"模型输入参数不一致, 缺少 {missing}, 多余 {extra}"
            )

        return self.forward(theta,procedure_file_label)
