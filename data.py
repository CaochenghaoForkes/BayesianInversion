import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from config import InputCard

@dataclass
class ObservedData:
    """已加载的全部观测数据"""

    names: list[str]
    values: dict[str,np.ndarray]
    metadata: dict[str,Any] = field(default_factory=dict)

    def component(
        self,
        data_name: str,
        component_name: str,
    ) -> np.ndarray:
        """按名称提取一个数据分量"""

        source_metadata = self.metadata["sources"][data_name]
        component_index = source_metadata["component_index"]
        if component_name not in component_index:
            raise KeyError(
                f"观测数据 {data_name} 不包含分量 {component_name}"
            )

        return self.values[data_name][:,component_index[component_name]]

class Data:
    """观测数据加载器"""

    def __init__(
        self,
        data_config: InputCard.Data
    ):
        self.data_config = data_config

    def load(self) -> ObservedData:
        """加载并检查全部观测数据"""

        values = {}
        for dataset in self.data_config.datasets:
            values[dataset.name] = self._read_csv(dataset)

        names = [dataset.name for dataset in self.data_config.datasets]
        metadata = {
            "sources": {
                dataset.name: self._source_metadata(dataset,values[dataset.name])
                for dataset in self.data_config.datasets
            }
        }

        return ObservedData(
            names=names,
            values=values,
            metadata=metadata,
        )

    @staticmethod
    def _read_csv(
        dataset: InputCard.Data.Dataset,
    ) -> np.ndarray:
        """读取一个 dataset 对应的 CSV 分量"""

        path = dataset.source.path
        component_order = dataset.source.component_order
        raw_columns = {component: [] for component in component_order}

        with Path(path).open("r",encoding="utf-8-sig",newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = set(component_order) - fieldnames
            if missing_columns:
                raise ValueError(
                    f"CSV {path} 缺少数据列: {sorted(missing_columns)}"
                )

            for row_number,row in enumerate(reader,start=2):
                for column in component_order:
                    raw_value = row[column]
                    if raw_value is None or not raw_value.strip():
                        raise ValueError(
                            f"CSV {path} 第 {row_number} 行的 {column} 为空"
                        )
                    raw_columns[column].append(raw_value)

        columns = [Data._validate_values(component,raw_columns[component]) for component in component_order]
        matrix = np.column_stack(columns)

        return Data._validate_values(dataset.name,matrix)

    @staticmethod
    def _validate_values(name: str,values: Any) -> np.ndarray:
        """统一转换并检查任意维数的数值数据"""

        try:
            array = np.asarray(values,dtype=np.float64)
        except (TypeError,ValueError) as error:
            raise ValueError(f"观测数据 {name} 不能转换为浮点数") from error

        if array.size == 0:
            raise ValueError(f"观测数据 {name} 不能为空")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"观测数据 {name} 包含非有限数值")

        return array.copy()

    @staticmethod
    def _source_metadata(
        dataset: InputCard.Data.Dataset,
        values: np.ndarray,
    ) -> dict[str,Any]:
        """生成单个观测数据的来源元数据"""

        source = dataset.source
        component_order = source.component_order
        return {
            "type":source.type,
            "path":source.path,
            "component_order":component_order.copy(),
            "component_index":{name:index for index,name in enumerate(component_order)},
            "shape":values.shape,
            "dtype":str(values.dtype),
        }
