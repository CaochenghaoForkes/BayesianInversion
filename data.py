import csv
from pathlib import Path

from config import InputCard

class Data:
    """观测数据"""

    def __init__(
        self,
        data_config: InputCard.Data
    ):
        # 读入配置
        self.data_config = data_config
        self.name = self.data_config.name
        self.value = self.data_config.value
        self.data_source = self.data_config.data_source
        # 初始化
        self.extract_dict = {
            'csv':self._extract_data_from_csv,
            'json':self._extract_data_from_raw_json
        }
        self.data_dict = self._extract_data()

    def _extract_data(self) -> dict[str,list[float]]:
        """提取观测数据"""

        if self.data_source == "json":
            mode = "json"
        else:
            mode = "csv"

        data_dict = self.extract_dict[mode](self.data_source,self.name,self.value)

        return data_dict

    @staticmethod
    def _extract_data_from_csv(data_source:str,names:list[str],value:list[list[float]]) -> dict[str,list[float]]:
        """从csv提取数据"""

        data_dict = {name: [] for name in names}
        with Path(data_source).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for name in names:
                    data_dict[name].append(float(row[name]))

        return data_dict

    @staticmethod
    def _extract_data_from_raw_json(data_source:str,names:list[str],values:list[list[float]]) -> dict[str,list[float]]:
        """从原始json中提取数据"""

        data_dict = {}
        for name,value in zip(names,values):
            data_dict[name] = value.copy()

        return data_dict
