from typing import Dict, List, Tuple, Optional, Sequence, Union, Callable
import numpy as np

class TimeSeries:
    def __init__(self, times: List[float], values: List[Union[float, np.ndarray]]):
        assert len(times) == len(values) and len(times) >= 1
        pairs = sorted(zip(times, values), key=lambda x: x[0])

        self._times = np.array([p[0] for p in pairs], dtype=float)
        # 把每个 value 都转成 ndarray（标量会变成 0-d array）
        self._values = [np.asarray(p[1], dtype=float) for p in pairs]

        # shape 一致性检查（允许全是标量；允许全是同shape数组）
        shapes = [v.shape for v in self._values]
        if len(set(shapes)) != 1:
            raise ValueError(f"values must have the same shape, got shapes={set(shapes)}")

    def at(self, t: float, mode: str = "linear", side: str = "right"):
        ts = self._times
        vs = self._values
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]

        import bisect
        if side == "left":
            exact = bisect.bisect_left(ts, t)
            if exact < len(ts) and ts[exact] == t:
                return vs[exact]
        i = bisect.bisect_right(ts, t) - 1

        if mode == "step":
            return vs[i]

        t0, t1 = ts[i], ts[i + 1]
        v0, v1 = vs[i], vs[i + 1]

        w = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return v0 + w * (v1 - v0)

class Registry:
    '''
        注册表
    '''
    def __init__(self):
        self._reg: Dict[str, Dict[str, Callable]] = {}

    def register(self, family: str, name: str, fn: Callable):
        self._reg.setdefault(family, {})[name] = fn

    def get(self, family: str, name: str) -> Callable:
        try:
            return self._reg[family][name]
        except KeyError:
            raise ValueError(f"Model not found: {family}:{name}")

def round_sig(x, sig=5):
    '''
        保留sig位有效数字
    '''
    if isinstance(x,float):
        return float(f"{x:.{sig}g}")
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore'):
        mags = np.floor(np.log10(np.abs(x)))
    decimals = sig - 1 - mags
    result = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = it[0]
        if val == 0:
            result[idx] = 0
        else:
            d = int(decimals[idx])
            result[idx] = np.around(val, decimals=d)
        it.iternext()
    return result

def formatting_line(items, width=25, precision=10):
    cells = []
    for x in items:
        if isinstance(x, (int, float)):
            # 数字：科学计数法，宽度和小数位都是参数
            cells.append(f"{x:<{width}.{precision}e}")
        else:
            # 其他当作字符串（这里用的是左对齐 < ，你注释写的是右对齐，可以改成 >）
            cells.append(f"{str(x):<{width}}")
    return "".join(cells) + "\n"

def write_header(filepath,line1,line2):
    '''
        在已经写好数据的txt顶上添加列名
    '''
    with open(filepath, "r", encoding="utf-8") as f:
        old_content = f.read()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(line1)
        f.write(line2)
        f.write(old_content)

def extract_txt(file_path):
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
