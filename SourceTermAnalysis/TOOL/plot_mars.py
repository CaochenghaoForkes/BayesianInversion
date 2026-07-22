import numpy as np
import matplotlib.pyplot as plt
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

data = extract_txt('OUTPUT\COREOUTPUT\MarsIntermediateFiles\Tracer0_Cs137\ReleaseRate.txt')
t = data['Time(s)'] 
T = data['Element']
plt.plot(t,T)
plt.show()