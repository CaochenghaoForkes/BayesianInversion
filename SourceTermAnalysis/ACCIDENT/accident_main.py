from ACCIDENT.accident_xml_reader import AccidentXMLReader
from ACCIDENT.accident_control import AccidentControl
from ACCIDENT.accident_matrix import AccidentMatrix

def accident(accident_xml_path):

    # 读取对象
    reader = AccidentXMLReader(accident_xml_path)

    # 读取信息
    reader.read_configuration()

    # 控制对象
    control = AccidentControl(reader)

    # 事故矩阵对象
    matrix = AccidentMatrix(reader)

    # 开启时间循环
    while not control.finish_swtich:

        # 更新转移
        matrix.transfer.update_transfer_diag(control.time_tot+control.time_step/2,control.step_tot)

        # 更新消失
        matrix.remove.update_remove_vector(control.time_tot+control.time_step/2,control.step_tot)

        # 更新释放
        matrix.release.update_release_vector(control.time_tot+control.time_step/2,control.step_tot,control.time_step)            

        # 生成事故矩阵
        matrix.generate_completed_accident_matrix()

        # 更新盘存量
        matrix.update_inventory(control.time_tot,control.step_tot,control.time_step)

        # 计算积分值
        matrix.calculate_integral(control.time_step)

        # 推进时间步
        control.procced(matrix)