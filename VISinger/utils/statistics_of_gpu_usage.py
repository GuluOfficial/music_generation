# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/7/27
# Description: GPU 使用情况统计


import os
import re
import time
import sys
from datetime import datetime
import argparse


_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


# 获取特定格式时间
def get_format_time_now():
    return datetime.now().strftime(_TIME_FORMAT)


def get_process_id(p_names=["python",]):
    pids = []
    for pname in p_names:
        cmd = "ps -aux|grep " + pname + "|cut -c 9-17"
        results = os.popen(cmd).readlines()
        for result in results:
            pid = result.strip()
            pids.append(pid)

    return pids

# 获取显存使用情况
def parseGPUMem(str_content):
    lines = str_content.split("\n")
    target_line = lines[8]
    mem_part = target_line.split("|")[2]
    use_mem = mem_part.split("/")[0]
    total_mem = mem_part.split("/")[1]
    use_mem_int = int(re.sub("\D", "", use_mem))
    total_mem_int = int(re.sub("\D", "", total_mem))
    return use_mem_int, total_mem_int


# 获取GPU使用情况
def parseGPUUseage(str_content):
    lines = str_content.split("\n")

    usage_info = []
    gid = 0
    for line in lines:
        # 查找GPU使用情况行
        if line.find("%") > -1:
            # 查找算力占用字段
            t_str = line[:line.rfind("%")+1]
            c_usage_percent = t_str[t_str.rfind("|")+1:].strip()

            # 查找显存占用字段
            t_str = t_str[:t_str.rfind("|")]
            t_str = t_str[t_str.rfind("|")+1:]
            mems = t_str.split("/")
            m_used = float(mems[0].replace("MiB", "").strip()) * 100
            m_total = float(mems[1].replace("MiB", "").strip())
            m_usage_percent = str("%.1f" % (m_used/m_total)) + "%"

            usage_info.append(f"GPU:{gid},c_used:{c_usage_percent}, m_used:{m_usage_percent}")
            gid += 1

    return usage_info


# 获取监控进程显存使用情况
def parseProcessMem(str_content, process_name):
    part = str_content.split("|  GPU       PID   Type   Process name                             Usage      |")[1]
    lines = part.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        if line.__contains__(process_name):
            mem_use = int(line[-10:-5])
            return mem_use


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pnames', type=str, default="python,", help='需要查询的进程名称（模糊匹配）')
    parser.add_argument('--logdir', type=str, default="./", help='日志保存路径')
    parser.add_argument('--dname', type=str, default="13", help='服务器代号')
    parser.add_argument('--time_interval', type=float, default=1.0, help='查询时间间隔')

    args = parser.parse_args()

    str_command = "nvidia-smi"  # 需要执行的命令
    # process_names = get_process_id(args.pnames.split(","))
    out_path = os.path.join(args.logdir, get_format_time_now() + "_gpu_usage_statistics.txt")
    time_interval = args.time_interval

    fout = open(out_path, "wt")
    # fout.write(
    #     "Timestamp\tGPU Usage Percentage\tGPU Total Mem Usage\tGPU Total Mem Usage Percentage\tProcess Mem Usage\n")

    # print("pids:", process_names)
    while True:
        # for process_name in process_names:
        try:
            out = os.popen(str_command)
            text_content = out.read()
            out.close()

            usage_percentage = parseGPUUseage(text_content)
            str_outline = get_format_time_now() + " | " + " | ".join(usage_percentage)
            print(str_outline)
            fout.write(str_outline + "\n")
            time.sleep(time_interval)
        except:
            import traceback
            print("发生异常！")
            traceback.print_exc()