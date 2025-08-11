# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2020/11/16
# Description: 用于记录日志


import os
from datetime import datetime


class Logger:
    def __init__(self, log_save_dir="", log_name="train.log"):
        os.makedirs(log_save_dir, exist_ok=True)
        # log_save_path = os.path.join(log_save_dir, log_name)
        # self._log_file = open(log_save_path, 'wt', encoding='utf-8')

        self._date_format = '%Y-%m-%d %H:%M:%S.%f'

    def log(self, log_msg):
        date_msg = datetime.now().strftime(self._date_format)[:-3]

        msg = '[%s] %s\n' % (date_msg, log_msg)
        # self._log_file.write(msg)

        print(msg, end="")

    # def close(self):
    #     self._log_file.close()
