# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2020/11/16
# Description: 配置文件加载


import yaml
import traceback


class ConfigLoader:
    def __init__(self, config_file_path="./conf/config.yaml"):
        try:
            with open(config_file_path, 'rt', encoding="utf-8") as f:
                self.config = yaml.load(f, Loader=yaml.Loader)
        except Exception as e:
            traceback.print_exc()
            print("Load config file failure!")
            exit(1)

    # 加载预处理参数
    def get_preprocessing_params(self):
        return self.config['preprocessing_params']

    # 加载模型参数
    def get_model_params(self):
        return self.config['model_params']

    # 加载训练参数
    def get_training_params(self):
        return self.config['training_params']

    def get_all_params(self):
        return self.config
