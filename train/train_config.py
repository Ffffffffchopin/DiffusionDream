import json
import argparse
import os

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()
        self.parse_args()

    def load_config(self):
        """从JSON文件中加载配置"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        
        with open(self.config_file, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            setattr(self, key, value)

    def parse_args(self):
        """解析命令行参数并覆盖配置"""
        parser = argparse.ArgumentParser(description="Project Configuration")
        
        # 动态添加命令行参数
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                parser.add_argument(f'--{attr_name}', type=type(getattr(self, attr_name)), help=f'Set {attr_name}')
        
        args = parser.parse_args()
        
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                if hasattr(args, attr_name) and getattr(args, attr_name) is not None:
                    setattr(self, attr_name, getattr(args, attr_name))

def get_config(config_file='config.json'):
    """返回初始化的Config对象"""
    return Config(config_file)

if __name__ == "__main__":
    config = get_config()
    print("Configuration:")
    for attr_name in dir(config):
        if not attr_name.startswith('_'):
            print(f"{attr_name}: {getattr(config, attr_name)}")

        

