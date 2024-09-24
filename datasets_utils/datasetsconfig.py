import os
from pathlib import Path as path


class DatasetsConfig:

    def __init__(self):

        self.current_file_path = path(__file__).resolve()
        # self.dataset_path = self.current_file_path
        self.project_path = self.current_file_path.parent.parent
        # self.datasets_dir_path = self.current_dir_path.parent
        self.badiu_netdisk_path = self.project_path.parent.parent / "BaiduSyncdisk"
        self.video_out_dir = self.project_path / 'datasets_utils' / 'tmp' / 'tmp_video'
        self.image_out_dir = self.project_path / 'datasets_utils' / 'tmp' / 'tmp_images'
        self.dataset_path = self.badiu_netdisk_path / "DiffusionDream_Dataset"
        self.current_processing_bvid = ""
        #self.csv_path = self.badiu_netdisk_path / 'DiffusionDream_CSV' 
        self.csv_path = self.project_path.parent / 'autodl-tmp' / 'tmp_csv'
        self.csv_file = os.path.join(self.csv_path, 'diffusiondream.csv')
        self.current_processing_keyword = ""


datasset_config = DatasetsConfig()

if __name__ == "__main__":

    # print(datasset_config.datasets_dir_path)
    print(datasset_config.project_path)
    print(datasset_config.badiu_netdisk_path)
    print(datasset_config.badiu_netdisk_path.exists())
    print(datasset_config.video_out_dir)
    print(datasset_config.csv_path.exists())
    print(os.path.exists(datasset_config.csv_file))
