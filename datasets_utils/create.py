import pandas as pd

# 定义 CSV 文件的列名
columns = ['info','keyword','action','previous_frame_1','previous_frame_2','previous_frame_3','previous_frame_4','previous_frame_5','previous_frame_6','previous_frame_7','previous_frame_8','previous_frame_9','previous_frame_10','current_frame']




# 创建 DataFrame
df = pd.DataFrame(columns=columns)

# 将 DataFrame 写入 CSV 文件
df.to_csv('/root/autodl-tmp/tmp_csv/diffusiondream.csv', index=False)

print("CSV file has been created: example_pandas.csv")