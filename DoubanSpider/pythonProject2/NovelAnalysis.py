import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 加载并合并三个CSV文件
file_paths = [
    r'E:\BaiduNetdiskDownload\DoubanSpider\豆瓣图书标签_ TestAll.csv',
    r'E:\BaiduNetdiskDownload\DoubanSpider\豆瓣图书标签_ TestAll(1).csv',
    r'E:\BaiduNetdiskDownload\DoubanSpider\豆瓣图书标签_ TestAll(2).csv'
]

df_list = [pd.read_csv(file_path) for file_path in file_paths]
df = pd.concat(df_list, ignore_index=True)

# 检查重复项（根据'图片'字段）
duplicate_mask = df.duplicated(subset=['图片'])
duplicate_count = duplicate_mask.sum()

print(f"Number of duplicate rows: {duplicate_count}")

# 删除重复项
df = df.drop_duplicates(subset=['图片'])
print(f"Number of rows after removing duplicates: {len(df)}")

# 提取评论人数并转换为整数，填充缺失值为0
df['评论人数'] = df['pl'].str.extract(r'(\d+)')
df['评论人数'] = df['评论人数'].fillna(0).astype(int)

# 计算所有书籍的平均评分
C = df['数量'].mean()

# 设置最低评论人数阈值，通常取95%的书籍评论人数作为阈值
m = df['评论人数'].quantile(0.95)

# 计算加权平均评分
def weighted_rating(x, m=m, C=C):
    v = x['评论人数']
    R = x['数量']
    return (v / (v + m) * R) + (m / (v + m) * C)

# 应用加权平均评分计算重要性
df['重要性'] = df.apply(weighted_rating, axis=1)

# 确保没有缺失值
df['重要性'] = df['重要性'].fillna(0)

# 使用K-means聚类进行分组
kmeans = KMeans(n_clusters=5, random_state=0)
df['等级'] = kmeans.fit_predict(df[['重要性']])

# 获取每个聚类的中心点
cluster_centers = kmeans.cluster_centers_.flatten()

# 按中心点的大小排序，并映射到等级名称
sorted_indices = np.argsort(cluster_centers)
mapping = {sorted_indices[i]: grade for i, grade in enumerate(['很低', '低', '中', '高', '很高'])}
df['等级'] = df['等级'].map(mapping)

# 拆分标题字段
def split_title(title):
    parts = title.split('/')
    if len(parts) == 4:
        return parts[0].strip(), None, parts[1].strip(), parts[2].strip(), parts[3].strip()
    elif len(parts) == 5:
        return parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip()
    else:
        # Handle unexpected formats by returning None for missing parts
        return (parts + [None] * (5 - len(parts)))[:5]

df[['作者', '译者', '出版社', '出版时间', '价格']] = df['标题'].apply(lambda x: pd.Series(split_title(x)))

# 改进出版时间和价格的提取逻辑
df['出版时间'] = df['出版时间'].str.extract(r'(\d{4}-\d{1,2}(-\d{1,2})?)')[0].fillna(
    df['出版时间'].str.extract(r'(\d{4})')[0])

# 改进价格提取逻辑，处理带“元”和不带“元”的情况
df['价格'] = df['价格'].str.extract(r'(\d+\.\d+元?)')[0].fillna(
    df['价格'].str.extract(r'(\d+\.?\d*)')[0])
df['价格'] = df['价格'].apply(lambda x: f"{x}元" if pd.notna(x) and not x.endswith('元') else x)

# 保存为新的文件
output_file_path = './书籍等级分类_最终版1.xlsx'
df.to_excel(output_file_path, index=False)

# 过滤“很高”等级数据，并按重要性从高到低排序
high_importance_df = df[df['等级'] == '很高'].sort_values(by='重要性', ascending=False)
output_file_path_1 = './TestHighest1.xlsx'
high_importance_df.to_excel(output_file_path_1, index=False)

# 在Pycharm中显示数据
print(high_importance_df)
