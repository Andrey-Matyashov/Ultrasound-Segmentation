import pandas as pd

"""
OLD - Segmented
New - БД узлы УЗИ
"""

"""
Правило оформеления файла pathes.txt. Построчно:
1) Excel-таблица с старыми данными
2) Excel-таблица с новыми данными
3) Путь к папке с старыми данными
4) Путь к папке с новыми данными
"""

N_FOLDERS = 3 # используемые папки для тестирования

def parse_pathes(df, prefix):
    """
    Принимает таблицу с данными и префикс к папке с самими данными.
    
    Формирует список путей до тифов и масок в данной папке
    """
    tif_pathes = []
    gt_pathes = []
    
    for i in range(df.shape[0]):
        
        if df.iloc[i].isna().sum() == 0:
            tif_path = df.iloc[i, 1]
            gt_path = df.iloc[i, 2]
            pat_id = str(int(df.iloc[i, 0])) if isinstance(df.iloc[i, 0], float) else df.iloc[i, 0]
            
            cur_prefix = prefix + pat_id + '/'
            tif_path = cur_prefix + tif_path + '.tif'
            gt_path = cur_prefix + gt_path + '.tif'
            
            tif_pathes.append(tif_path)
            gt_pathes.append(gt_path)
            
    return tif_pathes, gt_pathes

def get_pathes(filename):
    
    prefixes = get_prefix(filename)
    prefix_old, prefix_new = prefixes[0], prefixes[1]
    prefix_old_data, prefix_new_data = prefixes[2], prefixes[3]
    
    data_old = pd.read_excel(prefix_old)
    data_new = pd.read_excel(prefix_new)
    
    if N_FOLDERS is not None:
        data_old = data_old[:N_FOLDERS]
        data_new = data_new[:N_FOLDERS]
    
    data_new = data_new.drop(labels=0, axis=0)
    
    
    tif_pathes = []
    gt_pathes = []
    
    tmp_data_new_1 = data_new.iloc[:, [0, 2, 3]]
    tmp_data_new_2 = data_new.iloc[:, [0, 4, 5]]
    tmp_data_old_1 = data_old.iloc[:, [0, 9, 10]]
    tmp_data_old_2 = data_old.iloc[:, [0, 11, 12]]
    
    tif1, gt1 = parse_pathes(tmp_data_new_1, prefix_new_data)
    tif2, gt2 = parse_pathes(tmp_data_new_2, prefix_new_data)
    
    tif_pathes += (tif1 + tif2)
    gt_pathes += (gt1 + gt2)
    
    tif1, gt1 = parse_pathes(tmp_data_old_1, prefix_old_data)
    tif2, gt2 = parse_pathes(tmp_data_old_2, prefix_old_data)
    
    tif_pathes += (tif1 + tif2)
    gt_pathes += (gt1 + gt2)
    
    return tif_pathes, gt_pathes
    
    
def get_prefix(filename):
    prefixes = []
    with open(filename, 'r') as f:
        for prefix in f:
            prefix = prefix.strip()
            prefixes.append(prefix)
    return prefixes
            
if __name__ == '__main__':
    print()