import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

def load_and_preprocess_data(data_dir: str, process_dir: str, selected_drugs: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    加载并预处理DepMap数据集，将处理后的数据保存到指定目录
    
    参数:
        data_dir: 原始数据所在目录
        process_dir: 处理后数据的保存目录
        selected_drugs: 要保留的药物列表，默认为None表示保留所有药物
    """
    # 数据文件路径
    file_paths = {
        "sample_info": os.path.join(data_dir, "sample_info.csv"),
        "expression": os.path.join(data_dir, "CCLE_expression.csv"),
        "crispr": os.path.join(data_dir, "CRISPR_gene_effect.csv"),
        "drug_sensitivity": os.path.join(data_dir, "Drug_sensitivity.csv")
    }
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在 - {data_dir}")
        return {}
    
    # 创建处理数据保存目录
    os.makedirs(process_dir, exist_ok=True)
    
    # 加载数据
    loaded_data = {}
    for data_type, file_path in file_paths.items():
        try:
            loaded_data[data_type] = pd.read_csv(file_path)
            print(f"成功加载 {data_type}: {loaded_data[data_type].shape}")
        except FileNotFoundError:
            print(f"警告：找不到 {data_type} 文件 - {file_path}")
            loaded_data[data_type] = None
    
    # 数据预处理
    processed_data = {}
    
    # 1. 处理样本信息
    if loaded_data["sample_info"] is not None:
        sample_info = loaded_data["sample_info"].copy()
        # 填充缺失的采集地点（避免使用inplace）
        sample_info["sample_collection_site"] = sample_info["sample_collection_site"].fillna("Unknown")
        
        # 其他列缺失值处理
        numeric_cols = sample_info.select_dtypes(include=np.number).columns
        sample_info[numeric_cols] = sample_info[numeric_cols].fillna(sample_info[numeric_cols].median())
        
        processed_data["sample_info"] = sample_info
        save_data(sample_info, "sample_info", process_dir)
    
    # 2. 处理基因表达数据
    if loaded_data["expression"] is not None:
        expression = loaded_data["expression"].copy()
        
        # 检测ID列
        id_col = detect_id_column(expression)
        if id_col:
            expr_id = expression[id_col]
            expr_values = expression.drop(id_col, axis=1)
            
            # 标准化处理（添加异常处理）
            try:
                expression_norm = (expr_values - expr_values.mean()) / (expr_values.std() + 1e-8)  # 避免除零
                expression_norm[id_col] = expr_id
                processed_data["expression"] = expression_norm
                save_data(expression_norm, "expression_norm", process_dir)
            except Exception as e:
                print(f"警告：基因表达标准化失败 - {e}")
                processed_data["expression"] = expression  # 使用原始数据
                save_data(expression, "expression_raw", process_dir)
        else:
            print("警告：基因表达数据中未找到ID列")
            processed_data["expression"] = expression
            save_data(expression, "expression_raw", process_dir)
    
    # 3. 处理CRISPR数据
    if loaded_data["crispr"] is not None:
        crispr = loaded_data["crispr"].copy()
        
        # 检测ID列
        id_col = detect_id_column(crispr)
        if id_col:
            crispr_id = crispr[id_col]
            crispr_values = crispr.drop(id_col, axis=1)
            
            # 确保所有列都是数值类型
            crispr_numeric = convert_to_numeric(crispr_values)
            
            # 处理CRISPR数据中的缺失值
            crispr_filled = crispr_numeric.fillna(crispr_numeric.median())
            
            # 重新合并ID列
            crispr_filled[id_col] = crispr_id
            processed_data["crispr"] = crispr_filled
            save_data(crispr_filled, "crispr", process_dir)
        else:
            print("警告：CRISPR数据中未找到ID列")
            # 尝试直接处理
            crispr_numeric = convert_to_numeric(crispr)
            crispr_filled = crispr_numeric.fillna(crispr_numeric.median())
            processed_data["crispr"] = crispr_filled
            save_data(crispr_filled, "crispr", process_dir)
    
    # 4. 处理药物敏感性数据
    if loaded_data["drug_sensitivity"] is not None:
        drug_sensitivity = loaded_data["drug_sensitivity"].copy()
        
        # 检测ID列
        id_col = detect_id_column(drug_sensitivity)
        if id_col and selected_drugs:
            # 确保DepMap_ID列存在并选择指定药物
            required_cols = [id_col] + [d for d in selected_drugs if d in drug_sensitivity.columns]
            if len(required_cols) > 1:  # 确保至少有ID列和一个药物
                drug_sensitivity = drug_sensitivity[required_cols]
                
                # 确保药物列是数值类型
                for col in drug_sensitivity.columns.drop(id_col):
                    drug_sensitivity[col] = pd.to_numeric(drug_sensitivity[col], errors='coerce')
                
                processed_data["drug_sensitivity"] = drug_sensitivity
                save_data(drug_sensitivity, f"drug_sensitivity_{'_'.join(selected_drugs)}", process_dir)
            else:
                print(f"警告：未找到指定的药物列，保留所有药物数据")
                processed_data["drug_sensitivity"] = drug_sensitivity
                save_data(drug_sensitivity, "drug_sensitivity_all", process_dir)
        else:
            # 保留所有药物数据
            processed_data["drug_sensitivity"] = drug_sensitivity
            save_data(drug_sensitivity, "drug_sensitivity_all", process_dir)
    
    print(f"数据处理完成，已保存到 {process_dir}")
    return processed_data

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """检测DataFrame中的ID列"""
    potential_id_cols = ['DepMap_ID', 'ModelID', 'CCLE_ID', 'Sample', 'sample_id', 'ID']
    for col in potential_id_cols:
        if col in df.columns:
            # 验证是否包含ID格式的数据
            if df[col].dtype == 'object' and df[col].str.contains('ACH-|CTG-').any():
                return col
    # 尝试通过唯一性检测
    for col in df.columns:
        if df[col].nunique() > 0.9 * len(df):  # 假设ID列的唯一值超过90%
            return col
    return None

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将DataFrame的所有列转换为数值类型"""
    numeric_df = df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    return numeric_df

def save_data(data: pd.DataFrame, name: str, directory: str) -> None:
    """保存DataFrame到CSV文件"""
    try:
        file_path = os.path.join(directory, f"{name}.csv")
        data.to_csv(file_path, index=False)
        print(f"已保存: {file_path} ({data.shape})")
    except Exception as e:
        print(f"错误：保存 {name} 数据失败 - {e}")

if __name__ == "__main__":
    # 数据路径配置
    DATA_DIR = r"D:\Caiyi\cancer_depmap_analysis\data"
    PROCESS_DIR = r"D:\Caiyi\cancer_depmap_analysis\process"
    
    # 选择要处理的药物（可根据需要修改）
    SELECTED_DRUGS = ["RS-0481", "Erlotinib"]  # 示例：处理RS-0481和Erlotinib两种药物
    
    # 执行数据处理
    processed_data = load_and_preprocess_data(
        data_dir=DATA_DIR,
        process_dir=PROCESS_DIR,
        selected_drugs=SELECTED_DRUGS
    )
    
    # 打印处理结果摘要
    if processed_data:
        print("\n处理结果摘要:")
        for data_type, df in processed_data.items():
            if df is not None:
                print(f"- {data_type}: {df.shape}")    