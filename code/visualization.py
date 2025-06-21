import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
from matplotlib.colors import ListedColormap
import datetime
import matplotlib.font_manager as fm

# 清除matplotlib字体缓存（解决字体不生效问题）
try:
    fm._rebuild()
except:
    pass

sns.set_theme(style="whitegrid", palette="bright")


class DepMapAnalyzer:
    def __init__(self, process_dir: str, output_dir: str, id_mapping: dict = None):
        self.process_dir = process_dir
        self.output_dir = output_dir
        self.sample_info = None
        self.expression = None
        self.crispr = None
        self.drug_sensitivity = None
        self.merged_data = None
        self.mutations = pd.DataFrame()  # 初始化空DataFrame避免报错
        self.copy_number = pd.DataFrame()  # 初始化空DataFrame避免报错
        self.id_columns = {
            "sample_info": "DepMap_ID",
            "expression": "DepMap_ID",
            "crispr": "DepMap_ID",
            "drug_sensitivity": "DepMap_ID",
            "mutations": "DepMap_ID",
            "copy_number": "DepMap_ID"
        }
        if id_mapping:
            self.id_columns.update(id_mapping)
        os.makedirs(self.output_dir, exist_ok=True)
        self.report = []  # 存储分析报告内容

    def load_data(self):
        try:
            file_paths = {
                "sample_info": os.path.join(self.process_dir, "sample_info.csv"),
                "expression": os.path.join(self.process_dir, "expression_norm.csv"),
                "crispr": os.path.join(self.process_dir, "crispr.csv"),
                "drug_sensitivity": os.path.join(self.process_dir, "drug_sensitivity_all.csv"),
                "mutations": os.path.join(self.process_dir, "mutations.csv"),  
                "copy_number": os.path.join(self.process_dir, "copy_number.csv")  
            }
            for name, path in file_paths.items():
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        if not df.empty:
                            id_col = self.id_columns.get(name, "DepMap_ID")
                            if id_col not in df.columns:
                                self.report.append(f"Warning: ID column '{id_col}' not found in {name}, automatically detecting valid ID column\n")
                                id_col = self._detect_valid_id(df)
                                self.id_columns[name] = id_col
                            df[id_col] = df[id_col].astype(str).str.strip().str.upper()
                            setattr(self, name, df)
                            self.report.append(f"\nSummary of {name} data:\n")
                            self.report.append(f"ID column: {id_col}, Current type: {df[id_col].dtype}\n")
                            self.report.append(f"Data size: {df.shape}\n")
                            self.report.append(f"Top 5 values in ID column: {df[id_col].head().tolist()[:5]}\n")
                            self.report.append(f"Number of unique values in ID column: {df[id_col].nunique()}\n")
                        else:
                            self.report.append(f"Warning: {name} file is empty - {path}\n")
                            setattr(self, name, pd.DataFrame()) 
                    except Exception as e:
                        self.report.append(f"Error reading {name}: {e}\n")
                        setattr(self, name, pd.DataFrame())
                else:
                    self.report.append(f"Warning: {name} file not found - {path}\n")
                    setattr(self, name, pd.DataFrame())
            return self._check_data_loaded()
        except Exception as e:
            self.report.append(f"Data loading failed: {e}\n")
            return False

    def _detect_valid_id(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.contains('ACH-').any():
                return col
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                return col
        return df.columns[0] if len(df.columns) > 0 else None

    def _check_data_loaded(self) -> bool:
        return (
            not self.sample_info.empty and
            not self.expression.empty and
            not self.crispr.empty and
            not self.drug_sensitivity.empty
        )

    def merge_datasets(self):
        if not self._check_data_loaded():
            self.report.append("Please load all valid data first\n")
            return False
        sample_id = self.id_columns["sample_info"]
        expr_id = self.id_columns["expression"]
        crispr_id = self.id_columns["crispr"]
        drug_id = self.id_columns["drug_sensitivity"]

        try:
            merged = pd.merge(
                self.sample_info, 
                self.expression, 
                left_on=sample_id, 
                right_on=expr_id, 
                how="inner"
            )
            self.report.append(f"After merging sample information and gene expression: {merged.shape}\n")
            merged = pd.merge(
                merged, 
                self.crispr, 
                left_on=sample_id, 
                right_on=crispr_id, 
                how="inner"
            )
            self.report.append(f"After merging CRISPR data: {merged.shape}\n")
            self.merged_data = pd.merge(
                merged, 
                self.drug_sensitivity, 
                left_on=sample_id, 
                right_on=drug_id, 
                how="inner"
            )
            self.report.append(f"After merging all data: {self.merged_data.shape}\n")
            return True
        except Exception as e:
            self.report.append(f"Error merging data: {e}\n")
            return False

    # 1. 简单图：组织分布柱状图（基础版）
    def visualize_simple_tissue_dist(self, top_n=10):
        if self.sample_info.empty:
            self.report.append("Sample information is empty, cannot draw tissue distribution bar chart\n")
            return
        tissue_dist = self.sample_info["sample_collection_site"].value_counts().head(top_n).reset_index()
        tissue_dist.columns = ["Tissue", "Number of cell lines"]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Number of cell lines", y="Tissue", data=tissue_dist, palette="Blues_r")
        plt.title("Top 10 tissue-derived cell line distribution", fontsize=14)
        plt.xlabel("Number of cell lines", fontsize=12)
        plt.ylabel("Tissue type", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "Simple tissue distribution bar chart.png")
        plt.savefig(save_path, dpi=300)
        self.report.append(f"Simple tissue distribution chart saved to: {save_path}\n")

    # 3. 复杂图：基因-药物相关性热图（自动选基因+药物）
    def visualize_complex_gene_drug_corr(self, top_n_genes=10, top_n_drugs=5):
        if self.merged_data.empty:
            self.report.append("Merged data is empty, cannot draw gene-drug correlation heatmap\n")
            return
        # 自动选高变基因（同前）
        gene_vars = self.expression.drop(columns=[self.id_columns["expression"]]).var(axis=0)
        top_genes = gene_vars.sort_values(ascending=False).head(top_n_genes).index.tolist()
        # 自动选高变药物（按方差）
        drug_vars = self.drug_sensitivity.drop(columns=[self.id_columns["drug_sensitivity"]]).var(axis=0)
        top_drugs = drug_vars.sort_values(ascending=False).head(top_n_drugs).index.tolist()
        
        # 提取交集样本
        sample_ids = self.merged_data[self.id_columns["sample_info"]].unique()
        expr_subset = self.expression[self.expression[self.id_columns["expression"]].isin(sample_ids)]
        drug_subset = self.drug_sensitivity[self.drug_sensitivity[self.id_columns["drug_sensitivity"]].isin(sample_ids)]
        
        # 整理基因表达和药物敏感性数据
        expr_matrix = expr_subset.set_index(self.id_columns["expression"])[top_genes]
        drug_matrix = drug_subset.set_index(self.id_columns["drug_sensitivity"])[top_drugs]
        merged_matrix = pd.merge(expr_matrix, drug_matrix, left_index=True, right_index=True, how="inner")
        
        # 计算相关性
        corr = merged_matrix.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Gene-drug sensitivity correlation heatmap (automatically select highly variable features)", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "Complex gene-drug correlation heatmap.png")
        plt.savefig(save_path, dpi=300)
        self.report.append(f"Complex gene-drug correlation heatmap saved to: {save_path}\n")

    # 4. 复杂图：样本聚类树+热图（多组学整合）
    def visualize_complex_sample_clustering(self, top_n_genes=20):
        if self.merged_data.empty:
            self.report.append("Merged data is empty, cannot draw sample clustering heatmap\n")
            return
        # 自动选高变基因
        gene_vars = self.expression.drop(columns=[self.id_columns["expression"]]).var(axis=0)
        top_genes = gene_vars.sort_values(ascending=False).head(top_n_genes).index.tolist()
        expr_subset = self.expression[[self.id_columns["expression"]] + top_genes]
        
        # 合并样本信息（组织标签）
        merged_expr = pd.merge(
            self.sample_info[[self.id_columns["sample_info"], "sample_collection_site"]],
            expr_subset,
            left_on=self.id_columns["sample_info"],
            right_on=self.id_columns["expression"],
            how="inner"
        )
        heatmap_data = merged_expr.set_index(["sample_collection_site", self.id_columns["sample_info"]]).drop(columns=[self.id_columns["expression"]])
        
        # 聚类 + 绘图
        row_linkage = linkage(heatmap_data, method='average', metric='correlation')
        col_linkage = linkage(heatmap_data.T, method='average', metric='correlation')
        
        plt.figure(figsize=(16, 12))
        g = sns.clustermap(
            heatmap_data, 
            row_linkage=row_linkage, 
            col_linkage=col_linkage, 
            cmap="viridis", 
            xticklabels=True, 
            yticklabels=False, 
            cbar_kws={"label": "Expression level"},
            tree_kws={"color": "gray"}
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.suptitle(f"Sample clustering heatmap (Top {top_n_genes} highly variable genes + tissue labels)", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "Complex sample clustering heatmap.png")
        g.savefig(save_path, dpi=300)
        self.report.append(f"Complex sample clustering heatmap saved to: {save_path}\n")

    # 5. 新增复杂图：CRISPR基因必需性分布（箱线图）
    def visualize_complex_crispr_essentiality(self, top_n=10):
        if self.crispr.empty:
            self.report.append("CRISPR data is empty, cannot draw gene essentiality distribution\n")
            return
        # 提取基因必需性
        crispr_data = self.crispr.drop(columns=[self.id_columns["crispr"]])
        gene_vars = crispr_data.var(axis=0)
        top_genes = gene_vars.sort_values(ascending=False).head(top_n).index.tolist()
        crispr_subset = crispr_data[top_genes]

        melted = crispr_subset.melt(var_name="Gene", value_name="Essentiality score")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Gene", y="Essentiality score", data=melted, palette="Set2")
        plt.title(f"Top {top_n} genes with the highest essentiality (CRISPR screening)", fontsize=14)
        plt.xlabel("Gene name", fontsize=12)
        plt.ylabel("Essentiality score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "Complex CRISPR gene essentiality distribution boxplot.png")
        plt.savefig(save_path, dpi=300)
        self.report.append(f"Complex CRISPR gene essentiality distribution boxplot saved to: {save_path}\n")

    # 生成分析报告
    def generate_report(self):
        report_path = os.path.join(self.output_dir, "DepMap analysis report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== DepMap Data Analysis Report ===\n")
            f.write(f"Analysis time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.writelines(self.report)
            f.write("\n=== Chart Explanation ===\n")
            f.write("1. Simple tissue distribution bar chart: Shows the number of cell lines from the top 10 tissue sources to quickly understand the data distribution\n")
            f.write("3. Complex gene-drug correlation heatmap: Analyzes the association between gene expression and drug sensitivity to mine drug sensitivity markers\n")
            f.write("4. Complex sample clustering heatmap: Integrates gene expression and tissue labels to reveal sample molecular subtypes and heterogeneity\n")
            f.write("5. Complex CRISPR gene essentiality distribution boxplot: Shows highly essential genes from CRISPR screening to find potential targets\n")
        self.report.append(f"\nAnalysis report saved to: {report_path}\n")

    # 统一执行流程（自动选基因+多图+报告）
    def run_analysis(self):
        self.report.append("Starting DepMap data visualization analysis...\n")
        if not self.load_data():
            self.report.append("Data loading failed, analysis terminated\n")
            self.generate_report()
            return
        if not self.merge_datasets():
            self.report.append("Dataset merging failed, analysis terminated\n")
            self.generate_report()
            return
        
        # 生成可视化图表
        self.visualize_simple_tissue_dist(top_n=10)
        self.visualize_complex_gene_drug_corr(top_n_genes=10, top_n_drugs=5)
        self.visualize_complex_sample_clustering(top_n_genes=20)
        self.visualize_complex_crispr_essentiality(top_n=10)
        self.generate_report()


if __name__ == "__main__":
    process_dir = r"D:\Caiyi\cancer_depmap_analysis\process"
    output_dir = r"D:\Caiyi\cancer_depmap_analysis\visualizations"
    analyzer = DepMapAnalyzer(process_dir, output_dir)
    analyzer.run_analysis()