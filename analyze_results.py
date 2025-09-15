import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

class JavaMetricsAnalyzer:
    def __init__(self, results_file="results/java_metrics_results.csv"):
        self.results_file = results_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.results_file)
            print(f"Dados carregados: {len(self.df)} repositórios")
            self.clean_data()
        except FileNotFoundError:
            print(f"Arquivo {self.results_file} não encontrado!")
    
    def clean_data(self):
        required_columns = ['stars', 'age_years', 'releases', 'loc', 
                          'cbo_mean', 'dit_mean', 'lcom_mean']
        
        before_cleaning = len(self.df)
        self.df = self.df.dropna(subset=required_columns)
        after_cleaning = len(self.df)
        
        print(f"Dados limpos: {before_cleaning} -> {after_cleaning} repositórios")
        
        for col in ['cbo_mean', 'dit_mean', 'lcom_mean']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
    
    def descriptive_statistics(self):
        print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
        
        metrics_columns = ['stars', 'age_years', 'releases', 'loc', 'comments',
                          'cbo_mean', 'dit_mean', 'lcom_mean', 'total_classes']
        
        desc_stats = self.df[metrics_columns].describe()
        print(desc_stats.round(2))
        
        return desc_stats
    
    def analyze_rq01_popularity_quality(self):
        print("\n=== RQ01: POPULARIDADE vs QUALIDADE ===")
        
        self.df['popularity_category'] = pd.cut(self.df['stars'], 
                                              bins=[0, 1000, 5000, 20000, float('inf')],
                                              labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
        
        quality_by_popularity = self.df.groupby('popularity_category').agg({
            'cbo_mean': ['mean', 'median', 'std'],
            'dit_mean': ['mean', 'median', 'std'],
            'lcom_mean': ['mean', 'median', 'std']
        }).round(3)
        
        print("Qualidade por categoria de popularidade:")
        print(quality_by_popularity)
        
        correlations = {
            'Stars vs CBO': stats.spearmanr(self.df['stars'], self.df['cbo_mean']),
            'Stars vs DIT': stats.spearmanr(self.df['stars'], self.df['dit_mean']),
            'Stars vs LCOM': stats.spearmanr(self.df['stars'], self.df['lcom_mean'])
        }
        
        print("\nCorrelações de Spearman:")
        for metric, (correlation, p_value) in correlations.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{metric}: r={correlation:.3f}, p={p_value:.3f} {significance}")
        
        return quality_by_popularity, correlations
    
    def analyze_rq02_maturity_quality(self):
        print("\n=== RQ02: MATURIDADE vs QUALIDADE ===")
        
        self.df['maturity_category'] = pd.cut(self.df['age_years'],
                                            bins=[0, 2, 5, 10, float('inf')],
                                            labels=['Novo', 'Jovem', 'Maduro', 'Muito Maduro'])
        
        quality_by_maturity = self.df.groupby('maturity_category').agg({
            'cbo_mean': ['mean', 'median', 'std'],
            'dit_mean': ['mean', 'median', 'std'],
            'lcom_mean': ['mean', 'median', 'std']
        }).round(3)
        
        print("Qualidade por categoria de maturidade:")
        print(quality_by_maturity)
        
        correlations = {
            'Age vs CBO': stats.spearmanr(self.df['age_years'], self.df['cbo_mean']),
            'Age vs DIT': stats.spearmanr(self.df['age_years'], self.df['dit_mean']),
            'Age vs LCOM': stats.spearmanr(self.df['age_years'], self.df['lcom_mean'])
        }
        
        print("\nCorrelações de Spearman:")
        for metric, (correlation, p_value) in correlations.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{metric}: r={correlation:.3f}, p={p_value:.3f} {significance}")
        
        return quality_by_maturity, correlations
    
    def analyze_rq03_activity_quality(self):
        print("\n=== RQ03: ATIVIDADE vs QUALIDADE ===")
        
        self.df['activity_category'] = pd.cut(self.df['releases'],
                                            bins=[-1, 0, 5, 20, float('inf')],
                                            labels=['Sem Releases', 'Baixa', 'Média', 'Alta'])
        
        quality_by_activity = self.df.groupby('activity_category').agg({
            'cbo_mean': ['mean', 'median', 'std'],
            'dit_mean': ['mean', 'median', 'std'],
            'lcom_mean': ['mean', 'median', 'std']
        }).round(3)
        
        print("Qualidade por categoria de atividade:")
        print(quality_by_activity)
        
        correlations = {
            'Releases vs CBO': stats.spearmanr(self.df['releases'], self.df['cbo_mean']),
            'Releases vs DIT': stats.spearmanr(self.df['releases'], self.df['dit_mean']),
            'Releases vs LCOM': stats.spearmanr(self.df['releases'], self.df['lcom_mean'])
        }
        
        print("\nCorrelações de Spearman:")
        for metric, (correlation, p_value) in correlations.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{metric}: r={correlation:.3f}, p={p_value:.3f} {significance}")
        
        return quality_by_activity, correlations
    
    def analyze_rq04_size_quality(self):
        print("\n=== RQ04: TAMANHO vs QUALIDADE ===")
        
        self.df['size_category'] = pd.cut(self.df['loc'],
                                        bins=[0, 10000, 50000, 200000, float('inf')],
                                        labels=['Pequeno', 'Médio', 'Grande', 'Muito Grande'])
        
        quality_by_size = self.df.groupby('size_category').agg({
            'cbo_mean': ['mean', 'median', 'std'],
            'dit_mean': ['mean', 'median', 'std'],
            'lcom_mean': ['mean', 'median', 'std']
        }).round(3)
        
        print("Qualidade por categoria de tamanho:")
        print(quality_by_size)
        
        correlations = {
            'LOC vs CBO': stats.spearmanr(self.df['loc'], self.df['cbo_mean']),
            'LOC vs DIT': stats.spearmanr(self.df['loc'], self.df['dit_mean']),
            'LOC vs LCOM': stats.spearmanr(self.df['loc'], self.df['lcom_mean'])
        }
        
        print("\nCorrelações de Spearman:")
        for metric, (correlation, p_value) in correlations.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{metric}: r={correlation:.3f}, p={p_value:.3f} {significance}")
        
        return quality_by_size, correlations
    
    def create_correlation_plots(self):
        print("\n=== GERANDO GRÁFICOS DE CORRELAÇÃO ===")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Correlações entre Métricas de Processo e Qualidade', fontsize=16)
        
        axes[0,0].scatter(self.df['stars'], self.df['cbo_mean'], alpha=0.6)
        axes[0,0].set_xlabel('Estrelas')
        axes[0,0].set_ylabel('CBO Médio')
        axes[0,0].set_title('Popularidade vs CBO')
        axes[0,0].set_xscale('log')
        
        axes[0,1].scatter(self.df['age_years'], self.df['dit_mean'], alpha=0.6)
        axes[0,1].set_xlabel('Idade (anos)')
        axes[0,1].set_ylabel('DIT Médio')
        axes[0,1].set_title('Maturidade vs DIT')
        
        axes[0,2].scatter(self.df['releases'], self.df['lcom_mean'], alpha=0.6)
        axes[0,2].set_xlabel('Número de Releases')
        axes[0,2].set_ylabel('LCOM Médio')
        axes[0,2].set_title('Atividade vs LCOM')
        
        axes[1,0].scatter(self.df['loc'], self.df['cbo_mean'], alpha=0.6)
        axes[1,0].set_xlabel('LOC')
        axes[1,0].set_ylabel('CBO Médio')
        axes[1,0].set_title('Tamanho vs CBO')
        axes[1,0].set_xscale('log')
        
        correlation_matrix = self.df[['stars', 'age_years', 'releases', 'loc', 
                                    'cbo_mean', 'dit_mean', 'lcom_mean']].corr()
        
        im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1,1].set_xticks(range(len(correlation_matrix.columns)))
        axes[1,1].set_yticks(range(len(correlation_matrix.columns)))
        axes[1,1].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[1,1].set_yticklabels(correlation_matrix.columns)
        axes[1,1].set_title('Matriz de Correlação')
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1,1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)
        
        quality_metrics = [self.df['cbo_mean'], self.df['dit_mean'], self.df['lcom_mean']]
        axes[1,2].boxplot(quality_metrics, labels=['CBO', 'DIT', 'LCOM'])
        axes[1,2].set_ylabel('Valores')
        axes[1,2].set_title('Distribuição das Métricas de Qualidade')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_full_report(self):
        print("=== RELATÓRIO COMPLETO DE ANÁLISE ===")
        
        desc_stats = self.descriptive_statistics()
        
        rq01_results = self.analyze_rq01_popularity_quality()
        rq02_results = self.analyze_rq02_maturity_quality()
        rq03_results = self.analyze_rq03_activity_quality()
        rq04_results = self.analyze_rq04_size_quality()
        
        self.create_correlation_plots()
        
        return {
            'descriptive_stats': desc_stats,
            'rq01': rq01_results,
            'rq02': rq02_results,
            'rq03': rq03_results,
            'rq04': rq04_results
        }

def main():
    analyzer = JavaMetricsAnalyzer()
    
    if analyzer.df is not None:
        results = analyzer.generate_full_report()
        print("\nAnálise completa finalizada!")
        print("Gráfico salvo como 'correlation_analysis.png'")
    else:
        print("Não foi possível carregar os dados. Verifique se o arquivo de resultados existe.")

if __name__ == "__main__":
    main()