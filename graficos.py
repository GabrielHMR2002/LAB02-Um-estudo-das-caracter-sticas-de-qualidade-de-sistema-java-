import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePlotGenerator:
    def __init__(self, results_file=None):
        if results_file is None:
            possible_files = [
                "results/java_metrics_results.csv",
                "java_metrics_results.csv",
                "results/partial_results_10.csv",
                "partial_results_10.csv"
            ]
            
            self.results_file = None
            for file in possible_files:
                if Path(file).exists():
                    self.results_file = file
                    print(f"Usando arquivo: {file}")
                    break
            
            if self.results_file is None:
                print("Nenhum arquivo de dados encontrado!")
                print("Arquivos procurados:")
                for file in possible_files:
                    print(f"  - {file}")
                print("\nExecute primeiro: python automate_metrics.py 10")
                exit(1)
        else:
            self.results_file = results_file
            
        self.df = None
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.results_file)
        
        required_columns = ['stars', 'age_years', 'releases', 'loc', 
                          'cbo_mean', 'dit_mean', 'lcom_mean']
        self.df = self.df.dropna(subset=required_columns)
        
        for col in ['cbo_mean', 'dit_mean', 'lcom_mean']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        self.df['popularity_category'] = pd.cut(self.df['stars'], 
                                              bins=[0, 1000, 5000, 20000, float('inf')],
                                              labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
        
        self.df['maturity_category'] = pd.cut(self.df['age_years'],
                                            bins=[0, 2, 5, 10, float('inf')],
                                            labels=['Novo', 'Jovem', 'Maduro', 'Muito Maduro'])
        
        self.df['activity_category'] = pd.cut(self.df['releases'],
                                            bins=[-1, 0, 5, 20, float('inf')],
                                            labels=['Sem Releases', 'Baixa', 'Média', 'Alta'])
        
        self.df['size_category'] = pd.cut(self.df['loc'],
                                        bins=[0, 10000, 50000, 200000, float('inf')],
                                        labels=['Pequeno', 'Médio', 'Grande', 'Muito Grande'])
    
    def plot_descriptive_statistics(self):
        plt.style.use('default')
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Estatísticas Descritivas das Métricas', fontsize=16, fontweight='bold')
        
        metrics = ['stars', 'age_years', 'releases', 'loc', 'cbo_mean', 'dit_mean', 'lcom_mean', 'comments']
        titles = ['Estrelas', 'Idade (anos)', 'Releases', 'LOC', 'CBO Médio', 'DIT Médio', 'LCOM Médio', 'Comentários']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 4
            col = i % 4
            
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                axes[row, col].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[row, col].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {data.mean():.2f}')
                axes[row, col].axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {data.median():.2f}')
                axes[row, col].set_title(title, fontweight='bold')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'descriptive_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_boxplots_summary(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribuição das Métricas de Qualidade por Categorias', fontsize=16, fontweight='bold')
        
        quality_metrics = ['cbo_mean', 'dit_mean', 'lcom_mean']
        quality_labels = ['CBO', 'DIT', 'LCOM']
        categories = ['popularity_category', 'maturity_category', 'activity_category', 'size_category']
        category_titles = ['Popularidade', 'Maturidade', 'Atividade', 'Tamanho']
        
        for idx, (cat, title) in enumerate(zip(categories, category_titles)):
            row = idx // 2
            col = idx % 2
            
            data_to_plot = []
            labels = []
            for metric in quality_metrics:
                for category in self.df[cat].cat.categories:
                    subset = self.df[self.df[cat] == category][metric]
                    if len(subset) > 0:
                        data_to_plot.append(subset)
                        labels.append(f'{metric.split("_")[0].upper()}\n{category}')
            
            bp = axes[row, col].boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral'] * len(self.df[cat].cat.categories)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[row, col].set_title(f'Qualidade por {title}', fontweight='bold')
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'boxplots_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 10))
        
        correlation_cols = ['stars', 'age_years', 'releases', 'loc', 'comments', 
                          'cbo_mean', 'dit_mean', 'lcom_mean', 'total_classes']
        
        available_cols = [col for col in correlation_cols if col in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8},
                   mask=mask)
        
        plt.title('Matriz de Correlação entre Métricas', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scatter_correlations(self):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Correlações entre Métricas de Processo e Qualidade', fontsize=16, fontweight='bold')
        
        process_metrics = ['stars', 'age_years', 'releases', 'loc']
        quality_metrics = ['cbo_mean', 'dit_mean', 'lcom_mean']
        process_labels = ['Estrelas', 'Idade (anos)', 'Releases', 'LOC']
        quality_labels = ['CBO Médio', 'DIT Médio', 'LCOM Médio']
        
        for i, (q_metric, q_label) in enumerate(zip(quality_metrics, quality_labels)):
            for j, (p_metric, p_label) in enumerate(zip(process_metrics, process_labels)):
                ax = axes[i, j]
                
                x = self.df[p_metric]
                y = self.df[q_metric]
                
                ax.scatter(x, y, alpha=0.6, s=30, color='steelblue')
                
                correlation, p_value = stats.spearmanr(x, y)
                
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(p_label, fontweight='bold')
                ax.set_ylabel(q_label, fontweight='bold')
                ax.set_title(f'{p_label} vs {q_label}\nr={correlation:.3f}, p={p_value:.3f}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if p_metric in ['stars', 'loc']:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_by_categories(self):
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle('Métricas de Qualidade por Categorias de Processo', fontsize=16, fontweight='bold')
        
        categories = ['popularity_category', 'maturity_category', 'activity_category', 'size_category']
        category_titles = ['Popularidade', 'Maturidade', 'Atividade', 'Tamanho']
        quality_metrics = ['cbo_mean', 'dit_mean', 'lcom_mean']
        quality_labels = ['CBO Médio', 'DIT Médio', 'LCOM Médio']
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        for i, (cat, cat_title) in enumerate(zip(categories, category_titles)):
            for j, (q_metric, q_label) in enumerate(zip(quality_metrics, quality_labels)):
                ax = axes[i, j]
                
                category_data = []
                category_labels = []
                category_colors = []
                
                for k, category in enumerate(self.df[cat].cat.categories):
                    subset = self.df[self.df[cat] == category][q_metric]
                    if len(subset) > 0:
                        category_data.append(subset.mean())
                        category_labels.append(category)
                        category_colors.append(colors[k % len(colors)])
                
                bars = ax.bar(category_labels, category_data, color=category_colors, edgecolor='black', alpha=0.8)
                
                for bar, value in zip(bars, category_data):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_data)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_title(f'{q_label} por {cat_title}', fontweight='bold')
                ax.set_ylabel(q_label, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_by_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distribution_comparison(self):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Distribuição das Métricas - Histograma vs Box Plot', fontsize=16, fontweight='bold')
        
        metrics = ['stars', 'age_years', 'releases', 'loc']
        titles = ['Estrelas', 'Idade (anos)', 'Releases', 'LOC']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            data = self.df[metric].dropna()
            
            axes[0, i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].set_title(f'Histograma - {title}', fontweight='bold')
            axes[0, i].set_ylabel('Frequência', fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)
            
            axes[1, i].boxplot(data, patch_artist=True, 
                              boxprops=dict(facecolor='lightgreen', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
            axes[1, i].set_title(f'Box Plot - {title}', fontweight='bold')
            axes[1, i].set_ylabel('Valores', fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_metrics_overview(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Visão Geral das Métricas de Qualidade', fontsize=16, fontweight='bold')
        
        quality_metrics = ['cbo_mean', 'dit_mean', 'lcom_mean']
        quality_labels = ['CBO', 'DIT', 'LCOM']
        
        quality_data = [self.df[metric].dropna() for metric in quality_metrics]
        
        axes[0, 0].boxplot(quality_data, labels=quality_labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 0].set_title('Distribuição das Métricas de Qualidade', fontweight='bold')
        axes[0, 0].set_ylabel('Valores', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        for i, (metric, label) in enumerate(zip(quality_metrics, quality_labels)):
            data = self.df[metric].dropna()
            axes[0, 1].hist(data, bins=30, alpha=0.6, label=label, edgecolor='black')
        axes[0, 1].set_title('Histograma das Métricas de Qualidade', fontweight='bold')
        axes[0, 1].set_xlabel('Valores', fontweight='bold')
        axes[0, 1].set_ylabel('Frequência', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        quality_corr = self.df[quality_metrics].corr()
        im = axes[1, 0].imshow(quality_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(quality_labels)))
        axes[1, 0].set_yticks(range(len(quality_labels)))
        axes[1, 0].set_xticklabels(quality_labels)
        axes[1, 0].set_yticklabels(quality_labels)
        axes[1, 0].set_title('Correlação entre Métricas de Qualidade', fontweight='bold')
        
        for i in range(len(quality_labels)):
            for j in range(len(quality_labels)):
                text = axes[1, 0].text(j, i, f'{quality_corr.iloc[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        means = [self.df[metric].mean() for metric in quality_metrics]
        bars = axes[1, 1].bar(quality_labels, means, color=['lightcoral', 'lightgreen', 'lightblue'], 
                             edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('Média das Métricas de Qualidade', fontweight='bold')
        axes[1, 1].set_ylabel('Valor Médio', fontweight='bold')
        
        for bar, value in zip(bars, means):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_summary(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resumo Estatístico das Correlações', fontsize=16, fontweight='bold')
        
        process_metrics = ['stars', 'age_years', 'releases', 'loc']
        quality_metrics = ['cbo_mean', 'dit_mean', 'lcom_mean']
        
        correlations = []
        p_values = []
        labels = []
        
        for p_metric in process_metrics:
            for q_metric in quality_metrics:
                corr, p_val = stats.spearmanr(self.df[p_metric], self.df[q_metric])
                correlations.append(corr)
                p_values.append(p_val)
                labels.append(f'{p_metric.replace("_", " ").title()}\nvs\n{q_metric.split("_")[0].upper()}')
        
        colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'lightgray' 
                 for p in p_values]
        
        bars = axes[0, 0].bar(range(len(correlations)), correlations, color=colors, edgecolor='black')
        axes[0, 0].set_title('Correlações de Spearman', fontweight='bold')
        axes[0, 0].set_ylabel('Coeficiente de Correlação', fontweight='bold')
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        significance_levels = ['p<0.001', 'p<0.01', 'p<0.05', 'p≥0.05']
        significance_colors = ['red', 'orange', 'yellow', 'lightgray']
        significance_counts = [sum(1 for p in p_values if p < 0.001),
                              sum(1 for p in p_values if 0.001 <= p < 0.01),
                              sum(1 for p in p_values if 0.01 <= p < 0.05),
                              sum(1 for p in p_values if p >= 0.05)]
        
        axes[0, 1].pie(significance_counts, labels=significance_levels, colors=significance_colors,
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Distribuição de Significância Estatística', fontweight='bold')
        
        log_p_values = [-np.log10(p) for p in p_values]
        axes[1, 0].bar(range(len(log_p_values)), log_p_values, color=colors, edgecolor='black')
        axes[1, 0].set_title('-log10(p-value)', fontweight='bold')
        axes[1, 0].set_ylabel('-log10(p-value)', fontweight='bold')
        axes[1, 0].set_xticks(range(len(labels)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1, 0].axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        axes[1, 0].axhline(y=-np.log10(0.001), color='yellow', linestyle='--', label='p=0.001')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        abs_correlations = [abs(c) for c in correlations]
        axes[1, 1].scatter(abs_correlations, log_p_values, c=colors, s=100, edgecolors='black')
        axes[1, 1].set_xlabel('|Correlação|', fontweight='bold')
        axes[1, 1].set_ylabel('-log10(p-value)', fontweight='bold')
        axes[1, 1].set_title('Força vs Significância das Correlações', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self):
        print("Gerando gráficos de estatísticas descritivas...")
        self.plot_descriptive_statistics()
        
        print("Gerando box plots por categoria...")
        self.plot_boxplots_summary()
        
        print("Gerando matriz de correlação...")
        self.plot_correlation_matrix()
        
        print("Gerando gráficos de dispersão...")
        self.plot_scatter_correlations()
        
        print("Gerando métricas de qualidade por categoria...")
        self.plot_quality_by_categories()
        
        print("Gerando comparação de distribuições...")
        self.plot_distribution_comparison()
        
        print("Gerando visão geral das métricas de qualidade...")
        self.plot_quality_metrics_overview()
        
        print("Gerando resumo estatístico...")
        self.plot_statistical_summary()
        
        print(f"\nTodos os gráficos foram salvos em: {self.output_dir}")
        print("Arquivos gerados:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")

def main():
    generator = ComprehensivePlotGenerator()
    generator.generate_all_plots()
    
    print("\n" + "="*50)
    print("GERAÇÃO DE GRÁFICOS CONCLUÍDA!")
    print("="*50)

if __name__ == "__main__":
    main()