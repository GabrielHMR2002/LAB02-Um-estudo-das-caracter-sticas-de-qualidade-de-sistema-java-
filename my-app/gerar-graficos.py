

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveVisualizer:
    def generate_comprehensive_mock_data(self, n_repos=1000):
        """
        Gera dados simulados realistas para 1000 repositórios Java
        """
        np.random.seed(42)  # Para reprodutibilidade
        
        # Gerar dados base dos repositórios
        repo_names = [f"repo_{i:04d}" for i in range(n_repos)]
        
        # Distribuições baseadas em dados reais do GitHub
        stars = np.random.lognormal(mean=7, sigma=2, size=n_repos).astype(int)
        stars = np.clip(stars, 100, 200000)  # Limitar valores extremos
        
        age_years = np.random.gamma(shape=2, scale=3, size=n_repos)
        age_years = np.clip(age_years, 0.5, 15)
        
        releases = np.random.negative_binomial(n=5, p=0.1, size=n_repos)
        releases = np.clip(releases, 0, 500)
        
        loc_total = np.random.lognormal(mean=10, sigma=1.5, size=n_repos).astype(int)
        loc_total = np.clip(loc_total, 1000, 5000000)
        
        loc_comments = (loc_total * np.random.beta(a=2, b=5, size=n_repos)).astype(int)
        
        # Métricas de qualidade com correlações realistas
        # CBO: correlacionado positivamente com tamanho, negativamente com popularidade
        cbo_base = 1 + np.log(loc_total) * 0.3 - np.log(stars) * 0.1
        cbo_noise = np.random.normal(0, 1, n_repos)
        cbo_mean = np.maximum(0.5, cbo_base + cbo_noise)
        
        # DIT: menos correlacionado, mais constante
        dit_mean = 2 + np.random.gamma(shape=1.5, scale=1, size=n_repos)
        dit_mean = np.clip(dit_mean, 1, 10)
        
        # LCOM: correlacionado com tamanho e inversamente com atividade
        lcom_base = np.log(loc_total) * 3 - np.log(releases + 1) * 2
        lcom_noise = np.random.normal(0, 5, n_repos)
        lcom_mean = np.maximum(1, lcom_base + lcom_noise)
        
        # Criar DataFrame
        data = pd.DataFrame({
            'repository': repo_names,
            'stars': stars,
            'age_years': age_years,
            'releases': releases,
            'loc_total': loc_total,
            'loc_comments': loc_comments,
            'cbo_mean': cbo_mean,
            'dit_mean': dit_mean,
            'lcom_mean': lcom_mean,
            'forks': (stars * np.random.beta(2, 8)).astype(int),
            'open_issues': np.random.poisson(lam=20, size=n_repos)
        })
        
        # Adicionar métricas derivadas
        data['comment_ratio'] = data['loc_comments'] / data['loc_total']
        data['popularity_log'] = np.log10(data['stars'])
        data['size_log'] = np.log10(data['loc_total'])
        
        print(f"Dados simulados gerados para {n_repos} repositórios")
        return data
    
    def calculate_correlations(self):
        """
        Calcula todas as correlações para as 4 RQs
        """
        correlations = {}
        
        # RQ01: Popularidade vs Qualidade
        correlations['RQ01'] = {
            'stars_cbo': stats.spearmanr(self.data['stars'], self.data['cbo_mean']),
            'stars_dit': stats.spearmanr(self.data['stars'], self.data['dit_mean']),
            'stars_lcom': stats.spearmanr(self.data['stars'], self.data['lcom_mean'])
        }
        
        # RQ02: Maturidade vs Qualidade
        correlations['RQ02'] = {
            'age_cbo': stats.spearmanr(self.data['age_years'], self.data['cbo_mean']),
            'age_dit': stats.spearmanr(self.data['age_years'], self.data['dit_mean']),
            'age_lcom': stats.spearmanr(self.data['age_years'], self.data['lcom_mean'])
        }
        
        # RQ03: Atividade vs Qualidade
        correlations['RQ03'] = {
            'releases_cbo': stats.spearmanr(self.data['releases'], self.data['cbo_mean']),
            'releases_dit': stats.spearmanr(self.data['releases'], self.data['dit_mean']),
            'releases_lcom': stats.spearmanr(self.data['releases'], self.data['lcom_mean'])
        }
        
        # RQ04: Tamanho vs Qualidade
        correlations['RQ04'] = {
            'loc_cbo': stats.spearmanr(self.data['loc_total'], self.data['cbo_mean']),
            'loc_dit': stats.spearmanr(self.data['loc_total'], self.data['dit_mean']),
            'loc_lcom': stats.spearmanr(self.data['loc_total'], self.data['lcom_mean'])
        }
        
        return correlations
    
    def plot_rq01_popularity_quality(self):
        """
        RQ01: Gráficos de Popularidade vs Qualidade
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RQ01: Relação entre Popularidade e Qualidade de Código', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Stars vs CBO
        ax1 = axes[0, 0]
        ax1.scatter(self.data['stars'], self.data['cbo_mean'], alpha=0.6, s=30, color='darkblue')
        ax1.set_xlabel('Número de Estrelas', fontsize=12)
        ax1.set_ylabel('CBO (Coupling Between Objects)', fontsize=12)
        ax1.set_title('Popularidade vs Acoplamento (CBO)', fontsize=14)
        ax1.set_xscale('log')
        
        # Linha de tendência
        z1 = np.polyfit(np.log(self.data['stars']), self.data['cbo_mean'], 1)
        p1 = np.poly1d(z1)
        x_trend = np.logspace(np.log10(self.data['stars'].min()), np.log10(self.data['stars'].max()), 100)
        ax1.plot(x_trend, p1(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        # Estatísticas
        corr_coef, p_val = stats.spearmanr(self.data['stars'], self.data['cbo_mean'])
        ax1.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 2: Stars vs DIT
        ax2 = axes[0, 1]
        ax2.scatter(self.data['stars'], self.data['dit_mean'], alpha=0.6, s=30, color='darkgreen')
        ax2.set_xlabel('Número de Estrelas', fontsize=12)
        ax2.set_ylabel('DIT (Depth Inheritance Tree)', fontsize=12)
        ax2.set_title('Popularidade vs Profundidade de Herança (DIT)', fontsize=14)
        ax2.set_xscale('log')
        
        z2 = np.polyfit(np.log(self.data['stars']), self.data['dit_mean'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_trend, p2(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['stars'], self.data['dit_mean'])
        ax2.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 3: Stars vs LCOM
        ax3 = axes[1, 0]
        ax3.scatter(self.data['stars'], self.data['lcom_mean'], alpha=0.6, s=30, color='darkred')
        ax3.set_xlabel('Número de Estrelas', fontsize=12)
        ax3.set_ylabel('LCOM (Lack of Cohesion)', fontsize=12)
        ax3.set_title('Popularidade vs Falta de Coesão (LCOM)', fontsize=14)
        ax3.set_xscale('log')
        
        z3 = np.polyfit(np.log(self.data['stars']), self.data['lcom_mean'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(x_trend, p3(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['stars'], self.data['lcom_mean'])
        ax3.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 4: Distribuição de Popularidade
        ax4 = axes[1, 1]
        ax4.hist(self.data['stars'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Número de Estrelas', fontsize=12)
        ax4.set_ylabel('Frequência', fontsize=12)
        ax4.set_title('Distribuição da Popularidade', fontsize=14)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rq01_popularity_quality_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rq02_maturity_quality(self):
        """
        RQ02: Gráficos de Maturidade vs Qualidade
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RQ02: Relação entre Maturidade e Qualidade de Código', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Age vs CBO
        ax1 = axes[0, 0]
        ax1.scatter(self.data['age_years'], self.data['cbo_mean'], alpha=0.6, s=30, color='orange')
        ax1.set_xlabel('Idade (anos)', fontsize=12)
        ax1.set_ylabel('CBO (Coupling Between Objects)', fontsize=12)
        ax1.set_title('Maturidade vs Acoplamento (CBO)', fontsize=14)
        
        z1 = np.polyfit(self.data['age_years'], self.data['cbo_mean'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(sorted(self.data['age_years']), p1(sorted(self.data['age_years'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['age_years'], self.data['cbo_mean'])
        ax1.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 2: Age vs DIT
        ax2 = axes[0, 1]
        ax2.scatter(self.data['age_years'], self.data['dit_mean'], alpha=0.6, s=30, color='brown')
        ax2.set_xlabel('Idade (anos)', fontsize=12)
        ax2.set_ylabel('DIT (Depth Inheritance Tree)', fontsize=12)
        ax2.set_title('Maturidade vs Profundidade de Herança (DIT)', fontsize=14)
        
        z2 = np.polyfit(self.data['age_years'], self.data['dit_mean'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(sorted(self.data['age_years']), p2(sorted(self.data['age_years'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['age_years'], self.data['dit_mean'])
        ax2.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 3: Age vs LCOM
        ax3 = axes[1, 0]
        ax3.scatter(self.data['age_years'], self.data['lcom_mean'], alpha=0.6, s=30, color='teal')
        ax3.set_xlabel('Idade (anos)', fontsize=12)
        ax3.set_ylabel('LCOM (Lack of Cohesion)', fontsize=12)
        ax3.set_title('Maturidade vs Falta de Coesão (LCOM)', fontsize=14)
        
        z3 = np.polyfit(self.data['age_years'], self.data['lcom_mean'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(sorted(self.data['age_years']), p3(sorted(self.data['age_years'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['age_years'], self.data['lcom_mean'])
        ax3.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 4: Boxplot por faixas de idade
        ax4 = axes[1, 1]
        age_bins = pd.cut(self.data['age_years'], bins=5, labels=['Muito Novo', 'Novo', 'Médio', 'Maduro', 'Muito Maduro'])
        data_boxplot = pd.DataFrame({
            'age_category': age_bins,
            'cbo': self.data['cbo_mean']
        })
        sns.boxplot(data=data_boxplot, x='age_category', y='cbo', ax=ax4)
        ax4.set_xlabel('Categoria de Idade', fontsize=12)
        ax4.set_ylabel('CBO Médio', fontsize=12)
        ax4.set_title('CBO por Faixa de Maturidade', fontsize=14)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rq02_maturity_quality_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rq03_activity_quality(self):
        """
        RQ03: Gráficos de Atividade vs Qualidade
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RQ03: Relação entre Atividade de Desenvolvimento e Qualidade', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Releases vs CBO
        ax1 = axes[0, 0]
        ax1.scatter(self.data['releases'], self.data['cbo_mean'], alpha=0.6, s=30, color='purple')
        ax1.set_xlabel('Número de Releases', fontsize=12)
        ax1.set_ylabel('CBO (Coupling Between Objects)', fontsize=12)
        ax1.set_title('Atividade vs Acoplamento (CBO)', fontsize=14)
        
        z1 = np.polyfit(self.data['releases'], self.data['cbo_mean'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(sorted(self.data['releases']), p1(sorted(self.data['releases'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['releases'], self.data['cbo_mean'])
        ax1.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 2: Releases vs DIT
        ax2 = axes[0, 1]
        ax2.scatter(self.data['releases'], self.data['dit_mean'], alpha=0.6, s=30, color='magenta')
        ax2.set_xlabel('Número de Releases', fontsize=12)
        ax2.set_ylabel('DIT (Depth Inheritance Tree)', fontsize=12)
        ax2.set_title('Atividade vs Profundidade de Herança (DIT)', fontsize=14)
        
        z2 = np.polyfit(self.data['releases'], self.data['dit_mean'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(sorted(self.data['releases']), p2(sorted(self.data['releases'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['releases'], self.data['dit_mean'])
        ax2.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 3: Releases vs LCOM
        ax3 = axes[1, 0]
        ax3.scatter(self.data['releases'], self.data['lcom_mean'], alpha=0.6, s=30, color='cyan')
        ax3.set_xlabel('Número de Releases', fontsize=12)
        ax3.set_ylabel('LCOM (Lack of Cohesion)', fontsize=12)
        ax3.set_title('Atividade vs Falta de Coesão (LCOM)', fontsize=14)
        
        z3 = np.polyfit(self.data['releases'], self.data['lcom_mean'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(sorted(self.data['releases']), p3(sorted(self.data['releases'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['releases'], self.data['lcom_mean'])
        ax3.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 4: Heatmap de atividade
        ax4 = axes[1, 1]
        activity_bins = pd.cut(self.data['releases'], bins=4, labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
        quality_bins = pd.cut(self.data['cbo_mean'], bins=4, labels=['Excelente', 'Boa', 'Regular', 'Ruim'])
        
        heatmap_data = pd.crosstab(activity_bins, quality_bins)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
        ax4.set_xlabel('Qualidade (CBO)', fontsize=12)
        ax4.set_ylabel('Atividade (Releases)', fontsize=12)
        ax4.set_title('Distribuição: Atividade vs Qualidade', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rq03_activity_quality_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rq04_size_quality(self):
        """
        RQ04: Gráficos de Tamanho vs Qualidade
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RQ04: Relação entre Tamanho do Código e Qualidade', fontsize=16, fontweight='bold')
        
        # Gráfico 1: LOC vs CBO
        ax1 = axes[0, 0]
        ax1.scatter(self.data['loc_total'], self.data['cbo_mean'], alpha=0.6, s=30, color='red')
        ax1.set_xlabel('Linhas de Código (LOC)', fontsize=12)
        ax1.set_ylabel('CBO (Coupling Between Objects)', fontsize=12)
        ax1.set_title('Tamanho vs Acoplamento (CBO)', fontsize=14)
        ax1.set_xscale('log')
        
        z1 = np.polyfit(np.log(self.data['loc_total']), self.data['cbo_mean'], 1)
        p1 = np.poly1d(z1)
        x_trend = np.logspace(np.log10(self.data['loc_total'].min()), np.log10(self.data['loc_total'].max()), 100)
        ax1.plot(x_trend, p1(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['loc_total'], self.data['cbo_mean'])
        ax1.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 2: LOC vs DIT
        ax2 = axes[0, 1]
        ax2.scatter(self.data['loc_total'], self.data['dit_mean'], alpha=0.6, s=30, color='navy')
        ax2.set_xlabel('Linhas de Código (LOC)', fontsize=12)
        ax2.set_ylabel('DIT (Depth Inheritance Tree)', fontsize=12)
        ax2.set_title('Tamanho vs Profundidade de Herança (DIT)', fontsize=14)
        ax2.set_xscale('log')
        
        z2 = np.polyfit(np.log(self.data['loc_total']), self.data['dit_mean'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_trend, p2(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['loc_total'], self.data['dit_mean'])
        ax2.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 3: LOC vs LCOM
        ax3 = axes[1, 0]
        ax3.scatter(self.data['loc_total'], self.data['lcom_mean'], alpha=0.6, s=30, color='darkgreen')
        ax3.set_xlabel('Linhas de Código (LOC)', fontsize=12)
        ax3.set_ylabel('LCOM (Lack of Cohesion)', fontsize=12)
        ax3.set_title('Tamanho vs Falta de Coesão (LCOM)', fontsize=14)
        ax3.set_xscale('log')
        
        z3 = np.polyfit(np.log(self.data['loc_total']), self.data['lcom_mean'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(x_trend, p3(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['loc_total'], self.data['lcom_mean'])
        ax3.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gráfico 4: Razão de comentários vs qualidade
        ax4 = axes[1, 1]
        ax4.scatter(self.data['comment_ratio'], self.data['cbo_mean'], alpha=0.6, s=30, color='gold')
        ax4.set_xlabel('Razão de Comentários (Comments/LOC)', fontsize=12)
        ax4.set_ylabel('CBO (Coupling Between Objects)', fontsize=12)
        ax4.set_title('Documentação vs Qualidade', fontsize=14)
        
        z4 = np.polyfit(self.data['comment_ratio'], self.data['cbo_mean'], 1)
        p4 = np.poly1d(z4)
        ax4.plot(sorted(self.data['comment_ratio']), p4(sorted(self.data['comment_ratio'])), "r--", alpha=0.8, linewidth=2)
        
        corr_coef, p_val = stats.spearmanr(self.data['comment_ratio'], self.data['cbo_mean'])
        ax4.text(0.05, 0.95, f'Spearman r = {corr_coef:.3f}\np-value = {p_val:.3e}', 
                transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rq04_size_quality_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """
        Gera todas as visualizações do projeto
        """
        print("Gerando visualizações abrangentes para todas as RQs...")
        
        print("1. RQ01 - Popularidade vs Qualidade...")
        self.plot_rq01_popularity_quality()
        
        print("2. RQ02 - Maturidade vs Qualidade...")
        self.plot_rq02_maturity_quality()
        
        print("3. RQ03 - Atividade vs Qualidade...")
        self.plot_rq03_activity_quality()
        
        print("4. RQ04 - Tamanho vs Qualidade...")
        self.plot_rq04_size_quality()
        
        print("5. Matriz de Correlação...")
        self.plot_correlation_matrix()
        
        print("6. Dashboard Resumo...")
        self.plot_summary_dashboard()
        
        print("7. Análises Avançadas...")
        self.plot_advanced_analysis()
        
        print(f"\nTodas as visualizações foram salvas em: {self.output_dir}/")
        
def main():
    """
    Função principal para executar toda a análise
    """
    print("SISTEMA COMPLETO DE VISUALIZAÇÃO - LABORATÓRIO 02")
    print("Análise de Qualidade de Repositórios Java")
    print("=" * 60)
    
    # Inicializar visualizador
    visualizer = ComprehensiveVisualizer()
    
    # Gerar todas as visualizações
    visualizer.generate_all_visualizations()
    
    # Gerar relatório estatístico
    visualizer.generate_statistical_report()
    
    # Salvar dados
    visualizer.save_data_to_csv()
    
    print("\nAnálise completa finalizada!")
    print(f"Verifique a pasta '{visualizer.output_dir}' para todos os arquivos gerados.")

if __name__ == "__main__":
    main()