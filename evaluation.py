"""
HRV Analysis for Music & Stress Study
Evaluates the influence of music on stress using Heart Rate Variability metrics
"""

# Updated HRVStressAnalyzer with ANOVA instead of Friedman
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class HRVStressAnalyzer:
    """Analyzes HRV data across different stress conditions"""
    
    def __init__(self, data_file=None):
        if data_file:
            self.df = pd.read_excel(data_file)
        else:
            self.df = self.create_sample_data()
        self.prepare_data()
        
    def create_sample_data(self):
        """Create sample data structure matching your format"""
        # This is example data - replace with your actual data loading
        participants = ['Leon', 'Anna', 'Max', 'Sophie', 'Tim', 'Laura']
        phases = ['baseline', 'stroop', 'post_rest']
        
        data = []
        np.random.seed(42)
        
        for i, participant in enumerate(participants):
            for phase in phases:
                # Simulate realistic HRV values
                if phase == 'baseline':
                    rmssd = np.random.normal(45, 8)
                    hr = np.random.normal(72, 5)
                    pnn50 = np.random.normal(25, 5)
                elif phase == 'stroop':  # Stress
                    rmssd = np.random.normal(30, 6)
                    hr = np.random.normal(85, 6)
                    pnn50 = np.random.normal(12, 4)
                else:  # post_rest
                    rmssd = np.random.normal(42, 7)
                    hr = np.random.normal(74, 5)
                    pnn50 = np.random.normal(22, 5)
                
                data.append({
                    'participant': participant,
                    'phase': phase,
                    'condition': 'real',
                    'RMSSD': rmssd,
                    'SDNN': rmssd * 1.3,
                    'pNN50': pnn50,
                    'mean_hr': hr,
                    'sd_hr': np.random.normal(5, 1)
                })
        
        return pd.DataFrame(data)
    
    def prepare_data(self):
        """Prepare data for analysis"""
        # Ensure phase order is correct
        phase_order = ['baseline', 'stroop', 'post_rest']
        self.df['phase'] = pd.Categorical(self.df['phase'], 
                                          categories=phase_order, 
                                          ordered=True)
        self.df = self.df.sort_values(['participant', 'phase'])
        
    def calculate_descriptive_stats(self):
        metrics = ['RMSSD', 'SDNN', 'pNN50', 'mean_hr']
        stats_df = self.df.groupby('phase')[metrics].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('se', lambda x: stats.sem(x)),
            ('median', 'median')
        ]).round(2)
        print("\nDescriptive Statistics:\n", stats_df)
        return stats_df
    
    def perform_statistical_tests(self):
        metrics = ['RMSSD', 'pNN50', 'mean_hr', 'SDNN']
        results = {}

        print("\n===== STATISTICAL TESTS (ANOVA + Wilcoxon) =====\n")

        for metric in metrics:
            print(f"\n📊 {metric}:")
            baseline = self.df[self.df['phase'] == 'baseline'][metric].values
            stroop = self.df[self.df['phase'] == 'stroop'][metric].values
            post_rest = self.df[self.df['phase'] == 'post_rest'][metric].values

            # One-way ANOVA
            f_stat, p_value = f_oneway(baseline, stroop, post_rest)
            print(f"ANOVA: F = {f_stat:.3f}, p = {p_value:.4f}")

            if p_value < 0.05:
                print("✓ Significant difference across phases")
                print("Post-hoc (Wilcoxon pairwise):")

                comparisons = [
                    ('baseline', 'stroop', baseline, stroop),
                    ('stroop', 'post_rest', stroop, post_rest),
                    ('baseline', 'post_rest', baseline, post_rest)
                ]

                for name1, name2, data1, data2 in comparisons:
                    stat_w, p_w = wilcoxon(data1, data2)
                    print(f"  {name1} vs {name2}: p = {p_w:.4f}")

            else:
                print("✗ No significant difference")

            results[metric] = {'anova_f': f_stat, 'anova_p': p_value}

        return results
    

    
    def calculate_percent_changes(self):
        """Calculate percentage changes from baseline"""
        changes = []
        
        for participant in self.df['participant'].unique():
            p_data = self.df[self.df['participant'] == participant]
            baseline_rmssd = p_data[p_data['phase'] == 'baseline']['RMSSD'].values[0]
            baseline_hr = p_data[p_data['phase'] == 'baseline']['mean_hr'].values[0]
            
            for phase in ['stroop', 'post_rest']:
                phase_data = p_data[p_data['phase'] == phase]
                rmssd_change = ((phase_data['RMSSD'].values[0] - baseline_rmssd) / baseline_rmssd) * 100
                hr_change = ((phase_data['mean_hr'].values[0] - baseline_hr) / baseline_hr) * 100
                
                changes.append({
                    'participant': participant,
                    'phase': phase,
                    'rmssd_change_pct': rmssd_change,
                    'hr_change_pct': hr_change
                })
        
        return pd.DataFrame(changes)
    
    def create_comprehensive_plots(self):
        """Create all visualization plots"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. RMSSD across phases (line plot with individual trajectories)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_individual_trajectories(ax1, 'RMSSD', 
                                          'RMSSD Across Phases\n(Parasympathetic Activity)')
        
        # 2. Mean HR across phases
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_individual_trajectories(ax2, 'mean_hr', 
                                          'Heart Rate Across Phases')
        
        # 3. pNN50 across phases
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_individual_trajectories(ax3, 'pNN50', 
                                          'pNN50 Across Phases')
        
        # 4. Box plots for RMSSD
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_boxplot(ax4, 'RMSSD', 'RMSSD Distribution by Phase')
        
        # 5. Box plots for Heart Rate
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_boxplot(ax5, 'mean_hr', 'Heart Rate Distribution by Phase')
        
        # 6. Percentage changes from baseline
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_percent_changes(ax6)
        
        # 7. Heatmap of all metrics
        ax7 = fig.add_subplot(gs[2, 0])
        self.plot_heatmap(ax7)
        
        # 8. Recovery analysis
        ax8 = fig.add_subplot(gs[2, 1])
        self.plot_recovery_analysis(ax8)
        
        # 9. Correlation between metrics
        ax9 = fig.add_subplot(gs[2, 2])
        self.plot_metric_correlations(ax9)
        
        plt.suptitle('HRV Analysis: Music & Stress Study', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig('hrv_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Comprehensive plot saved as 'hrv_comprehensive_analysis.png'")
        plt.show()
    
    def plot_individual_trajectories(self, ax, metric, title):
        """Plot individual participant trajectories"""
        for participant in self.df['participant'].unique():
            p_data = self.df[self.df['participant'] == participant]
            ax.plot(p_data['phase'], p_data[metric], 
                   marker='o', alpha=0.3, linewidth=1)
        
        # Add mean trajectory
        means = self.df.groupby('phase')[metric].mean()
        sems = self.df.groupby('phase')[metric].sem()
        ax.plot(means.index, means.values, 
               marker='o', linewidth=3, color='black', 
               label='Mean', markersize=10)
        ax.fill_between(range(len(means)), 
                        means - sems, means + sems, 
                        alpha=0.2, color='black')
        
        ax.set_xlabel('Phase')
        ax.set_ylabel(metric)
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_boxplot(self, ax, metric, title):
        """Create box plots for metric across phases"""
        sns.boxplot(data=self.df, x='phase', y=metric, ax=ax)
        sns.swarmplot(data=self.df, x='phase', y=metric, 
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Phase')
        ax.set_ylabel(metric)
    
    def plot_percent_changes(self, ax):
        """Plot percentage changes from baseline"""
        changes_df = self.calculate_percent_changes()
        
        x = np.arange(len(changes_df['phase'].unique()))
        width = 0.35
        
        rmssd_means = changes_df.groupby('phase')['rmssd_change_pct'].mean()
        hr_means = changes_df.groupby('phase')['hr_change_pct'].mean()
        rmssd_sems = changes_df.groupby('phase')['rmssd_change_pct'].sem()
        hr_sems = changes_df.groupby('phase')['hr_change_pct'].sem()
        
        ax.bar(x - width/2, rmssd_means, width, yerr=rmssd_sems, 
              label='RMSSD', alpha=0.8, capsize=5)
        ax.bar(x + width/2, hr_means, width, yerr=hr_sems, 
              label='Heart Rate', alpha=0.8, capsize=5)
        
        ax.set_ylabel('% Change from Baseline')
        ax.set_title('Percentage Changes from Baseline', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rmssd_means.index)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_heatmap(self, ax):
        """Create heatmap of normalized metrics"""
        metrics = ['RMSSD', 'SDNN', 'pNN50', 'mean_hr']
        pivot_data = self.df.groupby('phase')[metrics].mean()
        
        # Normalize data
        normalized = (pivot_data - pivot_data.mean()) / pivot_data.std()
        
        sns.heatmap(normalized.T, annot=True, fmt='.2f', 
                   cmap='RdYlGn', center=0, ax=ax,
                   cbar_kws={'label': 'Normalized Value'})
        ax.set_title('Normalized HRV Metrics Heatmap', fontweight='bold')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Metric')
    
    def plot_recovery_analysis(self, ax):
        """Analyze recovery from stress (stroop to post_rest)"""
        recovery_data = []
        
        for participant in self.df['participant'].unique():
            p_data = self.df[self.df['participant'] == participant]
            stroop_rmssd = p_data[p_data['phase'] == 'stroop']['RMSSD'].values[0]
            post_rmssd = p_data[p_data['phase'] == 'post_rest']['RMSSD'].values[0]
            baseline_rmssd = p_data[p_data['phase'] == 'baseline']['RMSSD'].values[0]
            
            # Recovery percentage
            recovery_pct = ((post_rmssd - stroop_rmssd) / 
                           (baseline_rmssd - stroop_rmssd)) * 100
            
            recovery_data.append({
                'participant': participant,
                'recovery_pct': recovery_pct,
                'stroop_rmssd': stroop_rmssd,
                'post_rmssd': post_rmssd
            })
        
        recovery_df = pd.DataFrame(recovery_data)
        
        ax.barh(recovery_df['participant'], recovery_df['recovery_pct'], 
               color='steelblue', alpha=0.7)
        ax.axvline(x=100, color='green', linestyle='--', 
                  linewidth=2, label='Full Recovery')
        ax.axvline(x=0, color='red', linestyle='--', 
                  linewidth=2, label='No Recovery')
        ax.set_xlabel('Recovery % (Stroop → Post-Rest)')
        ax.set_title('RMSSD Recovery Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    def plot_metric_correlations(self, ax):
        """Plot correlation between key metrics"""
        metrics = ['RMSSD', 'pNN50', 'mean_hr', 'SDNN']
        corr_matrix = self.df[metrics].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=1)
        ax.set_title('Metric Correlations', fontweight='bold')
    
    
    
    def export_results(self, filename='hrv_analysis_results.xlsx'):
        """Export all HRV analysis results to Excel including statistical tests."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            # -------------------------
            # Raw Data
            # -------------------------
            self.df.to_excel(writer, sheet_name='Raw Data', index=False)

            # -------------------------
            # Descriptive statistics
            # -------------------------
            stats_df = self.calculate_descriptive_stats()
            stats_df.to_excel(writer, sheet_name='Descriptive Stats')

            # -------------------------
            # Percentage changes
            # -------------------------
            changes_df = self.calculate_percent_changes()
            changes_df.to_excel(writer, sheet_name='Percent Changes', index=False)

            # -------------------------
            # Summary statistics
            # -------------------------
            summary = self.df.groupby('phase')[['RMSSD', 'pNN50', 'mean_hr', 'SDNN']].agg(
                ['mean', 'std', 'min', 'max']
            ).round(2)
            summary.to_excel(writer, sheet_name='Summary')

            # -------------------------
            # Statistical Tests (ANOVA + Wilcoxon)
            # -------------------------
            stat_results = self.perform_statistical_tests()

            # Convert dict → DataFrame for Excel
            stat_df = pd.DataFrame(stat_results).T
            stat_df.to_excel(writer, sheet_name='Statistical Tests')

            print(f"\n📁 Export completed → {filename}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*70)
        print("HRV STRESS STUDY ANALYSIS")
        print("="*70)
        print(f"Participants: {len(self.df['participant'].unique())}")
        print(f"Phases: {', '.join(self.df['phase'].unique())}")
        print(f"Total measurements: {len(self.df)}")
        
        # Descriptive statistics
        self.calculate_descriptive_stats()
        
        # Statistical tests
        self.perform_statistical_tests() 
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_comprehensive_plots()
        
        
        # Export results
        self.export_results()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  📊 hrv_comprehensive_analysis.png")
        print("  📑 hrv_analysis_results.xlsx")
        print("\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Option 1: Load your actual data
    #analyzer = HRVStressAnalyzer("C:/Users/sofia/Downloads/hrv_TechXPerience/test_data.xlsx")
    
    # Option 2: Use sample data (for demonstration)
    analyzer = HRVStressAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    