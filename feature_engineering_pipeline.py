import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure logging and styling
plt.style.use('seaborn-v0_8')
sns.set_palette('colorblind')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors) 
class AdvancedFootballFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates sophisticated football metrics for prospect identification"""
    
    def __init__(self, fit_mode=True):
        self.position_weights = {
            'GK': {'goalkeeping': 0.7, 'physical': 0.2, 'mental': 0.1},
            'CB': {'defending': 0.5, 'physical': 0.3, 'mental': 0.2},
            'FB': {'defending': 0.4, 'physical': 0.3, 'attacking': 0.3},
            'CM': {'passing': 0.4, 'technical': 0.3, 'physical': 0.3},
            'AM': {'technical': 0.5, 'attacking': 0.3, 'mental': 0.2},
            'WG': {'technical': 0.4, 'attacking': 0.4, 'physical': 0.2},
            'ST': {'attacking': 0.6, 'physical': 0.2, 'technical': 0.2}
        }
        self.fit_mode = fit_mode
        
    def fit(self, X, y=None):
        # Can store any necessary statistics here if needed
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Position-Specific Scores
        X = self._create_position_scores(X)
        
        # 2. Advanced Composite Metrics
        X = self._create_composite_metrics(X)
        
        # 3. Development Potential Metrics
        X = self._create_development_metrics(X)
        
        # 4. Physical Profile Metrics
        X = self._create_physical_metrics(X)
        
        # 5. Market Value Metrics
        X = self._create_market_metrics(X)
        
        return X
    
    def _create_position_scores(self, X):
        """Create position-specific performance scores"""
        # Goalkeeper specific
        if all(col in X.columns for col in ['goalkeeping_diving', 'goalkeeping_handling']):
            X['goalkeeping_score'] = (
                X['goalkeeping_diving'] * 0.25 +
                X['goalkeeping_handling'] * 0.25 +
                X['goalkeeping_positioning'] * 0.2 +
                X['goalkeeping_reflexes'] * 0.2 +
                X['goalkeeping_kicking'] * 0.1
            )
        
        # Defender specific
        X['defensive_score'] = (
            X['defending_marking_awareness'] * 0.4 +
            X['defending_standing_tackle'] * 0.3 +
            X['defending_sliding_tackle'] * 0.2 +
            X['mentality_interceptions'] * 0.1
        )
        
        # Midfielder specific
        X['playmaking_score'] = (
            X['passing'] * 0.3 +
            X['mentality_vision'] * 0.25 +
            X['skill_ball_control'] * 0.2 +
            X['mentality_composure'] * 0.15 +
            X['skill_long_passing'] * 0.1
        )
        
        # Forward specific
        X['attacking_score'] = (
            X['attacking_finishing'] * 0.35 +
            X['shooting'] * 0.25 +
            X['movement_reactions'] * 0.2 +
            X['power_shot_power'] * 0.1 +
            X['attacking_heading_accuracy'] * 0.1
        )
        
        return X
    
    def _create_composite_metrics(self, X):
        """Create advanced composite metrics with missing column handling."""
    
        X = X.copy()
    
        # Helper function to fill missing columns with zero
        def ensure_columns_exist(df, cols):
            for col in cols:
                if col not in df.columns:
                    df[col] = 0
    
        # Physical composite columns and weights
        physical_cols = [
            'power_strength',
            'power_stamina',
            'movement_sprint_speed',
            'movement_acceleration',
            'movement_agility'
        ]
        ensure_columns_exist(X, physical_cols)
        X['physical_composite'] = (
            X['power_strength'] * 0.3 +
            X['power_stamina'] * 0.25 +
            X['movement_sprint_speed'] * 0.2 +
            X['movement_acceleration'] * 0.15 +
            X['movement_agility'] * 0.1
        )
    
        # Technical composite columns and weights
        technical_cols = [
            'skill_ball_control',
            'skill_dribbling',
            'attacking_short_passing',
            'skill_curve',
            'skill_fk_accuracy'
        ]
        ensure_columns_exist(X, technical_cols)
        X['technical_composite'] = (
            X['skill_ball_control'] * 0.25 +
            X['skill_dribbling'] * 0.25 +
            X['attacking_short_passing'] * 0.2 +
            X['skill_curve'] * 0.15 +
            X['skill_fk_accuracy'] * 0.15
        )
    
        # Mental composite columns and weights
        mental_cols = [
            'mentality_composure',
            'mentality_vision',
            'mentality_positioning',
            'mentality_interceptions',
            'mentality_penalties'
        ]
        ensure_columns_exist(X, mental_cols)
        X['mental_composite'] = (
            X['mentality_composure'] * 0.25 +
            X['mentality_vision'] * 0.25 +
            X['mentality_positioning'] * 0.2 +
            X['mentality_interceptions'] * 0.15 +
            X['mentality_penalties'] * 0.15
        )
    
        # Versatility score: sum of defender, midfielder, forward flags if all exist
        position_cols = ['is_defender', 'is_midfielder', 'is_forward']
        if all(col in X.columns for col in position_cols):
            X['versatility_score'] = X[position_cols].sum(axis=1)
        else:
            # If missing any, create versatility_score filled with zeros
            X['versatility_score'] = 0
    
        return X

    
    def _create_development_metrics(self, X):
        """Metrics related to player development potential"""
        if 'potential' in X.columns and 'overall' in X.columns:
            # Calculate potential growth (difference between potential and current overall)
            X['potential_growth'] = X['potential'] - X['overall']
            
            # Age-adjusted potential
            if 'age' in X.columns:
                X['age_adjusted_potential'] = X['potential'] * (1 - 0.02 * (X['age'] - 17))
                
                # Development curve score
                X['dev_curve_score'] = X['potential_growth'] / (X['age'] + 1)
                
                # Skill progression potential
                if 'weak_foot' in X.columns and 'skill_moves' in X.columns:
                    X['skill_progression'] = (
                        X['potential_growth'] * 
                        (X['weak_foot'] + X['skill_moves']) / 10
                    )
        
        return X
    
    def _create_physical_metrics(self, X):
        """Advanced physical metrics"""
        if 'weight_kg' in X.columns and 'height_cm' in X.columns:
            # Body mass index
            X['bmi'] = X['weight_kg'] / ((X['height_cm'] / 100) ** 2)
            
        if 'movement_sprint_speed' in X.columns and 'power_stamina' in X.columns:
            # Speed-endurance balance
            X['speed_endurance'] = X['movement_sprint_speed'] * 0.6 + X['power_stamina'] * 0.4
            
        if 'power_strength' in X.columns and 'power_jumping' in X.columns and 'power_long_shots' in X.columns:
            # Power index
            X['power_index'] = X['power_strength'] * 0.5 + X['power_jumping'] * 0.3 + X['power_long_shots'] * 0.2
            
        return X
    
    def _create_market_metrics(self, X):
        """Market value and economic metrics"""
        if 'overall' in X.columns and 'value_eur' in X.columns:
            # Value efficiency
            X['value_efficiency'] = X['overall'] / (X['value_eur'] / 1e6 + 1)
            
        if 'overall' in X.columns and 'wage_eur' in X.columns:
            # Wage-to-performance ratio
            X['wage_ratio'] = X['overall'] / (X['wage_eur'] / 1e3 + 1)
            
        if 'value_eur' in X.columns and 'potential' in X.columns and 'contract_year' in X.columns:
            # Contract value score
            X['contract_score'] = (
                (X['value_eur'] / 1e6) * 
                (2025 - X['contract_year']) * 
                (X['potential'] / 100)
            )
        
        return X

class ProspectIdentifier:
    """Advanced prospect identification with configurable thresholds and position-specific logic"""
    
    def __init__(self, 
                 min_potential_diff=5,
                 max_age=23,
                 min_skill_moves=2,
                 min_weak_foot=2,
                 min_physical=50,
                 min_technical=50,
                 min_mental=50,
                 min_skill_progression=10):
        self.min_potential_diff = min_potential_diff
        self.max_age = max_age
        self.min_skill_moves = min_skill_moves
        self.min_weak_foot = min_weak_foot
        self.min_physical = min_physical
        self.min_technical = min_technical
        self.min_mental = min_mental
        self.min_skill_progression = min_skill_progression

    def _position_specific_criteria(self, df):
        """Generate position-specific boolean masks for prospect identification"""
        # Fill missing position-specific scores with 0
        for col in ['goalkeeping_score', 'defensive_score', 'playmaking_score', 'attacking_score']:
            if col not in df.columns:
                df[col] = 0

        gk = (
            df['player_positions'].str.contains('GK') &
            (df['goalkeeping_score'] >= 70)
        )
        cb = (
            df['player_positions'].str.contains('CB') &
            (df['defensive_score'] >= 65) &
            (df['physical_composite'] >= self.min_physical) &
            (df['mental_composite'] >= self.min_mental)
        )
        fb = (
            df['player_positions'].str.contains('LB|RB|LWB|RWB') &
            (df['defensive_score'] >= 60) &
            (df['physical_composite'] >= self.min_physical) &
            (df['technical_composite'] >= self.min_technical)
        )
        mid = (
            df['player_positions'].str.contains('CM|CDM|CAM') &
            (df['playmaking_score'] >= 65) &
            (df['technical_composite'] >= self.min_technical) &
            (df['mental_composite'] >= self.min_mental)
        )
        wing = (
            df['player_positions'].str.contains('LW|RW|LM|RM') &
            (df['attacking_score'] >= 65) &
            (df['technical_composite'] >= self.min_technical) &
            (df['physical_composite'] >= self.min_physical)
        )
        st = (
            df['player_positions'].str.contains('ST|CF') &
            (df['attacking_score'] >= 70) &
            (df['physical_composite'] >= self.min_physical)
        )

        return gk, cb, fb, mid, wing, st

    def add_is_prospect(self, df):
        """
        Create 'is_prospect' column (1 if player meets prospect criteria, else 0).
        """
        required_cols = [
            'potential', 'overall', 'age', 'skill_moves', 'weak_foot',
            'physical_composite', 'technical_composite', 'mental_composite',
            'skill_progression', 'player_positions'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for prospect identification: {missing}")

        base = (
            (df['potential'] - df['overall'] >= self.min_potential_diff) &
            (df['age'] <= self.max_age)
        )
        skill = (
            (df['skill_moves'] >= self.min_skill_moves) &
            (df['weak_foot'] >= self.min_weak_foot) &
            (df['skill_progression'] >= self.min_skill_progression)
        )

        gk, cb, fb, mid, wing, st = self._position_specific_criteria(df)

        df['is_prospect'] = (
            base & skill & (gk | cb | fb | mid | wing | st)
        ).astype(int)

        return df

    def identify_prospects(self, df):
        """Main prospect identification logic"""
        return self.add_is_prospect(df)


class ProspectReportGenerator:
    """Generates comprehensive PDF reports for prospect analysis"""
    
    def __init__(self, data, prospect_col='is_prospect'):
        self.data = data
        self.prospect_col = prospect_col
        self.report_date = datetime.now().strftime('%Y-%m-%d')
        self.colors = {
            'prospect': '#2ecc71',
            'non_prospect': '#e74c3c',
            'background': '#f9f9f9',
            'text': '#333333'
        }
        
    def generate_full_report(self, output_path='prospect_report.pdf'):
        """Generate complete PDF report with all visualizations"""
        with PdfPages(output_path) as pdf:
            # Title Page
            self._create_title_page(pdf)
            
            # Summary Statistics
            self._create_summary_page(pdf)
            
            # Prospect Analysis
            self._create_prospect_analysis_page(pdf)
            
            # Position Analysis
            self._create_position_analysis_page(pdf)
            
            # Age Distribution
            self._create_age_analysis_page(pdf)
            
            # Attribute Comparison
            self._create_attribute_comparison_page(pdf)
            
            # Top Prospects
            self._create_top_prospects_page(pdf)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'Football Prospect Analysis Report'
            d['Author'] = 'scouting system'
            
        print(f"✅ Report generated at {output_path}")
        return output_path

    def _create_title_page(self, pdf):
        """Create the title/cover page"""
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        # Add title and subtitle
        plt.text(0.5, 0.7, 'FOOTBALL PROSPECT ANALYSIS', 
                ha='center', va='center', fontsize=24, weight='bold', color=self.colors['text'])
        plt.text(0.5, 0.6, 'Comprehensive Scouting Report', 
                ha='center', va='center', fontsize=16, color=self.colors['text'])
        
        # Add date and info
        plt.text(0.5, 0.4, f"Report Date: {self.report_date}", 
                ha='center', va='center', fontsize=12, color=self.colors['text'])
        plt.text(0.5, 0.35, f"Players Analyzed: {len(self.data):,}", 
                ha='center', va='center', fontsize=12, color=self.colors['text'])
        
        # Add footer
        plt.text(0.5, 0.1, 'Scouting System', 
                ha='center', va='center', fontsize=10, style='italic', color=self.colors['text'])
        
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_summary_page(self, pdf):
        """Create page with summary statistics"""
        plt.figure(figsize=(11, 8.5))
        plt.subplots_adjust(top=0.85)
        
        # Calculate summary stats
        total_players = len(self.data)
        prospects = self.data[self.prospect_col].sum()
        prospect_pct = (prospects / total_players) * 100
        avg_potential = self.data['potential'].mean()
        avg_age = self.data['age'].mean()
        
        # Create summary table
        cell_text = [
            ["Total Players", f"{total_players:,}"],
            ["Identified Prospects", f"{prospects:,} ({prospect_pct:.1f}%)"],
            ["Average Potential", f"{avg_potential:.1f}"],
            ["Average Age", f"{avg_age:.1f}"]
        ]
        
        # Add table
        plt.table(cellText=cell_text,
                 colLabels=["Metric", "Value"],
                 loc='center',
                 cellLoc='left',
                 colWidths=[0.3, 0.3])
        
        plt.axis('off')
        plt.title('Key Statistics Summary', fontsize=16, weight='bold', pad=20)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_prospect_analysis_page(self, pdf):
        """Create page with prospect analysis"""
        plt.figure(figsize=(11, 8.5))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # 1. Growth Potential Comparison
        if 'potential_growth' in self.data.columns:
            sns.boxplot(data=self.data, x=self.prospect_col, y='potential_growth',
                       palette=[self.colors['non_prospect'], self.colors['prospect']], 
                       ax=axes[0, 0])
            axes[0, 0].set_title('Growth Potential Comparison')
            axes[0, 0].set_xticklabels(['Non-Prospects', 'Prospects'])
            axes[0, 0].set_ylabel('Potential - Overall')
        
        # 2. Age vs Potential
        if 'age' in self.data.columns and 'potential' in self.data.columns:
            sns.scatterplot(data=self.data, x='age', y='potential', hue=self.prospect_col,
                           palette=[self.colors['non_prospect'], self.colors['prospect']], 
                           alpha=0.6, ax=axes[0, 1])
            axes[0, 1].set_title('Age vs Potential Rating')
            axes[0, 1].legend(title='Prospect', labels=['No', 'Yes'])
        
        # 3. Value Efficiency
        if 'value_efficiency' in self.data.columns:
            sns.violinplot(data=self.data, x=self.prospect_col, y='value_efficiency',
                          palette=[self.colors['non_prospect'], self.colors['prospect']], 
                          ax=axes[1, 0])
            axes[1, 0].set_title('Value Efficiency Comparison')
            axes[1, 0].set_xticklabels(['Non-Prospects', 'Prospects'])
            axes[1, 0].set_ylabel('Overall / Value (millions)')
        
        # 4. Skill Moves Distribution
        if 'skill_moves' in self.data.columns:
            # Get all possible skill move values (1-5 typically)
            all_skills = sorted(self.data['skill_moves'].unique())
            
            # Get counts for both groups, ensuring same index
            prospect_skills = (self.data[self.data[self.prospect_col] == 1]['skill_moves']
                              .value_counts()
                              .reindex(all_skills, fill_value=0)
                              .sort_index())
            
            non_prospect_skills = (self.data[self.data[self.prospect_col] == 0]['skill_moves']
                                  .value_counts()
                                  .reindex(all_skills, fill_value=0)
                                  .sort_index())
            
            # Plotting
            width = 0.35
            x = np.arange(len(all_skills))
            axes[1, 1].bar(x - width/2, non_prospect_skills, width, 
                          label='Non-Prospects', color=self.colors['non_prospect'])
            axes[1, 1].bar(x + width/2, prospect_skills, width, 
                          label='Prospects', color=self.colors['prospect'])
            
            axes[1, 1].set_title('Skill Moves Distribution')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(all_skills)
            axes[1, 1].set_xlabel('Skill Moves Rating')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
        
        plt.suptitle('Prospect Attribute Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_position_analysis_page(self, pdf):
        """Create page with position analysis"""
        plt.figure(figsize=(11, 8.5))
        
        # Create position groups
        position_data = self.data.copy()
        position_data['position_group'] = np.select(
            [
                position_data['player_positions'].str.contains('GK'),
                position_data['player_positions'].str.contains('CB|RB|LB|WB|SW'),
                position_data['player_positions'].str.contains('CM|CDM|CAM|RM|LM'),
                position_data['player_positions'].str.contains('ST|CF|LW|RW')
            ],
            ['Goalkeeper', 'Defender', 'Midfielder', 'Forward'],
            default='Other'
        )
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Position Distribution
        position_counts = position_data['position_group'].value_counts()
        axes[0, 0].pie(position_counts, labels=position_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Position Distribution')
        
        # Prospect Position Distribution
        prospect_positions = position_data[position_data[self.prospect_col] == 1]['position_group'].value_counts()
        axes[0, 1].bar(prospect_positions.index, prospect_positions.values, 
                       color=self.colors['prospect'])
        axes[0, 1].set_title('Prospects by Position')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Count')
        
        # Position Potential Comparison
        sns.boxplot(data=position_data, x='position_group', y='potential', 
                   hue=self.prospect_col, 
                   palette=[self.colors['non_prospect'], self.colors['prospect']],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Potential by Position')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Potential Rating')
        axes[1, 0].legend(title='Prospect', labels=['No', 'Yes'])
        
        # Position Attributes Heatmap
        position_attrs = position_data.groupby('position_group')[
            ['physical_composite', 'technical_composite', 'mental_composite']
        ].mean()
        sns.heatmap(position_attrs, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1, 1])
        axes[1, 1].set_title('Attribute Averages by Position')
        
        plt.suptitle('Position Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_age_analysis_page(self, pdf):
        """Create page with age distribution analysis"""
        plt.figure(figsize=(11, 8.5))
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Age Distribution
        sns.histplot(data=self.data, x='age', hue=self.prospect_col, 
                    palette=[self.colors['non_prospect'], self.colors['prospect']],
                    bins=20, kde=True, multiple='stack', ax=axes[0])
        axes[0].set_title('Age Distribution')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Count')
        axes[0].legend(title='Prospect', labels=['No', 'Yes'])
        
        # Age vs Potential with Regression
        sns.regplot(data=self.data[self.data[self.prospect_col] == 1], 
                   x='age', y='potential', 
                   color=self.colors['prospect'], 
                   scatter_kws={'alpha':0.3},
                   label='Prospects', ax=axes[1])
        sns.regplot(data=self.data[self.data[self.prospect_col] == 0], 
                   x='age', y='potential', 
                   color=self.colors['non_prospect'], 
                   scatter_kws={'alpha':0.1},
                   label='Non-Prospects', ax=axes[1])
        axes[1].set_title('Age vs Potential with Trend Lines')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Potential Rating')
        axes[1].legend()
        
        plt.suptitle('Age Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_attribute_comparison_page(self, pdf):
        """Create page comparing key attributes"""
        plt.figure(figsize=(11, 8.5))
        
        # Select key attributes to compare
        attributes = ['physical_composite', 'technical_composite', 'mental_composite',
                     'defensive_score', 'playmaking_score', 'attacking_score']
        
        # Prepare data
        melted = self.data.melt(id_vars=[self.prospect_col], 
                              value_vars=attributes,
                              var_name='attribute', 
                              value_name='rating')
        
        # Create plot
        sns.boxplot(data=melted, x='attribute', y='rating', hue=self.prospect_col,
                   palette=[self.colors['non_prospect'], self.colors['prospect']])
        
        # Formatting
        plt.title('Attribute Comparison: Prospects vs Non-Prospects', fontsize=16, weight='bold')
        plt.xlabel('Attribute')
        plt.ylabel('Rating')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Prospect', labels=['No', 'Yes'])
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def _create_top_prospects_page(self, pdf):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Top Prospects', fontsize=16, weight='bold')
    
        # Use self.data and self.prospect_col here
        table_data = self.data[self.data[self.prospect_col] == 1].copy()
    
        if table_data.empty:
            ax.text(0.5, 0.5, "No top prospects found", ha='center', va='center', fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
            return
    
        table_data = table_data.sort_values(by='potential_growth', ascending=False).head(15)

        table_data = table_data[['short_name', 'age', 'club_name', 'overall', 'potential', 'potential_growth']]
    
        col_labels = ['Name', 'Age', 'Club', 'Overall', 'Potential', 'potential_growth']
        cell_text = [list(row) for _, row in table_data.iterrows()]
    
        table = plt.table(cellText=cell_text,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
    
        pdf.savefig(fig)
        plt.close(fig)


class FootballProspectPipeline:
    """Complete end-to-end prospect analysis pipeline with proper train-test handling"""
    
    def __init__(self, data_path, fit_mode=True, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.fit_mode = fit_mode
        self.test_size = test_size
        self.random_state = random_state
        self.cleaner = None
        self.engineer = None
        self.train_data = None
        self.test_data = None
        self.report_path = None
    
    def run_pipeline(self):
        """Execute complete pipeline with proper train-test handling"""
        try:
            # 1. Load data
            raw_data = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully")
            
            # 2. Split data if in fit mode
            if self.fit_mode:
                self.train_data, self.test_data = train_test_split(
                    raw_data, 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )
                print(f"✂️ Data split into train ({len(self.train_data)}) and test ({len(self.test_data)}) sets")
            else:
                self.train_data = raw_data
            
            # 3. Clean data
            self.cleaner = FootballDataCleaner(fit_mode=self.fit_mode)
            cleaned_train = self.cleaner.fit_transform(self.train_data)
            
            # 4. Feature engineering
            self.engineer = AdvancedFootballFeatureEngineer(fit_mode=self.fit_mode)
            engineered_train = self.engineer.fit_transform(cleaned_train)
            
            # 5. Prospect identification
            prospect_id = ProspectIdentifier()
            processed_train = prospect_id.identify_prospects(engineered_train)
            print(f"✅ Identified {processed_train['is_prospect'].sum()} prospects in training set")
            
            # 6. Generate report (only on training data)
            if self.fit_mode:
                report_gen = ProspectReportGenerator(processed_train)
                self.report_path = report_gen.generate_full_report()
                print("📊 Generated training data report")
                
                # Process test data using fitted transformers
                if self.test_data is not None:
                    cleaned_test = self.cleaner.transform(self.test_data)
                    engineered_test = self.engineer.transform(cleaned_test)
                    processed_test = prospect_id.identify_prospects(engineered_test)
                    print(f"🧪 Processed test set with {processed_test['is_prospect'].sum()} prospects")
                    
                    # Return both train and test data when in fit mode
                    return processed_train, processed_test, self.report_path
            
            # Return only processed data when not in fit mode
            return processed_train, self.report_path
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise

