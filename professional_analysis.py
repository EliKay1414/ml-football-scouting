"""
Enhanced Professional EDA Analysis for Football Player Scouting System
with Data Quality Assessment and Modeling Readiness Checks
"""

# Standard library imports
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler
import warnings
import json

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# Set up configurations
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Visualization style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else plt.style.use('seaborn')
sns.set_palette("viridis")

# Constants
DEFAULT_REQUIRED_COLUMNS = [
    'age', 'overall', 'potential', 'value_eur', 'wage_eur', 
    'player_positions', 'preferred_foot', 'international_reputation',
    'potential_growth', 'physical_composite', 'technical_composite',
    'is_prospect'
]

class DataQualityReport:
    """Generates comprehensive data quality assessment"""
    
    @staticmethod
    def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics"""
        report = {
            'basic_stats': {},
            'data_quality': {},
            'recommendations': []
        }
        
        # Basic statistics
        report['basic_stats']['shape'] = df.shape
        report['basic_stats']['dtypes'] = df.dtypes.value_counts().to_dict()
        
        # Data quality metrics
        missing_values = df.isnull().sum()
        report['data_quality']['missing_values'] = {
            'total': missing_values.sum(),
            'columns': missing_values[missing_values > 0].sort_values(ascending=False).to_dict(),
            'pct_missing': (missing_values / len(df)).to_dict()
        }
        
        # Detect constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        report['data_quality']['constant_columns'] = constant_cols
        
        # Detect high cardinality categoricals
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = {}
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count > 50:  # Threshold for high cardinality
                high_cardinality[col] = unique_count
        report['data_quality']['high_cardinality'] = high_cardinality
        
        # Generate recommendations
        if len(constant_cols) > 0:
            report['recommendations'].append(f"Remove constant columns: {constant_cols}")
        
        if len(high_cardinality) > 0:
            report['recommendations'].append(
                f"Consider reducing cardinality for: {list(high_cardinality.keys())}"
            )
        
        return report

class FootballEDA:
    """Comprehensive EDA for football player scouting with modeling readiness checks"""
    
    def __init__(self, data_path: Path, output_dir: Path, required_columns: List[str] = None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.required_columns = required_columns or DEFAULT_REQUIRED_COLUMNS
        self.logger = self._setup_logger()
        self.df = self._load_and_validate_data()
        self.quality_report = DataQualityReport.generate_quality_report(self.df)
        
        # Create output directories
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging system"""
        logger = logging.getLogger('FootballEDA')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # File handler
            fh = RotatingFileHandler(
                self.output_dir / 'eda_analysis.log',
                maxBytes=1e6,
                backupCount=3
            )
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            
            logger.addHandler(ch)
            logger.addHandler(fh)
        
        return logger

    def _get_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the dataset and analysis"""
        return {
            'data_source': str(self.data_path),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_shape': self.df.shape,
            'target_variable': 'is_prospect' if 'is_prospect' in self.df.columns else None,
            'numerical_features': len(self.df.select_dtypes(include=np.number).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': int(self.df.isnull().sum().sum())
        }
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate input data with comprehensive NaN handling"""
        try:
            self.logger.info(f"Loading data from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Validate required columns
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            # Basic cleaning if needed
            if 'is_prospect' in df.columns:
                df['is_prospect'] = df['is_prospect'].astype(int)
            
            # Handle missing values - choose one strategy:
            
            # Strategy 1: Drop rows with missing values (if few missing)
            # df = df.dropna()
            
            # Strategy 2: Impute missing values (recommended)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif pd.api.types.is_string_dtype(df[col]):
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            # Log cleaning results
            missing_after = df.isnull().sum().sum()
            self.logger.info(f"Missing values after cleaning: {missing_after}")
            
            return df
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance with robust NaN handling"""
        results = {}
        
        if 'is_prospect' not in self.df.columns:
            return results
        
        try:
            # Prepare data with additional NaN check
            X = self.df.select_dtypes(include=np.number).drop(columns=['is_prospect'], errors='ignore')
            y = self.df['is_prospect']
            
            # Final safety check for NaN values
            if X.isnull().any().any() or y.isnull().any():
                self.logger.warning("NaN values detected in feature importance analysis - performing additional cleaning")
                X = X.fillna(X.median())
                y = y.fillna(y.mode()[0])
            
            # Correlation analysis
            corr = X.corrwith(y).sort_values(ascending=False)
            results['correlation'] = corr.to_dict()
            
            # Mutual information
            mi = mutual_info_classif(X, y, random_state=42)
            mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            results['mutual_information'] = mi_series.to_dict()
            
            # Plot top features
            plot_dir = self.output_dir / 'plots' / 'feature_importance'
            plot_dir.mkdir(exist_ok=True)
            
            for method, scores in [('Correlation', corr), ('Mutual_Info', mi_series)]:
                plt.figure(figsize=(10, 8))
                top_features = scores.head(15)
                sns.barplot(x=top_features.values, y=top_features.index)
                plt.title(f'Top 15 Features by {method} with Target')
                plot_path = plot_dir / f"top_features_{method.lower()}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                results[f"{method.lower()}_plot"] = plot_path
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {str(e)}")
            # Return partial results if available
            if 'correlation' in results:
                return results
            return {}
        
        return results

    def _convert_to_serializable(self, obj):
        """Convert pandas/numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif pd.api.types.is_float_dtype(obj):
            return float(obj) if not pd.isna(obj) else None
        elif pd.api.types.is_integer_dtype(obj):
            return int(obj) if not pd.isna(obj) else None
        elif pd.api.types.is_bool_dtype(obj):
            return bool(obj) if not pd.isna(obj) else None
        elif isinstance(obj, pd.Series):
            return self._convert_to_serializable(obj.to_dict())
        elif isinstance(obj, pd.DataFrame):
            return self._convert_to_serializable(obj.to_dict(orient='records'))
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'dtype') and 'Int64' in str(obj.dtype):
            return int(obj) if not pd.isna(obj) else None
        else:
            return str(obj)
    
    def analyze_data_distributions(self) -> Dict[str, Path]:
        """Analyze distributions of all features"""
        plots = {}
        plot_dir = self.output_dir / 'plots' / 'distributions'
        plot_dir.mkdir(exist_ok=True)
        
        # Numerical features
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols[:15]:  # Limit to 15 numerical features
            try:
                plt.figure(figsize=(12, 6))
                
                # Histogram with density
                ax1 = plt.subplot(1, 2, 1)
                sns.histplot(self.df[col], kde=True, ax=ax1)
                ax1.set_title(f'Distribution of {col}')
                
                # Boxplot
                ax2 = plt.subplot(1, 2, 2)
                sns.boxplot(y=self.df[col], ax=ax2)
                ax2.set_title(f'Boxplot of {col}')
                
                plt.tight_layout()
                plot_path = plot_dir / f"dist_{col}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                plots[f"num_{col}"] = plot_path
            except Exception as e:
                self.logger.warning(f"Failed to plot {col}: {str(e)}")
        
        # Categorical features
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols[:10]:  # Limit to 10 categorical features
            try:
                plt.figure(figsize=(12, 6))
                value_counts = self.df[col].value_counts().nlargest(20)
                
                # Bar plot
                ax1 = plt.subplot(1, 2, 1)
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax1)
                ax1.set_title(f'Top 20 {col} values')
                
                # Pie chart for top 5
                ax2 = plt.subplot(1, 2, 2)
                top5 = value_counts.nlargest(5)
                ax2.pie(top5, labels=top5.index, autopct='%1.1f%%')
                ax2.set_title(f'Top 5 {col} distribution')
                
                plt.tight_layout()
                plot_path = plot_dir / f"dist_{col}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                plots[f"cat_{col}"] = plot_path
            except Exception as e:
                self.logger.warning(f"Failed to plot {col}: {str(e)}")
        
        return plots
    
    def analyze_target_relationships(self) -> Dict[str, Path]:
        """Analyze relationships with target variable (is_prospect)"""
        plots = {}
        plot_dir = self.output_dir / 'plots' / 'target_analysis'
        plot_dir.mkdir(exist_ok=True)
        
        if 'is_prospect' not in self.df.columns:
            self.logger.warning("Target column 'is_prospect' not found")
            return plots
        
        try:
            # Target distribution
            plt.figure(figsize=(10, 6))
            target_dist = self.df['is_prospect'].value_counts(normalize=True)
            sns.barplot(x=target_dist.index, y=target_dist.values)
            plt.title(f'Target Distribution (Proportion: {target_dist[1]:.2%} prospects)')
            plot_path = plot_dir / "target_distribution.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plots["target_dist"] = plot_path
            
            # Numerical features vs target
            num_cols = [col for col in self.df.select_dtypes(include=np.number).columns 
                       if col != 'is_prospect'][:10]  # Limit to 10
            
            for col in num_cols:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Boxplot
                    ax1 = plt.subplot(1, 2, 1)
                    sns.boxplot(x='is_prospect', y=col, data=self.df, ax=ax1)
                    ax1.set_title(f'{col} by Prospect Status')
                    
                    # Density plot
                    ax2 = plt.subplot(1, 2, 2)
                    sns.kdeplot(data=self.df, x=col, hue='is_prospect', ax=ax2)
                    ax2.set_title(f'{col} Distribution by Prospect Status')
                    
                    plt.tight_layout()
                    plot_path = plot_dir / f"target_num_{col}.png"
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
                    plots[f"target_num_{col}"] = plot_path
                except Exception as e:
                    self.logger.warning(f"Failed to plot {col} vs target: {str(e)}")
            
            # Categorical features vs target
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()[:5]  # Limit
            
            for col in cat_cols:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Stacked bar plot
                    ax1 = plt.subplot(1, 2, 1)
                    cross_tab = pd.crosstab(self.df[col], self.df['is_prospect'], normalize='index')
                    cross_tab.plot(kind='bar', stacked=True, ax=ax1)
                    ax1.set_title(f'Prospect Proportion by {col}')
                    ax1.set_ylabel('Proportion')
                    
                    # Heatmap of normalized counts
                    ax2 = plt.subplot(1, 2, 2)
                    crosstab = pd.crosstab(self.df[col], self.df['is_prospect'])
                    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
                    ax2.set_title(f'Prospect Count by {col}')
                    
                    plt.tight_layout()
                    plot_path = plot_dir / f"target_cat_{col}.png"
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
                    plots[f"target_cat_{col}"] = plot_path
                except Exception as e:
                    self.logger.warning(f"Failed to plot {col} vs target: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Target analysis failed: {str(e)}")
        
        return plots
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance for modeling"""
        results = {}
        
        if 'is_prospect' not in self.df.columns:
            return results
        
        try:
            # Prepare data
            X = self.df.select_dtypes(include=np.number).drop(columns=['is_prospect'], errors='ignore')
            y = self.df['is_prospect']
            
            # Correlation analysis
            corr = X.corrwith(y).sort_values(ascending=False)
            results['correlation'] = corr.to_dict()
            
            # Mutual information
            mi = mutual_info_classif(X, y, random_state=42)
            mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            results['mutual_information'] = mi_series.to_dict()
            
            # Plot top features
            plot_dir = self.output_dir / 'plots' / 'feature_importance'
            plot_dir.mkdir(exist_ok=True)
            
            for method, scores in [('Correlation', corr), ('Mutual_Info', mi_series)]:
                plt.figure(figsize=(10, 8))
                top_features = scores.head(15)
                sns.barplot(x=top_features.values, y=top_features.index)
                plt.title(f'Top 15 Features by {method} with Target')
                plot_path = plot_dir / f"top_features_{method.lower()}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                results[f"{method.lower()}_plot"] = plot_path
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {str(e)}")
        
        return results
    
    def analyze_feature_interactions(self) -> Dict[str, Path]:
        """Analyze interactions between top features"""
        plots = {}
        plot_dir = self.output_dir / 'plots' / 'interactions'
        plot_dir.mkdir(exist_ok=True)
        
        if 'is_prospect' not in self.df.columns:
            return plots
        
        try:
            # Get top numerical features
            num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 2:
                return plots
            
            # Limit to top 5 features for visualization
            top_features = num_cols[:5]
            
            # Pairplot with target hue
            plt.figure(figsize=(12, 10))
            sns.pairplot(
                data=self.df[top_features + ['is_prospect']],
                hue='is_prospect',
                plot_kws={'alpha': 0.6},
                corner=True
            )
            plot_path = plot_dir / "feature_pairplot.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plots["feature_pairplot"] = plot_path
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[top_features].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1
            )
            plt.title("Feature Correlation Heatmap")
            plot_path = plot_dir / "correlation_heatmap.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plots["correlation_heatmap"] = plot_path
            
        except Exception as e:
            self.logger.error(f"Feature interaction analysis failed: {str(e)}")
        
        return plots
    
    def generate_modeling_readiness_report(self) -> Dict[str, Any]:
        """Assess data readiness for modeling"""
        report = {
            'data_quality': self.quality_report,
            'checks': {},
            'recommendations': []
        }
        
        # Check for class imbalance
        if 'is_prospect' in self.df.columns:
            class_balance = self.df['is_prospect'].value_counts(normalize=True)
            report['checks']['class_balance'] = class_balance.to_dict()
            
            if min(class_balance) < 0.2:
                report['recommendations'].append(
                    "Significant class imbalance detected - consider resampling techniques"
                )
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        report['checks']['missing_values'] = missing_values
        if missing_values > 0:
            report['recommendations'].append(
                f"Data contains {missing_values} missing values - impute before modeling"
            )
        
        # Check for high cardinality categoricals
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = {col: self.df[col].nunique() for col in cat_cols 
                           if self.df[col].nunique() > 20}
        report['checks']['high_cardinality'] = high_cardinality
        if high_cardinality:
            report['recommendations'].append(
                f"High cardinality categorical features detected: {list(high_cardinality.keys())}"
            )
        
        # Check for constant features
        constant_features = [col for col in self.df.columns if self.df[col].nunique() == 1]
        report['checks']['constant_features'] = constant_features
        if constant_features:
            report['recommendations'].append(
                f"Remove constant features: {constant_features}"
            )
        
        # Check feature-target relationships
        if 'is_prospect' in self.df.columns:
            num_cols = self.df.select_dtypes(include=np.number).columns
            weak_relationships = []
            
            for col in num_cols:
                if col != 'is_prospect':
                    corr = self.df[[col, 'is_prospect']].corr().iloc[0,1]
                    if abs(corr) < 0.1:  # Threshold for weak relationship
                        weak_relationships.append(col)
            
            report['checks']['weak_relationships'] = weak_relationships
            if weak_relationships:
                report['recommendations'].append(
                    f"Features with weak relationship to target: {weak_relationships[:10]}..."
                )
        
        return report
    
    def save_report_as_pdf(self, selected_plots: List[str]) -> Path:
        """Save selected EDA plots into a single PDF report"""
        pdf_path = self.output_dir / 'reports' / 'full_eda_report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.7, 'Football Player Scouting EDA Report', 
                    ha='center', va='center', fontsize=20, weight='bold')
            plt.text(0.5, 0.6, f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # Add plots
            plots_dir = self.output_dir / 'plots'
            for plot_file in selected_plots:
                plot_path = plots_dir / plot_file
                if plot_path.exists():
                    try:
                        img = imread(plot_path)
                        fig, ax = plt.subplots(figsize=(11, 8.5))
                        ax.imshow(img)
                        ax.axis('off')
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception as e:
                        self.logger.warning(f"Failed to add {plot_file} to PDF: {str(e)}")
        
        self.logger.info(f"PDF report saved to: {pdf_path}")
        return pdf_path
    
    def execute_full_analysis(self) -> Dict[str, Any]:
        """Execute complete EDA pipeline with proper type conversion"""
        try:
            analysis_results = {
                'metadata': self._get_metadata(),
                'distributions': self.analyze_data_distributions(),
                'target_analysis': self.analyze_target_relationships(),
                'feature_importance': self.analyze_feature_importance(),
                'feature_interactions': self.analyze_feature_interactions(),
                'modeling_readiness': self.generate_modeling_readiness_report()
            }
    
            # Convert all results to serializable types
            serializable_results = self._convert_to_serializable(analysis_results)
            
            # Save JSON report
            report_path = self.output_dir / 'reports' / 'eda_results.json'
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
    
            # Generate PDF report with key plots
            key_plots = [
                'distributions/dist_potential_growth.png',
                'target_analysis/target_distribution.png',
                'feature_importance/top_features_correlation.png',
                'feature_interactions/feature_pairplot.png'
            ]
            pdf_path = self.save_report_as_pdf(key_plots)
            
            return {
                'status': 'success',
                'report_paths': {
                    'json': str(report_path),
                    'pdf': str(pdf_path)
                },
                'analysis_summary': {
                    'num_features_analyzed': len(self.df.columns),
                    'target_present': 'is_prospect' in self.df.columns,
                    'data_quality_issues': len(self.quality_report['recommendations'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Full analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

# Example usage
if __name__ == "__main__":
    data_path = Path("data/processed/players_processed.csv")
    output_dir = Path("reports/eda")
    
    try:
        eda = FootballEDA(data_path, output_dir)
        results = eda.execute_full_analysis()
        print(f"Analysis completed. Reports saved to: {output_dir}")
    except Exception as e:
        print(f"EDA failed: {str(e)}")