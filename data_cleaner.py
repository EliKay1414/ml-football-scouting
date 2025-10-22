# src/data_cleaner.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Set, Dict, Optional, Tuple, Union
import json
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from scipy import stats
from scipy.stats import zscore
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette('colorblind')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FIFADataCleaner:
    """Robust FIFA player data cleaning pipeline with comprehensive error handling"""
    
    # Define all class constants
    FINANCIAL_COLS = ['value_eur', 'wage_eur', 'release_clause_eur']
    CORE_COLS = ['sofifa_id', 'short_name', 'player_positions', 'overall', 'potential']
    ATTRIBUTE_COLS = [
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
        'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
        'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
        'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
        'movement_agility', 'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
        'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
        'mentality_positioning', 'mentality_vision', 'mentality_penalties',
        'mentality_composure', 'defending_marking_awareness',
        'defending_standing_tackle', 'defending_sliding_tackle',
        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
        'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'
    ]
    CATEGORICAL_COLS = [
        'player_positions', 'nationality_name', 'club_name',
        'league_name', 'preferred_foot', 'work_rate', 'body_type'
    ]
    IRRELEVANT_COLS = [
        'player_url', 'player_face_url', 'club_logo_url', 'club_flag_url',
        'nation_logo_url', 'nation_flag_url', 'long_name',
        'club_jersey_number', 'nation_jersey_number', 'real_face',
        'player_tags', 'player_traits', 'dob',
        'club_team_id', 'nationality_id', 'nation_team_id', 'potential',
        'club_joined', 'club_contract_valid_until',
        'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
        'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
        'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk', 'league_name', 'club_position', 'club_loaned_from', 'nationality_name', 'nation_position' 
    ]
    
    def __init__(self, raw_df: pd.DataFrame, project_root: Path):
        self.raw_df = raw_df.copy()
        self.df = None
        self.project_root = project_root
        self.visualization_dir = project_root / "visualizations"
        self.visualization_dir.mkdir(exist_ok=True)
        self.report_dir = project_root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.cleaning_report = {
            'steps': {},
            'issues': {},
            'metrics': {},
            'recommendations': []
        }
        self._validate_input()
        self._analyze_columns()
        self._load_config()
    
    def _load_config(self):
        """Load cleaning configuration parameters"""
        self.config = {
            "min_rating": 40,
            "max_rating": 99,
            "min_age": 16,
            "max_age": 45,
            "min_height": 150,
            "max_height": 220,
            "min_weight": 50,
            "max_weight": 120,
            "outlier_threshold": 3.5,
            "max_category_levels": 50,
            "missing_value_threshold": 0.5,
            "clean_numeric": True
        }
    
    def _validate_input(self):
        """Validate input data structure"""
        if not isinstance(self.raw_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if len(self.raw_df) == 0:
            raise ValueError("Empty DataFrame provided")
        if not any(col in self.raw_df.columns for col in ['overall', 'potential']):
            raise ValueError("DataFrame must contain either 'overall' or 'potential' rating")
    
    def _analyze_columns(self):
        """Analyze which columns exist in the dataset"""
        self.existing_cols = set(self.raw_df.columns)
        logger.info(f"Found {len(self.existing_cols)} columns in dataset")
        
        self.financial_cols = [col for col in self.FINANCIAL_COLS if col in self.existing_cols]
        self.core_cols = [col for col in self.CORE_COLS if col in self.existing_cols]
        self.attribute_cols = [col for col in self.ATTRIBUTE_COLS if col in self.existing_cols]
        self.categorical_cols = [col for col in self.CATEGORICAL_COLS if col in self.existing_cols]
        self.irrelevant_cols = [col for col in self.IRRELEVANT_COLS if col in self.existing_cols]
        
        logger.info(f"Identified {len(self.financial_cols)} financial columns")
        logger.info(f"Identified {len(self.attribute_cols)} player attribute columns")
        logger.info(f"Identified {len(self.categorical_cols)} categorical columns")
        logger.info(f"Identified {len(self.irrelevant_cols)} irrelevant columns to remove")
        
        self.cleaning_report['initial_columns'] = {
            'total': len(self.existing_cols),
            'financial': len(self.financial_cols),
            'attributes': len(self.attribute_cols),
            'categorical': len(self.categorical_cols),
            'irrelevant': len(self.irrelevant_cols)
        }

    def clean(self) -> pd.DataFrame:
        """Execute the complete cleaning pipeline"""
        try:
            self.df = self.raw_df.copy()
            
            #  Remove irrelevant columns
            removed_cols = self._remove_irrelevant_columns()
            self.cleaning_report['steps']['column_removal'] = {
                'removed_columns': removed_cols,
                'remaining_columns': len(self.df.columns)
            }
            
            #  Standardize data types
            self._standardize_data_types()
            
            #  Handle duplicates
            duplicates_removed = self._handle_duplicates()
            self.cleaning_report['steps']['duplicate_handling'] = {
                'duplicates_removed': duplicates_removed
            }
            
            #  Clean player positions
            self._clean_player_positions()
            
            #  Clean categorical features
            encoding_report = self._clean_categorical_features()
            self.cleaning_report['steps']['categorical_cleaning'] = encoding_report
            
            #  Handle missing data FIRST
            missing_data_report = self._handle_missing_data()
            self.cleaning_report['steps']['missing_data'] = missing_data_report
            
            #  clean numeric features
            outlier_report = self._clean_numeric_features()
            self.cleaning_report['steps']['numeric_cleaning'] = outlier_report
            
            #  Validate ratings
            self._validate_ratings()
            
            #  Optimize memory
            memory_savings = self._optimize_memory_usage()
            self.cleaning_report['steps']['memory_optimization'] = memory_savings
            
            # Final quality checks
            self._final_quality_checks()
            
            # Generate report
            self._generate_report()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Cleaning failed: {str(e)}", exc_info=True)
            self.cleaning_report['errors'] = str(e)
            try:
                self._generate_report()
            except Exception as report_error:
                logger.error(f"Failed to generate error report: {str(report_error)}")
            raise RuntimeError(f"Cleaning failed: {str(e)}") from e
    
    def _remove_irrelevant_columns(self) -> List[str]:
        """Remove unnecessary columns"""
        initial_cols = set(self.df.columns)
        cols_to_remove = [col for col in self.irrelevant_cols if col in self.df.columns]
        cols_to_remove = [col for col in cols_to_remove if col not in self.FINANCIAL_COLS]
        
        if cols_to_remove:
            self.df.drop(columns=cols_to_remove, inplace=True)
            removed_cols = initial_cols - set(self.df.columns)
            logger.info(f"Removed {len(removed_cols)} irrelevant columns")
            return list(removed_cols)
        return []

    def _clean_player_positions(self):
        """Clean and standardize player positions"""
        if 'player_positions' in self.df.columns:
            self.df['player_positions'] = (
                self.df['player_positions']
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(' ', '')
                .fillna('UNKNOWN')
            )
            
            self.df['primary_position'] = (
                self.df['player_positions']
                .str.split(',')
                .str[0]
                .str.strip()
            )
            
            position_mapping = {
                'GK': 'Goalkeeper',
                'CB': 'Defender', 'LB': 'Defender', 'RB': 'Defender', 
                'LWB': 'Defender', 'RWB': 'Defender',
                'CDM': 'Midfielder', 'CM': 'Midfielder', 'CAM': 'Midfielder',
                'LM': 'Midfielder', 'RM': 'Midfielder',
                'LW': 'Forward', 'RW': 'Forward', 'CF': 'Forward', 'ST': 'Forward'
            }
            
            self.df['position_category'] = (
                self.df['primary_position']
                .map(position_mapping)
                .fillna('Other')
            )

        
    def _clean_numeric_features(self) -> Dict:
        """Clean numeric features: clip values, handle outliers using z-score, and cap financial extremes."""
        outlier_report = {}
        threshold = self.config["outlier_threshold"]
    
        # 1. Clip physical attributes to realistic ranges
        if 'age' in self.df.columns:
            initial_outliers = len(self.df[~self.df['age'].between(self.config["min_age"], self.config["max_age"])])
            self.df['age'] = self.df['age'].clip(self.config["min_age"], self.config["max_age"]).astype('Int64')
            outlier_report['age'] = {'outliers_found': initial_outliers, 'action': 'clipped'}
    
        if 'height_cm' in self.df.columns:
            initial_outliers = len(self.df[~self.df['height_cm'].between(self.config["min_height"], self.config["max_height"])])
            self.df['height_cm'] = self.df['height_cm'].clip(self.config["min_height"], self.config["max_height"]).astype('Int64')
            outlier_report['height_cm'] = {'outliers_found': initial_outliers, 'action': 'clipped'}
    
        if 'weight_kg' in self.df.columns:
            initial_outliers = len(self.df[~self.df['weight_kg'].between(self.config["min_weight"], self.config["max_weight"])])
            self.df['weight_kg'] = self.df['weight_kg'].clip(self.config["min_weight"], self.config["max_weight"]).astype('Int64')
            outlier_report['weight_kg'] = {'outliers_found': initial_outliers, 'action': 'clipped'}
    
        # 2. Detect and handle outliers in player attribute columns using z-score
        for col in self.attribute_cols:
            if col in self.df.columns:
                try:
                    z_scores = zscore(self.df[col].dropna())
                    abs_z_scores = np.abs(z_scores)
                    outliers_mask = abs_z_scores > threshold
                    outliers_indices = self.df[col].dropna().index[outliers_mask]
    
                    if len(outliers_indices) > 0:
                        self.df.loc[outliers_indices, col] = np.nan  # mark as missing
                        logger.info(f"{len(outliers_indices)} outliers removed from {col} using z-score > {threshold}")
                        outlier_report[col] = {
                            'outliers_found': int(len(outliers_indices)),
                            'method': 'z-score',
                            'threshold': threshold,
                            'action': 'set to NaN for re-imputation'
                        }
                except Exception as e:
                    logger.warning(f"Z-score failed for {col}: {str(e)}")
    
        # 3. Clip attribute values to be within 0-100 range
        for col in self.attribute_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(0, 100).astype('Int64')
    
        # 4. Handle financial columns (log transform + cap outliers)
        for col in self.FINANCIAL_COLS:
            if col in self.df.columns:
                self.df[f'log_{col}'] = np.log1p(self.df[col])  # log(1 + x)
                cap_value = self.df[col].quantile(0.99)
                outliers = self.df[col] > cap_value
                if outliers.any():
                    self.df.loc[outliers, col] = cap_value
                    outlier_report[col] = {
                        'outliers_found': int(outliers.sum()),
                        'action': 'capped at 99th percentile',
                        'cap_value': float(cap_value)
                    }
    
        return outlier_report

    def _clean_categorical_features(self) -> Dict:
        """Clean and encode categorical features"""
        encoding_report = {}
        
        for col in self.categorical_cols:
            if col in self.df.columns and col not in ['player_positions', 'primary_position']:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .fillna('UNKNOWN')
                )
                
                unique_count = len(self.df[col].unique())
                if unique_count > self.config["max_category_levels"]:
                    top_categories = self.df[col].value_counts().nlargest(
                        self.config["max_category_levels"] - 1).index.tolist()
                    self.df[col] = np.where(
                        self.df[col].isin(top_categories),
                        self.df[col],
                        'OTHER'
                    )
                    encoding_report[col] = {
                        'action': 'grouped',
                        'original_unique': unique_count,
                        'final_unique': len(self.df[col].unique())
                    }
                else:
                    encoding_report[col] = {
                        'action': 'cleaned',
                        'unique_values': unique_count
                    }
        
        if 'work_rate' in self.df.columns:
            self.df[['work_rate_att', 'work_rate_def']] = (
                self.df['work_rate'].str.split('/', expand=True))
            self.df.drop(columns=['work_rate'], inplace=True)
            
            for wr_col in ['work_rate_att', 'work_rate_def']:
                self.df[wr_col] = (
                    self.df[wr_col]
                    .str.strip()
                    .str.upper()
                    .fillna('MEDIUM')
                )
            
            encoding_report['work_rate'] = {
                'action': 'split',
                'new_columns': ['work_rate_att', 'work_rate_def']
            }
        
        return encoding_report

    def _handle_missing_data(self) -> Dict:
        """Comprehensive missing data handling with position-aware imputation"""
        missing_report = {}
    
        # 1. Drop columns with >50% missing values (actually drop them here)
        cols_to_drop = []
        for col in self.df.columns:
            missing_pct = self.df[col].isna().mean()
            if missing_pct > self.config["missing_value_threshold"]:
                cols_to_drop.append(col)
                missing_report[f'dropped_{col}'] = {
                    'missing_pct': missing_pct,
                    'action': 'dropped_column'
                }
    
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >50% missing values")
    
        # 2. Handle goalkeeper attributes (fillna: 0 for non-GK, median for GK)
        gk_cols = [col for col in self.df.columns if 'goalkeeping_' in col]
        for col in gk_cols:
            if col in self.df.columns and self.df[col].isna().any():
                gk_mask = (self.df['position_category'] == 'Goalkeeper') if 'position_category' in self.df.columns else False
                if gk_mask is not False and gk_mask.any():
                    gk_median = self.df.loc[gk_mask, col].median()
                    self.df[col] = np.where(gk_mask,
                                           self.df[col].fillna(gk_median),
                                           self.df[col].fillna(0))
                    missing_report[f'{col}_missing'] = {
                        'action': '0 for non-GKs, median for GKs',
                        'imputed_value': float(gk_median) if not pd.isna(gk_median) else 'N/A'
                    }
                else:
                    self.df[col] = self.df[col].fillna(0)
                    missing_report[f'{col}_missing'] = {'action': 'filled with 0'}
    
        # 3. Core attributes with position-aware median imputation
        core_attrs = ['pace', 'shooting', 'passing', 'dribbling', 
                     'defending', 'physic', 'mentality_composure']
        for attr in core_attrs:
            if attr in self.df.columns and self.df[attr].isna().any():
                if 'primary_position' in self.df.columns:
                    self.df[attr] = self.df.groupby('primary_position')[attr].transform(
                        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else self.df[attr].median()))
                    missing_report[f'{attr}_missing'] = {
                        'action': 'position-specific median with fallback'
                    }
                else:
                    median_val = self.df[attr].median()
                    self.df[attr] = self.df[attr].fillna(median_val)
                    missing_report[f'{attr}_missing'] = {
                        'action': f'filled with median {median_val:.1f}'
                    }
    
        # 4. Financial columns imputation (position + overall + age grouped median)
        financial_cols = ['value_eur', 'wage_eur', 'release_clause_eur']
        for col in financial_cols:
            if col in self.df.columns and self.df[col].isna().any():
                if all(x in self.df.columns for x in ['overall', 'age', 'primary_position']):
                    # Use group median with fallback
                    def fill_func(x):
                        med = x.median()
                        return x.fillna(med if not pd.isna(med) else self.df[col].median())
                    
                    self.df[col] = self.df.groupby(
                        ['primary_position', pd.qcut(self.df['overall'], 5, duplicates='drop'), pd.qcut(self.df['age'], 3, duplicates='drop')]
                    )[col].transform(fill_func)
                    missing_report[f'{col}_missing'] = {
                        'action': 'position/rating/age grouped median with fallback'
                    }
                else:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    missing_report[f'{col}_missing'] = {
                        'action': f'filled with median {median_val:.1f}'
                    }
    
        # 5. league_level sensible default
        if 'league_level' in self.df.columns and self.df['league_level'].isna().any():
            self.df['league_level'] = self.df['league_level'].fillna(3).astype(int)
            missing_report['league_level_missing'] = {
                'action': 'filled with 3 (lowest tier)'
            }
    
        # 6. Final pass: fill remaining missing values by type
        remaining_missing = self.df.columns[self.df.isna().any()].tolist()
        if remaining_missing:
            logger.warning(f"Filling remaining missing values in: {remaining_missing}")
            for col in remaining_missing:
                if col in self.df.columns:  # Check if column still exists
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        median_val = self.df[col].median()
                        self.df[col] = self.df[col].fillna(median_val)
                        missing_report[f'{col}_missing'] = {
                            'action': f'filled with median {median_val:.1f}'
                        }
                    else:
                        self.df[col] = self.df[col].fillna('UNKNOWN')
                        missing_report[f'{col}_missing'] = {
                            'action': 'filled with UNKNOWN'
                        }
    
        # Final verification (no missing left)
        if self.df.isna().any().any():
            remaining = self.df.isna().sum().sum()
            logger.warning(f"Warning: {remaining} missing values remain after cleaning")
            # Instead of raising an error, we'll just log it
            self.cleaning_report['issues']['remaining_missing_values'] = remaining
    
        return missing_report

    def _standardize_data_types(self):
        """Convert columns to proper data types"""
        if 'sofifa_id' in self.df.columns:
            self.df['sofifa_id'] = self.df['sofifa_id'].astype(str)
        
        if 'year' in self.df.columns:
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce').fillna(0).astype(int)
        
        for col in self.FINANCIAL_COLS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def _handle_duplicates(self) -> int:
        """Handle duplicate records with proper error handling"""
        try:
            if 'sofifa_id' in self.df.columns:
                initial_count = len(self.df)
                completeness = self.df.notna().sum(axis=1)
                self.df = self.df.iloc[completeness.argsort()[::-1]]
                self.df = self.df.drop_duplicates(subset=['sofifa_id'], keep='first')
                duplicates_removed = initial_count - len(self.df)
                logger.info(f"Removed {duplicates_removed} duplicate records")
                return duplicates_removed
            return 0
        except Exception as e:
            logger.error(f"Error handling duplicates: {str(e)}")
            return 0  # Return 0 if duplicate handling fails

    def _validate_ratings(self):
        """Validate and clean rating columns"""
        if 'overall' in self.df.columns:
            invalid = ~self.df['overall'].between(
                self.config["min_rating"], self.config["max_rating"])
            if invalid.any():
                logger.warning(f"Found {invalid.sum()} invalid overall ratings")
            self.df['overall'] = self.df['overall'].clip(
                self.config["min_rating"], self.config["max_rating"]).astype('Int64')
        
        if 'potential' in self.df.columns:
            invalid = ~self.df['potential'].between(
                self.config["min_rating"], self.config["max_rating"])
            if invalid.any():
                logger.warning(f"Found {invalid.sum()} invalid potential ratings")
            self.df['potential'] = self.df['potential'].clip(
                self.config["min_rating"], self.config["max_rating"]).astype('Int64')
            
            if 'overall' in self.df.columns:
                mask = self.df['potential'] < self.df['overall']
                if mask.any():
                    logger.warning(f"Fixed {mask.sum()} cases where potential < overall")
                self.df.loc[mask, 'potential'] = self.df.loc[mask, 'overall']
    
    def _optimize_memory_usage(self) -> Dict:
        """Reduce memory footprint with intelligent type conversion"""
        initial_memory = self.df.memory_usage(deep=True).sum()
        
        for col in self.df.select_dtypes(include=['integer']):
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        
        for col in self.df.select_dtypes(include=['float']):
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        for col in self.df.select_dtypes(include=['object']):
            num_unique = len(self.df[col].unique())
            num_total = len(self.df)
            if 1 < num_unique < num_total // 2:
                self.df[col] = self.df[col].astype('category')
        
        final_memory = self.df.memory_usage(deep=True).sum()
        savings = initial_memory - final_memory
        
        return {
            'initial_memory': int(initial_memory),
            'final_memory': int(final_memory),
            'savings': int(savings),
            'savings_pct': float((savings / initial_memory) * 100)
        }
    
    def _final_quality_checks(self):
        """Final data quality validation with comprehensive missing value handling"""
        # 1. Check for remaining missing values
        missing_cols = self.df.columns[self.df.isna().any()].tolist()
        
        if missing_cols:
            logger.warning(f"Columns with remaining missing values: {missing_cols}")
            
            # Create detailed missing value report
            missing_counts = self.df[missing_cols].isna().sum().to_dict()
            missing_pcts = (self.df[missing_cols].isna().mean() * 100).round(2).to_dict()
            
            self.cleaning_report['issues']['remaining_missing'] = {
                'columns': missing_cols,
                'counts': {k: int(v) for k, v in missing_counts.items()},
                'percentages': {k: float(v) for k, v in missing_pcts.items()}
            }
            
            # Try one last imputation pass for numeric columns
            for col in missing_cols:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    logger.info(f"Filled remaining missing values in {col} with median {median_val:.1f}")
        
        # 2. Check for invalid ratings
        if 'overall' in self.df.columns:
            invalid = self.df[~self.df['overall'].between(
                self.config["min_rating"], self.config["max_rating"])]
            if not invalid.empty:
                logger.warning(f"Found {len(invalid)} players with invalid overall ratings")
                self.cleaning_report['issues']['invalid_overall'] = int(len(invalid))
        
        if 'potential' in self.df.columns:
            invalid = self.df[~self.df['potential'].between(
                self.config["min_rating"], self.config["max_rating"])]
            if not invalid.empty:
                logger.warning(f"Found {len(invalid)} players with invalid potential ratings")
                self.cleaning_report['issues']['invalid_potential'] = int(len(invalid))
        
        # 3. Check for high cardinality categoricals
        high_cardinality = {}
        for col in self.df.select_dtypes(include=['category', 'object']):
            unique_count = len(self.df[col].unique())
            if unique_count > self.config["max_category_levels"]:
                high_cardinality[col] = int(unique_count)
                self.cleaning_report['recommendations'].append(
                    f"Consider reducing cardinality for {col} ({unique_count} unique values)")
        
        if high_cardinality:
            self.cleaning_report['issues']['high_cardinality'] = high_cardinality
        
        # 4. Final check - ensure no missing values in core columns
        core_cols_missing = [col for col in self.core_cols 
                            if col in self.df.columns and self.df[col].isna().any()]
        if core_cols_missing:
            logger.error(f"Critical: Missing values in core columns: {core_cols_missing}")
            for col in core_cols_missing:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna('UNKNOWN')
        
        logger.info("Completed final quality checks")
    
    def _generate_report(self):
        """Generate a comprehensive cleaning report with robust JSON serialization"""
        report_path = self.report_dir / "cleaning_report.json"
        
        def convert_to_serializable(obj):
            """Recursively convert numpy/pandas types to native Python types"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):  # Handle pandas NA values
                return None
            elif hasattr(obj, 'dtype'):  # Catch any other pandas/numpy types
                return str(obj)
            return obj
    
        try:
            # First convert the entire report to serializable types
            serializable_report = convert_to_serializable(self.cleaning_report)
            
            # Then verify it's fully serializable by doing a test dump
            json.dumps(serializable_report)  # Test serialization
            
            # If we get here, we can safely write the file
            with open(report_path, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            logger.info(f"Saved cleaning report to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            
            # Try to save a minimal error report
            try:
                error_report = {
                    'error': str(e),
                    'report_available': False,
                    'message': 'Full report could not be generated due to serialization error'
                }
                with open(report_path, 'w') as f:
                    json.dump(error_report, f, indent=2)
                logger.warning(f"Saved minimal error report to {report_path}")
            except:
                logger.error("Could not save any report file")
            
            return None
def clean_fifa_data(raw_df: pd.DataFrame, project_root: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Main cleaning function with robust error handling
    
    Args:
        raw_df: Raw input DataFrame
        project_root: Path to project root directory
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
        Returns (None, error_report) if cleaning fails
    """
    try:
        cleaner = FIFADataCleaner(raw_df, project_root)
        cleaned_df = cleaner.clean()
        
        # Ensure all expected report sections exist
        if 'steps' not in cleaner.cleaning_report:
            cleaner.cleaning_report['steps'] = {}
        if 'duplicate_handling' not in cleaner.cleaning_report['steps']:
            cleaner.cleaning_report['steps']['duplicate_handling'] = {'duplicates_removed': 0}
            
        return cleaned_df, cleaner.cleaning_report
        
    except Exception as e:
        error_report = {
            'error': str(e),
            'status': 'failed',
            'steps_completed': cleaner.cleaning_report.get('steps', {})
        }
        logger.error(f"Fatal error during cleaning: {str(e)}", exc_info=True)
        return None, error_report