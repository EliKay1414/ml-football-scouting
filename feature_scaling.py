import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    MaxAbsScaler, PowerTransformer, QuantileTransformer,
    OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
import logging
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = None

    def fit(self, X, y=None):
        self.iforest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.iforest.fit(X)
        return self

    def transform(self, X):
        preds = self.iforest.predict(X)
        X_copy = X.copy()
        # Replace outliers with NaN (to avoid changing row count)
        X_copy[preds == -1] = np.nan
        return X_copy



class SmartScaler(BaseEstimator, TransformerMixin):
    """
    Enhanced automatic scaler that:
    1. Handles sparse data efficiently
    2. Provides better outlier detection
    3. Includes feature importance feedback
    4. Optimizes for tree-based and neural network models
    """
    
    def __init__(self, method='auto', tree_mode=False, nn_mode=False):
        """
        Parameters:
        -----------
        method : str
            Scaling method ('auto', 'standard', 'robust', 'minmax', 'maxabs', 'quantile')
        tree_mode : bool
            Optimize for tree-based models (less scaling needed)
        nn_mode : bool
            Optimize for neural networks (more careful scaling)
        """
        self.method = method
        self.tree_mode = tree_mode
        self.nn_mode = nn_mode
        self.scalers_ = {}
        self.feature_stats_ = {}
        self.feature_importance_ = {}
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            
        X = self._ensure_dataframe(X)
        
        for col in self.feature_names_:
            col_data = X[col]
            
            # Skip empty or constant features
            if col_data.nunique() <= 1:
                self.scalers_[col] = 'passthrough'
                continue
                
            # Store feature statistics
            self._store_feature_stats(col, col_data)
            
            # Handle sparse data
            if hasattr(col_data, 'sparse'):
                col_data = col_data.sparse.to_dense()
                
            # Select scaler based on data characteristics
            if self.method != 'auto':
                scaler = self._get_fixed_scaler(self.method)
            else:
                scaler = self._select_best_scaler(col_data)
                
            # Fit the scaler
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if scaler != 'passthrough':
                    scaler.fit(X[[col]])
            self.scalers_[col] = scaler
            
        return self
    
    def transform(self, X):
        X = self._ensure_dataframe(X)
        X_scaled = X.copy()
        
        for col, scaler in self.scalers_.items():
            if col in X.columns and scaler != 'passthrough':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_scaled[col] = scaler.transform(X[[col]]).ravel()
                    
        return X_scaled
    
    def _ensure_dataframe(self, X):
        """Ensure input is DataFrame with proper column names"""
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, 'feature_names_') and len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            return pd.DataFrame(X)
        return X
    
    def _store_feature_stats(self, col, data):
        """Calculate and store feature statistics"""
        stats = {
            'mean': data.mean(),
            'std': data.std(),
            'skew': data.skew(),
            'kurtosis': data.kurtosis(),
            'zeros_ratio': (data == 0).mean(),
            'outlier_ratio': self._calculate_outlier_ratio(data)
        }
        self.feature_stats_[col] = stats
        
    def _calculate_outlier_ratio(self, data):
        """Calculate percentage of outliers using IQR method"""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return ((data < lower_bound) | (data > upper_bound)).mean()
    
    def _select_best_scaler(self, data):
        """Intelligently select the best scaler based on data characteristics"""
        stats = {
            'skew': abs(data.skew()),
            'kurtosis': abs(data.kurtosis()),
            'zeros_ratio': (data == 0).mean(),
            'outlier_ratio': self._calculate_outlier_ratio(data)
        }
        
        # For tree-based models, scaling is less critical
        if self.tree_mode:
            return 'passthrough'
            
        # Handle sparse data (common in one-hot encoded features)
        if stats['zeros_ratio'] > 0.8:
            return MaxAbsScaler()
            
        # Handle heavy-tailed distributions
        if stats['skew'] > 2 or stats['kurtosis'] > 3.5:
            if (data > 0).all():
                try:
                    return PowerTransformer(method='box-cox')
                except:
                    return PowerTransformer(method='yeo-johnson')
            return PowerTransformer(method='yeo-johnson')
            
        # Handle features with many outliers
        if stats['outlier_ratio'] > 0.05:
            return RobustScaler()
            
        # Default to standard scaling for neural networks
        if self.nn_mode:
            return StandardScaler()
            
        # For other cases, use quantile transformer for robust normalization
        return QuantileTransformer(output_distribution='normal', n_quantiles=100)
    
    def _get_fixed_scaler(self, method):
        """Return specified scaler type"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'passthrough': 'passthrough'
        }
        return scalers.get(method, StandardScaler())

class FeatureQualityAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyzes feature quality and provides feedback for:
    - Constant/quasi-constant features
    - Highly correlated features
    - Features with high missing values
    - Features with suspicious distributions
    """
    
    def __init__(self, correlation_threshold=0.95, missing_threshold=0.5):
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.quality_report_ = {}
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.copy()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        self._analyze_constant_features(X)
        self._analyze_missing_values(X)
        self._analyze_correlations(X)
        self._analyze_distributions(X)
        
        return self
    
    def transform(self, X):
        # This transformer doesn't modify data, just analyzes it
        return X
    
    def get_report(self):
        """Return detailed quality report"""
        return pd.DataFrame.from_dict(self.quality_report_, orient='index')
    
    def _analyze_constant_features(self, X):
        """Identify constant or quasi-constant features"""
        # Select only numeric columns to avoid ValueError
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns to analyze for constant features")
            return
        
        variance_threshold = VarianceThreshold()
        variance_threshold.fit(X[numeric_cols])
        
        for col in numeric_cols:
            idx = list(numeric_cols).index(col)
            if not variance_threshold.get_support()[idx]:
                self.quality_report_[col] = {
                    'issue': 'constant_feature',
                    'severity': 'high',
                    'suggestion': 'Consider removing this feature'
                }

    
    def _analyze_missing_values(self, X):
        """Identify features with excessive missing values"""
        missing_ratios = X.isna().mean()
        
        for col, ratio in missing_ratios.items():
            if ratio > self.missing_threshold:
                self.quality_report_[col] = {
                    'issue': f"high_missing_values ({ratio:.1%})",
                    'severity': 'high' if ratio > 0.8 else 'medium',
                    'suggestion': 'Consider imputation or removal'
                }
    
    def _analyze_correlations(self, X):
        """Identify highly correlated feature pairs"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns to analyze for correlations")
            return
    
        corr_matrix = X[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
        for col in numeric_cols:
            correlated_with = upper[col][upper[col] > self.correlation_threshold]
            if not correlated_with.empty:
                self.quality_report_[col] = {
                    'issue': f"high_correlation_with {correlated_with.idxmax()} ({correlated_with.max():.2f})",
                    'severity': 'medium',
                    'suggestion': 'Consider removing one of the correlated features'
                }

    
    def _analyze_distributions(self, X):
        """Identify problematic distributions"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns to analyze distributions")
            return
        
        for col in numeric_cols:
            data = X[col].dropna()
            
            # Check for extreme skewness
            skew = data.skew()
            if abs(skew) > 5:
                self.quality_report_[col] = {
                    'issue': f"extreme_skew ({skew:.2f})",
                    'severity': 'medium',
                    'suggestion': 'Consider log/power transformation'
                }
                
            # Check for binary features with imbalance
            if data.nunique() == 2:
                ratio = data.value_counts(normalize=True).min()
                if ratio < 0.1:
                    self.quality_report_[col] = {
                        'issue': f"imbalanced_binary ({ratio:.1%})",
                        'severity': 'medium',
                        'suggestion': 'Consider oversampling or class weights'
                    }

class FootballFeaturePreprocessor:
    """
    Enhanced feature preprocessor for football player data with:
    - Better memory management
    - Advanced scaling options
    - Feature quality analysis
    - Model-specific optimizations
    """
    
    def __init__(self, target='overall', model_type='auto', 
                 max_categories=50, n_components=None):
        """
        Parameters:
        -----------
        target : str
            Target variable name
        model_type : str
            Type of model ('tree', 'linear', 'nn', 'auto')
        max_categories : int
            Maximum number of categories for one-hot encoding
        n_components : int
            Number of PCA components to keep (None for no PCA)
        """
        self.target = target
        self.model_type = model_type
        self.max_categories = max_categories
        self.n_components = n_components
        self.feature_analyzer = FeatureQualityAnalyzer()
        
    def _identify_feature_types(self, X):
        """Enhanced feature type identification with dynamic detection"""
        features = {
            'base_numerical': ['overall', 'potential', 'value_eur', 'wage_eur', 'age', 
                              'height_cm', 'weight_kg', 'release_clause_eur'],
            'skill_numerical': ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                               'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
                               'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
                               'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
                               'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
                               'movement_agility', 'movement_reactions', 'movement_balance',
                               'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
                               'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
                               'mentality_positioning', 'mentality_vision', 'mentality_penalties',
                               'mentality_composure', 'defending_marking_awareness',
                               'defending_standing_tackle', 'defending_sliding_tackle'],
            'gk_numerical': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                            'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'],
            'categorical': ['preferred_foot', 'body_type', 'club_name'],
            'ordinal': ['weak_foot', 'skill_moves', 'international_reputation', 'league_level'],
            'mixed': ['work_rate'],
            'position': ['player_positions']
        }
        
        # Dynamic detection for numerical features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        extra_numerical = [col for col in numerical_cols 
                          if col not in sum(features.values(), []) 
                          and col != self.target]
        features['extra_numerical'] = extra_numerical
        
        # Dynamic detection for categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        extra_categorical = [col for col in categorical_cols 
                            if col not in sum(features.values(), [])]
        features['extra_categorical'] = extra_categorical
        
        # Only keep feature types that exist in the data
        valid_features = {}
        for ft_type, cols in features.items():
            valid_cols = [col for col in cols if col in X.columns]
            if valid_cols:
                valid_features[ft_type] = valid_cols
                
        return valid_features
    
    def _build_numerical_pipeline(self):
        """Build pipeline for numerical features with model-specific scaling"""
        tree_mode = self.model_type in ['tree', 'auto']
        nn_mode = self.model_type == 'nn'
        
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier', self._build_outlier_handler()),
            ('scaler', SmartScaler(tree_mode=tree_mode, nn_mode=nn_mode))
        ]
        
        if self.n_components:
            steps.append(('pca', PCA(n_components=self.n_components)))
            
        return Pipeline(steps)
    
    def _build_outlier_handler(self):
        return Pipeline([
            ('detect', IsolationForestTransformer(contamination=0.05, random_state=42)),
            ('impute', SimpleImputer(strategy='median'))
        ])

    
    def _build_categorical_pipeline(self):
        """Enhanced categorical pipeline with rare category handling"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                    max_categories=self.max_categories,
                                    sparse_output=True,
                                    dtype=np.int8))
        ])
    
    def _build_ordinal_pipeline(self):
        """Ordinal pipeline with optimal binning"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('binner', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')),
            ('scaler', MinMaxScaler())
        ])
    
    def _build_position_pipeline(self):
        """Enhanced position processing"""
        return Pipeline([
            ('processor', PositionProcessor()),
            ('encoder', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
        ])
    
    def _build_workrate_pipeline(self):
        """Enhanced work rate processing"""
        return Pipeline([
            ('splitter', WorkRateSplitter()),
            ('scaler', MinMaxScaler())
        ])
    
    def fit(self, X, y=None):
        """Fit preprocessor with feature quality analysis"""
        X = X.dropna(axis=1, how='all')
        
        # Analyze feature quality first
        self.feature_analyzer.fit(X)
        logger.info("Feature quality analysis completed")
        
        # Get feature types
        feature_types = self._identify_feature_types(X)
        
        # Build transformers
        transformers = []
        if 'base_numerical' in feature_types:
            transformers.append(('base_num', self._build_numerical_pipeline(), 
                               feature_types['base_numerical']))
        if 'skill_numerical' in feature_types:
            transformers.append(('skill_num', self._build_numerical_pipeline(), 
                               feature_types['skill_numerical']))
        if 'gk_numerical' in feature_types:
            transformers.append(('gk_num', self._build_numerical_pipeline(), 
                               feature_types['gk_numerical']))
        if 'extra_numerical' in feature_types:
            transformers.append(('extra_num', self._build_numerical_pipeline(), 
                               feature_types['extra_numerical']))
        if 'categorical' in feature_types:
            transformers.append(('cat', self._build_categorical_pipeline(), 
                               feature_types['categorical']))
        if 'extra_categorical' in feature_types:
            transformers.append(('extra_cat', self._build_categorical_pipeline(), 
                               feature_types['extra_categorical']))
        if 'ordinal' in feature_types:
            transformers.append(('ordinal', self._build_ordinal_pipeline(), 
                               feature_types['ordinal']))
        if 'position' in feature_types:
            transformers.append(('pos', self._build_position_pipeline(), 
                               feature_types['position']))
        if 'mixed' in feature_types:
            transformers.append(('work_rate', self._build_workrate_pipeline(), 
                               feature_types['mixed']))
        
        # Build column transformer
        self.preprocessor = ColumnTransformer(
            transformers,
            remainder='drop',
            sparse_threshold=1.0
        )
        
        self.preprocessor.fit(X, y)
        self._set_feature_names()
        self.is_fitted_ = True
        
        logger.info(f"Preprocessor fitted with {len(self.feature_names_)} features")
        return self
    
    def _set_feature_names(self):
        """Set feature names after transformation"""
        feature_names = []
    
        for name, trans, cols in self.preprocessor.transformers_:
            try:
                # Case 1: If it's a pipeline
                if hasattr(trans, 'steps'):
                    last_step = trans.steps[-1][1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        names = last_step.get_feature_names_out(cols)
                    else:
                        names = cols  # fallback if last step has no feature names
                # Case 2: Transformer directly has get_feature_names_out
                elif hasattr(trans, 'get_feature_names_out'):
                    names = trans.get_feature_names_out(cols)
                # Case 3: Hardcoded mappings or fallbacks
                elif name == 'work_rate':
                    names = ['work_rate_attack', 'work_rate_defense']
                else:
                    names = cols  # default fallback
            except Exception as e:
                # Optional: log or print the fallback
                logger.warning(f"⚠️ Could not get feature names for transformer '{name}': {e}")
                names = cols
    
            # Append feature names to the master list
            if isinstance(names, (list, np.ndarray)):
                feature_names.extend(names)
            else:
                feature_names.append(names)
    
        self.feature_names_ = feature_names

    
    def transform(self, X, chunk_size=None):
        """Transform data with optional chunking"""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted")
            
        if chunk_size is None:
            return self._transform_full(X)
        return self._transform_chunked(X, chunk_size)
    
    def _transform_full(self, X):
        """Transform full dataset"""
        X_transformed = self.preprocessor.transform(X)
        
        if sparse.issparse(X_transformed):
            return pd.DataFrame.sparse.from_spmatrix(
                X_transformed.astype(np.float32),
                columns=self.feature_names_,
                index=X.index
            )
        return pd.DataFrame(
            X_transformed,
            columns=self.feature_names_,
            index=X.index
        )
    
    def _transform_chunked(self, X, chunk_size):
        """Transform data in chunks"""
        chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size]
            chunk_transformed = self.preprocessor.transform(chunk)
            
            if sparse.issparse(chunk_transformed):
                df_chunk = pd.DataFrame.sparse.from_spmatrix(
                    chunk_transformed.astype(np.float32),
                    columns=self.feature_names_,
                    index=chunk.index
                )
            else:
                df_chunk = pd.DataFrame(
                    chunk_transformed,
                    columns=self.feature_names_,
                    index=chunk.index
                )
            chunks.append(df_chunk)
            
        return pd.concat(chunks)
    
    def get_feature_quality_report(self):
        """Get detailed feature quality report"""
        return self.feature_analyzer.get_report()

# Helper classes (unchanged from original)
class PositionProcessor(BaseEstimator, TransformerMixin):
    """Processes player positions to extract primary position."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        results = []
        for positions in X:
            if pd.isna(positions[0]) or not positions[0]:
                results.append(['UNK'])
            else:
                primary_pos = positions[0].split(',')[0].strip()
                results.append([primary_pos])
                
        return np.array(results, dtype=object)

class WorkRateSplitter(BaseEstimator, TransformerMixin):
    """Splits work rate into attack and defense components."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        results = []
        for work_rate in X:
            if pd.isna(work_rate[0]) or not work_rate[0]:
                results.append([0, 0])
                continue
                
            try:
                attack, defense = work_rate[0].split('/')
                attack_score = self._convert_work_rate(attack.strip())
                defense_score = self._convert_work_rate(defense.strip())
                results.append([attack_score, defense_score])
            except:
                results.append([0, 0])
                
        return np.array(results)
    
    def _convert_work_rate(self, rate):
        mapping = {'High': 3, 'Medium': 2, 'Low': 1}
        return mapping.get(rate, 0)

