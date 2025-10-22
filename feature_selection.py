import sys
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import logging
from collections import Counter
from typing import Dict, List, Optional
import warnings
import os

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler("feature_engineering.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Ensure all numeric columns are properly formatted."""
    def __init__(self, numeric_cols: List[str]):
        self.numeric_cols = numeric_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col].astype(str), errors='coerce')
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

class BalancedFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector with built-in imbalance handling and leakage prevention."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._validate_config(config)
        self.selected_features = None
        self.feature_scores = None
        self.preprocessor = None
        self.engineered_features = []
        self.is_fitted = False
        self.sampler_ = None
        self.feature_names_out_ = None
        self.original_feature_names_ = None
        self.logger = logging.getLogger(__name__) 

    def _validate_config(self, config: Dict) -> Dict:
        defaults = {
            "target_column": "is_prospect",
            "must_include_features": [
                "physical_ratio",
                "wage_value_ratio",
                "skill_index",
                "defense_index"
            ],
            "protected_features": [
                "height_cm", "weight_kg", "value_eur"
            ],
            "numeric_cols": [
                "age", "height_cm", "weight_kg", "value_eur", 
                "international_reputation", "weak_foot", "skill_moves",
                "pace", "shooting", "passing", "dribbling", "defending", "physic"
            ],
            "categorical_cols": [],
            "exclude_cols": [
                "sofifa_id", "short_name", "player_positions",
                "long_name", "year", "potential", "overall"
            ],
            "target_feature_count": 40,
            "min_features_to_select": 35,
            "selection_strategy": "hybrid_balanced",
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "prospect_max_age": 24,
            "prospect_threshold": 5,
            "sampling_strategy": "auto",
            "smote_k_neighbors": 5,
            "undersample_ratio": 0.5,
            "target_ratio": 0.5
        }
        return {**defaults, **(config or {})}

    def _get_feature_names_out(self):
        """Get feature names after preprocessing"""
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            return self.preprocessor.get_feature_names_out()
        return self.original_feature_names_

    def _initialize_sampler(self, y: pd.Series):
        n_minority = Counter(y).most_common()[-1][1]
        
        smote = SMOTE(
            sampling_strategy=self.config.get("sampling_strategy", "auto"),
            k_neighbors=min(self.config.get("smote_k_neighbors", 5), n_minority - 1),
            random_state=self.config.get("random_state", 42)
        )
        
        self.sampler_ = smote

    def _get_leakproof_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate features that don't leak target information"""
        X = X.copy()
        
        # Physical attributes (safe)
        X['physical_ratio'] = (X['height_cm'] * X['weight_kg']) / 10000
        
        # Skill indices (using only current abilities)
        X['skill_index'] = (X['dribbling'] + X['passing'] + X['shooting']) / 3
        X['defense_index'] = (X['defending'] + X['physic']) / 2
        X['attack_defense_ratio'] = X['skill_index'] / (X['defense_index'] + 0.1)
        
        # Value efficiency (current value vs skills)
        X['value_efficiency'] = X['skill_index'] / (X['value_eur'] + 1000)
        
        return X

    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        try:
            # Get importance without resampling to prevent data leakage
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=self.config["random_state"],
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X, y)
            importance = pd.Series(model.feature_importances_, index=X.columns)

            # Get mutual information scores
            mi_scores = mutual_info_classif(
                X, y,
                random_state=self.config["random_state"],
                n_neighbors=5
            )
            importance_mi = pd.Series(mi_scores, index=X.columns)

            # Boost importance of engineered features
            for feat in self.engineered_features:
                if feat in importance.index:
                    importance[feat] *= 1.5
                    importance_mi[feat] *= 1.5

            # Combine scores with weighted ranking
            combined_scores = (importance.rank() * 0.6 + importance_mi.rank() * 0.4).rank()
            return combined_scores.sort_values(ascending=False)

        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}", exc_info=True)
            raise

    def _filter_leaky_features(self, features: List[str]) -> List[str]:
        """Remove features that could leak future information"""
        forbidden_substrings = [
            "potential", "progression", "dev_curve", "growth", "development",
            "skill_progression", "mental_composite", "technical_composite",
            "physical_composite", "composite"
        ]
        
        def is_leaky(feat: str) -> bool:
            # Remove prefix for checking
            clean_feat = feat.replace('remainder__', '').replace('num__', '')
            return any(keyword in clean_feat.lower() for keyword in forbidden_substrings)
        
        return [f for f in features if not is_leaky(f)]

    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        # First get all candidate features by importance
        candidate_features = list(self.feature_scores.index)
        
        # Filter out leaky features
        safe_features = self._filter_leaky_features(candidate_features)
        
        # Get protected features (must include)
        protected_raw = set(
            self.config.get("must_include_features", []) +
            self.config.get("protected_features", []) +
            self.engineered_features
        )
        
        # Match protected features in safe features
        protected = [
            col for col in safe_features 
            if col.replace('remainder__', '').replace('num__', '') in protected_raw
        ]
        
        target_features = min(self.config.get("target_feature_count", 30), len(safe_features))
        min_features = max(self.config.get("min_features_to_select", 20), len(protected))
        
        if target_features < min_features:
            target_features = min_features
        
        # Select top non-protected features
        remaining_features = [f for f in safe_features if f not in protected]
        top_features = [
            f for f in remaining_features 
            if f in self.feature_scores.index
        ][:target_features - len(protected)]
        
        selected = protected + top_features
        
        # Final validation of selected features
        final_selected = self._filter_leaky_features(selected)
        self.logger.info(f"\nFinal selected features (no potential-related leakage): {final_selected}")
        
        # Save the final selected features
        with open("selected_features.pkl", "wb") as f:
            pickle.dump(final_selected, f)
            
        return final_selected

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        try:
            self._initialize_sampler(y_train)
            X_filtered = X_train.drop(columns=self.config["exclude_cols"], errors='ignore')
            X_leakproof = self._get_leakproof_features(X_filtered)
            
            self.engineered_features = [
                'physical_ratio', 
                'skill_index', 
                'defense_index', 
                'attack_defense_ratio',
                'value_efficiency'
            ]
            
            full_numeric_cols = list(set(
                [col for col in self.config["numeric_cols"] if col not in self.config["exclude_cols"]] +
                self.engineered_features
            ))
            
            self.original_feature_names_ = list(X_leakproof.columns)
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('cleaner', FeatureCleaner(full_numeric_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), full_numeric_cols)
                ],
                remainder='passthrough'
            )
            
            X_processed = self.preprocessor.fit_transform(X_leakproof)
            self.feature_names_out_ = self._get_feature_names_out()
            
            X_processed_df = pd.DataFrame(
                X_processed,
                columns=self.feature_names_out_,
                index=X_leakproof.index
            )
            
            self.feature_scores = self._get_feature_importance(X_processed_df, y_train)
            self.selected_features = self._perform_feature_selection(X_processed_df, y_train)
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Feature selection fitting failed: {str(e)}", exc_info=True)
            raise

        return self

    def transform(self, X: pd.DataFrame, apply_smote: bool = False, y: Optional[pd.Series] = None):
        if not self.is_fitted:
            raise RuntimeError("Must fit before transforming")
    
        try:
            if self.config["target_column"] in X.columns:
                self.logger.warning(f"Dropping target column '{self.config['target_column']}' from input data to prevent leakage.")
                X = X.drop(columns=[self.config["target_column"]])
    
            X_filtered = X.drop(columns=self.config["exclude_cols"], errors='ignore')
            X_leakproof = self._get_leakproof_features(X_filtered)
    
            X_processed = self.preprocessor.transform(X_leakproof)
    
            X_processed_df = pd.DataFrame(
                X_processed,
                columns=self.feature_names_out_,
                index=X_leakproof.index
            )
    
            available_features = [f for f in self.selected_features if f in X_processed_df.columns]
    
            if not available_features:
                raise ValueError("No selected features available in transformed data")
            
            result = X_processed_df[available_features]
            
            if apply_smote and y is not None and self.sampler_ is not None:
                X_res, y_res = self.sampler_.fit_resample(result, y)
                self.logger.info(f"Applied SMOTE - Before: {Counter(y)} After: {Counter(y_res)}")
                return X_res, y_res

            
            
            return result
    
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}", exc_info=True)
            raise

class FeatureEngineeringPipeline:
    """Main pipeline class for handling feature engineering process."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.selector = None
        self.train_path = None
        self.test_path = None
        self.output_dir = None

    def _setup_output_directory(self, output_dir: Optional[str] = None):
        """Ensure output directory exists"""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path.cwd()

    def _load_data(self, train_path: str, test_path: Optional[str] = None):
        """Load training and test data"""
        self.train_path = Path(train_path)
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training file not found at: {self.train_path}")
            
        train_df = pd.read_feather(self.train_path)
        self.logger.info(f"Loaded training data from {self.train_path}")

        test_df = None
        if test_path:
            self.test_path = Path(test_path)
            if self.test_path.exists():
                test_df = pd.read_feather(self.test_path)
                self.logger.info(f"Loaded test data from {self.test_path}")
            else:
                self.logger.warning(f"Test file not found at: {self.test_path}")

        return train_df, test_df

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create is_prospect column if it doesn't exist"""
        if 'is_prospect' not in df.columns:
            max_age = self.config.get('prospect_max_age', 24)
            min_rep = self.config.get('prospect_threshold', 3)
            
            self.logger.info(f"Creating is_prospect column with thresholds: age<={max_age}, rep>={min_rep}")
            df['is_prospect'] = (
                (df['age'] <= max_age) &
                (df['international_reputation'] >= min_rep)
            ).astype(int)
        return df

    def _validate_data(self, train_df: pd.DataFrame):
        """Validate input data meets requirements"""
        if train_df.empty:
            raise ValueError("Training data is empty")
            
        if 'is_prospect' not in train_df.columns:
            raise ValueError("Target column 'is_prospect' not found and couldn't be created")
            
        positive_count = train_df['is_prospect'].sum()
        if positive_count == 0:
            raise ValueError("No positive examples in training data - adjust threshold or age filter")
            
        self.logger.info(f"{positive_count}/{len(train_df)} players marked as prospects (ratio: {positive_count/len(train_df):.3f})")

    def _process_data(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None):
        """Main processing logic with proper SMOTE application"""
        y_train = train_df['is_prospect']
        X_train = train_df.drop(columns=['is_prospect'])
    
        # Initialize and fit selector
        self.selector = BalancedFeatureSelector(config=self.config)
        self.selector.fit(X_train, y_train)
    
        # Transform training data with SMOTE
        X_train_selected, y_train_resampled = self.selector.transform(
            X_train, 
            apply_smote=True, 
            y=y_train
        )
    
        # Process test data if available (without SMOTE)
        X_test_selected = None
        if test_df is not None:
            y_test = test_df.get("is_prospect")  # get target if present
            X_test_selected = self.selector.transform(test_df)
    
            if y_test is not None:
                # Combine features and target for test set (preserve labels for evaluation)
                output_test = pd.concat([X_test_selected, y_test.rename("is_prospect")], axis=1)
            else:
                output_test = X_test_selected
        else:
            output_test = None
    
        return {
            "train": pd.concat([X_train_selected, y_train_resampled.rename("is_prospect")], axis=1),
            "test": output_test
        }


    def _save_output(self, output: Dict, output_dir: Optional[str] = None):
        """Save processed data to files"""
        self._setup_output_directory(output_dir)
        
        # Save training data
        train_output_path = self.output_dir / "processed_train.feather"
        output["train"].to_feather(train_output_path)
        self.logger.info(f"Saved processed training data to {train_output_path}")

        # Save test data if available
        if output["test"] is not None:
            test_output_path = self.output_dir / "processed_test.feather"
            output["test"].to_feather(test_output_path)
            self.logger.info(f"Saved processed test data to {test_output_path}")

        # Save selected features
        features_path = self.output_dir / "selected_features.pkl"
        with open(features_path, "wb") as f:
            pickle.dump(self.selector.selected_features, f)
        self.logger.info(f"Saved selected features to {features_path}")

    def run(self, train_path: str, test_path: Optional[str] = None, output_dir: Optional[str] = None):
        """Main execution method"""
        try:
            # Load and validate data
            train_df, test_df = self._load_data(train_path, test_path)
            train_df = self._create_target_variable(train_df)
            self._validate_data(train_df)

            # Process data
            processed_data = self._process_data(train_df, test_df)

            # Save results
            self._save_output(processed_data, output_dir)

            return processed_data

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

def run_balanced_pipeline(
    train_df: Optional[pd.DataFrame] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None
):
    """
    Entry point for the feature engineering pipeline.
    
    Args:
        train_df: Optional pre-loaded training DataFrame
        train_path: Path to training data if not pre-loaded
        test_path: Path to test data (optional)
        output_dir: Directory to save processed data
        config: Configuration dictionary
    
    Returns:
        Dictionary with processed train and test DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting leakage-proof feature engineering pipeline...")

    try:
        pipeline = FeatureEngineeringPipeline(config=config)
        
        if train_df is not None:
            # Handle case where train_df is provided directly
            train_df = pipeline._create_target_variable(train_df.copy())
            pipeline._validate_data(train_df)
            
            # Create temporary paths for processing
            temp_train_path = Path("temp_train.feather")
            train_df.to_feather(temp_train_path)
            
            # Run pipeline with temporary path
            result = pipeline.run(
                train_path=str(temp_train_path),
                test_path=test_path,
                output_dir=output_dir
            )
            
            # Clean up temporary file
            temp_train_path.unlink(missing_ok=True)
        else:
            # Normal case with file paths
            result = pipeline.run(
                train_path=train_path,
                test_path=test_path,
                output_dir=output_dir
            )
            
        logger.info("Pipeline completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise