import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

class ProspectModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.features = None
        self.feature_importances = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        import pickle
    
        train_df = pd.read_feather(self.config["paths"]["features_train"])
        test_df = pd.read_feather(self.config["paths"]["features_test"])
    
        try:
            with open(self.config["paths"]["selected_features"], 'rb') as f:
                self.features = pickle.load(f)
            print(f"✅ Loaded {len(self.features)} selected features.")
        except Exception as e:
            print(f"Could not load selected features from .pkl: {e}")
            self.features = [col for col in train_df.columns if col != 'is_prospect']
            print(f"Falling back to all features: {len(self.features)}")
    
        X = train_df[self.features]
        y = train_df["is_prospect"]
    
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.config["training"]["random_state"],
            stratify=y
        )
    
        X_test = test_df[self.features]
        y_test = test_df["is_prospect"]
    
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_models(self, X_train, y_train):
        """Initialize and return a dictionary of models wrapped in preprocessing and optional SMOTE pipelines."""
        models = {}
        
        # Class imbalance info
        class_counts = y_train.value_counts()
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        imbalance_ratio = majority_count / minority_count
        
        print(f"\nClass Distribution:\n{class_counts}")
        print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
        # Separate numeric and categorical features for preprocessing
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        def make_resampling_pipeline(model):
            """Wrap model with preprocessing and optional SMOTE handling."""
            # Skip SMOTE if minority class is too small or data is nearly balanced
            if minority_count < 5:
                print(f" Not enough minority samples ({minority_count}) for SMOTE")
                return Pipeline([
                    ('preprocessing', preprocessor),
                    ('model', model)
                ])
            if imbalance_ratio <= 1.1:
                print(f"ℹ️ Data nearly balanced (ratio={imbalance_ratio:.2f}), skipping SMOTE")
                return Pipeline([
                    ('preprocessing', preprocessor),
                    ('model', model)
                ])
            
            # Configure SMOTE parameters safely
            k_neighbors = min(5, minority_count - 1)
            sampling_strategy = min(0.5, (minority_count / majority_count) * 2)
            
            print(f"Using SMOTE with k_neighbors={k_neighbors}, sampling_strategy={sampling_strategy:.2f}")
            
            return ImbPipeline([
                ('preprocessing', preprocessor),
                ('smote', SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=self.config["training"]["random_state"]
                )),
                ('model', model)
            ])

        # Initialize models with parameters from config
        models['XGBoost'] = make_resampling_pipeline(XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            use_label_encoder=False,
            **self.config["models"]["XGBoost"]  
        ))
        
        models['LightGBM'] = make_resampling_pipeline(LGBMClassifier(
            is_unbalance=True,
            **self.config["models"]["LightGBM"]
        ))
        
        models['RandomForest'] = make_resampling_pipeline(RandomForestClassifier(
            class_weight='balanced',
            **self.config["models"]["RandomForest"]
        ))
        
        models['HistGradientBoosting'] = make_resampling_pipeline(
            HistGradientBoostingClassifier(**self.config["models"]["HistGradientBoosting"])
        )
        
        models['LogisticRegression'] = make_resampling_pipeline(LogisticRegression(
            class_weight='balanced',
            **self.config["models"]["LogisticRegression"]
        ))
        
        svm_config = self.config["models"]["SVM"].copy()
        svm_config.pop("probability", None)  # will set explicitly
        
        models['SVM'] = make_resampling_pipeline(SVC(
            class_weight='balanced',
            probability=True,
            **svm_config
        ))
        
        models['AdaBoost'] = make_resampling_pipeline(AdaBoostClassifier(
            **self.config["models"]["AdaBoost"]
        ))
        
        return models
    

    def train_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.models = self.get_models(X_train, y_train)
    
        # Define hyperparameter grids for tuning per model
        param_grids = {
            'XGBoost': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [4, 6],
                'model__learning_rate': [0.05, 0.1]
            },
            'LightGBM': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [4, 6],
                'model__learning_rate': [0.05, 0.1]
            },
            'RandomForest': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [6, 8],
                'model__min_samples_split': [2, 5]
            },
            'HistGradientBoosting': {
                'model__max_iter': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1]
            },
            'LogisticRegression': {
                'model__C': [0.1, 1, 10],
                'model__l1_ratio': [0.0, 0.5, 1.0]  # l1_ratio relevant for elasticnet penalty
            },
            'SVM': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf']
            },
            'AdaBoost': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.05, 0.1]
            }
        }
    
        for name, pipeline in self.models.items():
            print(f"\nTuning and training {name}...")
    
            param_grid = param_grids.get(name, None)
    
            if param_grid:
                # Setup cross-validation for tuning
                cv = StratifiedKFold(n_splits=self.config["training"]["cv_folds"], shuffle=True, random_state=self.config["training"]["random_state"])
    
                # Initialize GridSearchCV with ROC AUC scoring
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                    refit=True
                )
    
                # Fit GridSearchCV (this will do CV internally)
                grid_search.fit(X_train, y_train)
    
                print(f"Best params for {name}: {grid_search.best_params_}")
                print(f"Best CV AUC for {name}: {grid_search.best_score_:.4f}")
    
                # Save best model from GridSearchCV
                best_model = grid_search.best_estimator_
    
                # Store results
                self.results[name] = {
                    'model': best_model,
                    'cv_mean_auc': grid_search.best_score_,
                    'cv_std_auc': np.nan  # std not directly available here
                }
    
                # Replace model with best model for future use
                self.models[name] = best_model
    
                # Store feature importances if possible
                if hasattr(X_train, "columns"):
                    try:
                        self._store_feature_importances(name, best_model, X_train.columns)
                    except Exception as e:
                        print(f"Could not store feature importances for {name}: {e}")
    
            else:
                # If no param grid, just train normally
                try:
                    pipeline.fit(X_train, y_train)
                    y_pred_proba = pipeline.predict_proba(X_train)[:, 1]
                    auc_score = roc_auc_score(y_train, y_pred_proba)
    
                    self.results[name] = {
                        'model': pipeline,
                        'cv_mean_auc': auc_score,
                        'cv_std_auc': np.nan
                    }
                    self.models[name] = pipeline
                    if hasattr(X_train, "columns"):
                        self._store_feature_importances(name, pipeline, X_train.columns)
                except Exception as e:
                    print(f"Error training {name}: {e}")
    
        return self.results

    def _store_feature_importances(self, model_name, model, feature_names):
        """Extract and store feature importances or coefficients if available."""
        try:
            # If model is a pipeline, get the last step (estimator)
            if isinstance(model, (Pipeline, ImbPipeline)):
                final_estimator = model.steps[-1][1]
            else:
                final_estimator = model
    
            if hasattr(final_estimator, 'feature_importances_'):
                importances = final_estimator.feature_importances_
            elif hasattr(final_estimator, 'coef_'):
                importances = np.abs(final_estimator.coef_).flatten()
            else:
                # No importances available
                return
    
            self.feature_importances[model_name] = pd.Series(
                importances,
                index=feature_names
            ).sort_values(ascending=False)
        except Exception as e:
            print(f"Could not get feature importances for {model_name}: {e}")
            
    def save_models_and_artifacts(self):
        """Save trained models, feature importances, and training results."""
        output_dir = Path(self.config["paths"]["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = output_dir / f"{name.replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
        
        # Save feature importances
        for name, importance in self.feature_importances.items():
            importance_path = output_dir / f"{name.replace(' ', '_')}_feature_importance.csv"
            importance.to_csv(importance_path)
        
        # Save training summary results
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_path = output_dir / "training_results.csv"
        results_df.to_csv(results_path)
        
        print(f"Models and artifacts saved to {output_dir}")
