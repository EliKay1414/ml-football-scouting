# evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           classification_report, confusion_matrix,
                           precision_recall_curve, roc_curve)
import shap
import pickle
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProspectModelEvaluator:
    """Comprehensive evaluation pipeline for football prospect models."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize evaluator with configuration.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.test_data = None
        self.best_model = None
        self.explainer = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Set default paths
            paths = config.setdefault("paths", {})
            paths.setdefault("models_dir", "models/prospect_models_v1")
            paths.setdefault("test_data", "data/features/test_features.csv")
            paths.setdefault("new_data", "data/new_players.csv")
            paths.setdefault("output_dir", "evaluation_results")
            paths.setdefault("deployment_dir", "models/deployment")
            
            # Set default metrics
            metrics = config.setdefault("metrics", {})
            metrics.setdefault("threshold", 0.5)
            metrics.setdefault("positive_class", 1)
            metrics.setdefault("top_n_features", 15)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def load_models(self) -> Dict:
        """Load all trained models from specified directory."""
        models_dir = Path(self.config["paths"]["models_dir"])
        self.models = {}
        
        for model_file in models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {str(e)}")
        
        if not self.models:
            raise ValueError("No models found in specified directory")
        return self.models

    def load_test_data(self):
        test_features_path = Path(self.config["paths"]["test_features"])
        test_labels_path = Path(self.config["paths"]["test_labels"])
        
        # Check if test features file exists
        if not test_features_path.exists():
            raise FileNotFoundError(f" Test features file not found at: {test_features_path.resolve()}")
        
        # Check if test labels file exists
        if not test_labels_path.exists():
            raise FileNotFoundError(f" Test labels file not found at: {test_labels_path.resolve()}")
        
        # Load features and labels
        X_test = pd.read_feather(test_features_path)

        y_test_df = pd.read_csv(test_labels_path)
        
        # Check if 'is_prospect' column exists
        if "is_prospect" not in y_test_df.columns:
            raise ValueError(f" 'is_prospect' column not found in test labels file: {test_labels_path.resolve()}")
        
        y_test = y_test_df["is_prospect"]  # Extract the target column
        
        return X_test, y_test


    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all models on test set and return metrics."""
        self.results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Calculate metrics
                metrics = {
                    "accuracy": np.mean(y_pred == y_test),
                    "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                    "avg_precision": average_precision_score(y_test, y_proba) if y_proba is not None else None,
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "calibration": self._calculate_calibration(model, X_test, y_test) if y_proba is not None else None
                }
                
                self.results[model_name] = metrics
                logger.info(f"Completed evaluation for {model_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                self.results[model_name] = {"error": str(e)}
        
        return self.results

    def _calculate_calibration(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Calculate model calibration metrics."""
        prob_true, prob_pred = calibration_curve(
            y_test, 
            model.predict_proba(X_test)[:, 1],
            n_bins=10
        )
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }

    def compare_models(self) -> pd.DataFrame:
        """Compare model performance and return sorted results."""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")
            
        comparison = []
        for model_name, metrics in self.results.items():
            if "error" in metrics:
                comparison.append({
                    "model": model_name,
                    "error": metrics["error"]
                })
            else:
                comparison.append({
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "avg_precision": metrics["avg_precision"],
                    "precision": metrics["classification_report"]["1"]["precision"],
                    "recall": metrics["classification_report"]["1"]["recall"],
                    "f1_score": metrics["classification_report"]["1"]["f1-score"]
                })
        
        return pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)

    def analyze_feature_importance(self, model_name: str, X_test: pd.DataFrame) -> pd.DataFrame:
        """Analyze and return feature importance for a specific model."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            model = model.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X_test.columns
            
            return pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return pd.DataFrame()

    def explain_with_shap(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        sample_size: int = 100,
        plot: bool = True
    ) -> Optional[np.ndarray]:
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in self.models")
        if hasattr(model, "named_steps"):
            model = model.named_steps.get("model", list(model.named_steps.values())[-1])
        if len(X_test) > sample_size:
            X_sample = resample(X_test, n_samples=sample_size, random_state=42)
        else:
            X_sample = X_test.copy()
        try:
            if hasattr(model, "feature_importances_") or "xgboost" in model.__module__.lower():
                explainer = shap.TreeExplainer(model)
            else:
                background = shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
    
            shap_values = explainer.shap_values(X_sample)
    
            if plot:
                values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
                shap.summary_plot(values_to_plot, X_sample, show=True)
    
            return shap_values
    
        except Exception as exc:
            logger.error("SHAP explanation failed for '%s': %s", model_name, exc)
            return None

    def _get_metrics_array(self) -> Tuple[np.ndarray, List[str]]:
        """Convert evaluation results into a numpy array of metrics."""
        valid_models = {
            k: v for k, v in self.results.items() 
            if "error" not in v and isinstance(v, dict)
        }
        
        # Define metrics and accessors
        metrics_config = [
            ('roc_auc', lambda x: x['roc_auc']),
            ('avg_precision', lambda x: x['avg_precision']),
            ('f1_score', lambda x: x['classification_report']['1']['f1-score']),
            ('recall', lambda x: x['classification_report']['1']['recall']),
            ('precision', lambda x: x['classification_report']['1']['precision']),
            ('accuracy', lambda x: x['accuracy'])
        ]
        
        # Create scores matrix
        scores = []
        model_names = []
        for model_name, metrics in valid_models.items():
            model_names.append(model_name)
            row = []
            for metric, accessor in metrics_config:
                try:
                    value = accessor(metrics)
                    row.append(value if value is not None else 0)
                except (KeyError, TypeError):
                    row.append(0)  # Default value if metric missing
            scores.append(row)
        
        return np.array(scores), model_names
    
    def _is_pareto_efficient(self, scores: np.ndarray) -> np.ndarray:
        """Identify Pareto-efficient points (non-dominated) from a set of multi-metric scores."""
        n_points = scores.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_efficient[i]:
                # Keep i if no other point dominates it
                domination = np.all(scores >= scores[i], axis=1) & np.any(scores > scores[i], axis=1)
                is_efficient[domination] = False
                is_efficient[i] = True  # Keep self
        return is_efficient

    
    def get_pareto_frontier(self) -> pd.DataFrame:
        """
        Return DataFrame of models with Pareto-efficient performance.
        Identifies models that are not dominated in all metrics.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")
        
        # Get scores and model names
        scores, model_names = self._get_metrics_array()
        
        # Identify Pareto-efficient models
        is_efficient = self._is_pareto_efficient(scores)
        pareto_models = [model_names[i] for i, flag in enumerate(is_efficient) if flag]
        
        # Get the comparison DataFrame
        comparison_df = self.compare_models()
        
        # Add Pareto efficiency flag
        comparison_df["is_pareto"] = comparison_df["model"].isin(pareto_models)
        
        return comparison_df
    
    def plot_pareto_frontier(self, 
                            x_metric: str = "recall",
                            y_metric: str = "precision",
                            output_dir: Optional[Path] = None) -> Path:
        """Generate and save a Pareto frontier plot."""
        if not output_dir:
            output_dir = Path(self.config["paths"]["output_dir"])
        
        df = self.get_pareto_frontier()
        
        plt.figure(figsize=(10, 6))
        
        # Create colormap and sizes
        colors = ['green' if x else 'gray' for x in df['is_pareto']]
        sizes = [120 if x else 80 for x in df['is_pareto']]
        
        # Plot all models
        plt.scatter(
            x=df[x_metric],
            y=df[y_metric],
            c=colors,
            s=sizes,
            alpha=0.7
        )
        
        # Highlight Pareto frontier
        pareto_df = df[df['is_pareto']].sort_values(x_metric)
        plt.plot(
            pareto_df[x_metric],
            pareto_df[y_metric],
            'g--',
            alpha=0.5,
            label='Pareto Frontier'
        )
        
        # Add labels for Pareto models
        for _, row in pareto_df.iterrows():
            plt.text(
                row[x_metric] + 0.01,
                row[y_metric] + 0.01,
                row['model'],
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        plt.title(f"Pareto Frontier: {y_metric} vs {x_metric}")
        plt.xlabel(f"{x_metric} (Higher is better)")
        plt.ylabel(f"{y_metric} (Higher is better)")
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.scatter([], [], c='green', label='Pareto Optimal')
        plt.scatter([], [], c='gray', label='Other Models')
        plt.legend()
        
        plot_path = output_dir / f"pareto_{x_metric}_vs_{y_metric}.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path
    
    def select_best_model(self, 
                          priority_metrics: List[str] = ["roc_auc", "f1_score", "precision", "recall"],
                          weights: Optional[Dict[str, float]] = None,
                          min_recall: Optional[float] = None,
                          min_precision: Optional[float] = None) -> str:
    
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")
            
        # Get comparison data
        comparison_df = self.compare_models()
        
        # Filter by minimum recall if specified
        if min_recall is not None:
            comparison_df = comparison_df[comparison_df['recall'] >= min_recall]
            if comparison_df.empty:
                logger.warning(f"No models meet minimum recall of {min_recall}")
                return self._select_by_weighted_score()
        
        # Filter by minimum precision if specified
        if min_precision is not None:
            comparison_df = comparison_df[comparison_df['precision'] >= min_precision]
            if comparison_df.empty:
                logger.warning(f"No models meet minimum precision of {min_precision}")
                return self._select_by_weighted_score()
        
        # Set default weights if not provided
        if weights is None:
            weights = {metric: 1.0 for metric in priority_metrics}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Normalize metrics (0-1 scaling)
        normalized_df = comparison_df.copy()
        for metric in priority_metrics:
            col = normalized_df[metric]
            normalized_df[metric] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        # Calculate composite score
        for metric, weight in weights.items():
            normalized_df[metric] = normalized_df[metric] * weight
        
        normalized_df['composite_score'] = normalized_df[list(weights.keys())].sum(axis=1)
        
        # Select model with highest composite score
        best_model = normalized_df.loc[normalized_df['composite_score'].idxmax(), 'model']
        
        self.best_model = {
            "name": best_model,
            "model": self.models[best_model],
            "metrics": self.results[best_model],
            "selection_method": f"weighted_composite({','.join(priority_metrics)})"
        }
        
        logger.info(f"Selected best model: {best_model} with composite score {normalized_df['composite_score'].max():.3f}")
        return best_model

    def _select_by_weighted_score(self,
                                  weights: Optional[Dict[str, float]] = None,
                                  normalize: bool = True) -> str:
        """Fallback model chooser using a weighted composite score."""
        if weights is None:
            weights = {'roc_auc': 0.3, 'f1_score': 0.3, 'accuracy': 0.0,
                       'recall': 0.2, 'precision': 0.2}
        
        # Normalize weights so they sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    
        comparison_df = self.compare_models()
    
        # Optional metric normalization (0‑1 min‑max) to keep scales comparable
        if normalize:
            for m in weights:
                col = comparison_df[m]
                comparison_df[m] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    
        # Apply weights
        for metric, w in weights.items():
            comparison_df[metric] = comparison_df[metric] * w
    
        comparison_df['weighted_score'] = comparison_df[list(weights)].sum(axis=1)
        best_model = comparison_df.loc[comparison_df['weighted_score'].idxmax(), 'model']
    
        self.best_model = {
            "name": best_model,
            "model": self.models[best_model],
            "metrics": self.results[best_model],
            "selection_method": "weighted_score"
        }
        return best_model

    def find_optimal_thresholds(self,
                              min_precision: float = 0.5,
                              min_recall: float = 0.5) -> Dict[str, Dict]:
        if self.test_data is None or "is_prospect" not in self.test_data.columns:
            raise ValueError("Test data not loaded or missing target column 'is_prospect'")
        
        thresholds = {}
        X_test = self.test_data.drop(columns=["is_prospect"])
        y_test = self.test_data["is_prospect"]
        
        for model_name, model in self.models.items():
            try:
                # Skip models without predict_proba
                if not hasattr(model, "predict_proba"):
                    logger.warning(f"Skipping {model_name} - no predict_proba method")
                    thresholds[model_name] = None
                    continue
                    
                # Get predicted probabilities
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate precision-recall curve
                precision, recall, thresholds_candidates = precision_recall_curve(y_test, y_proba)
                thresholds_candidates = np.append(thresholds_candidates, 1)  # Add threshold=1
                
                # Find thresholds meeting both constraints
                valid_indices = np.where(
                    (precision[:-1] >= min_precision) & 
                    (recall[:-1] >= min_recall)
                )[0]
                
                if len(valid_indices) > 0:
                    # Select threshold with highest recall
                    best_idx = valid_indices[np.argmax(recall[valid_indices])]
                    thresholds[model_name] = {
                        "threshold": float(thresholds_candidates[best_idx]),
                        "precision": float(precision[best_idx]),
                        "recall": float(recall[best_idx])
                    }
                else:
                    logger.warning(f"No threshold meets requirements for {model_name}")
                    thresholds[model_name] = None
                    
            except Exception as e:
                logger.error(f"Error finding threshold for {model_name}: {str(e)}")
                thresholds[model_name] = None
        
        return thresholds
    
    def save_best_model(self) -> Path:
        """
        Save the best model for deployment with comprehensive metadata.
        Returns the path to the saved model file.
        """
        if not self.best_model:
            raise ValueError("No best model selected. Run select_best_model() first.")
            
        try:
            deployment_dir = Path(self.config["paths"]["deployment_dir"])
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for versioning
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"best_model_{timestamp}.pkl"
            model_path = deployment_dir / model_filename
            
            # Save the model with compression
            joblib.dump(
                self.best_model["model"],
                model_path,
                compress=3,
                protocol=pickle.HIGHEST_PROTOCOL
            )
            
            # Prepare metadata
            metadata = {
                "model_info": {
                    "name": self.best_model["name"],
                    "type": type(self.best_model["model"]).__name__,
                    "selection_method": self.best_model.get("selection_method", "unknown"),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "version": timestamp
                },
                "performance_metrics": self.best_model["metrics"],
                "config": self.config
            }
            
            # Save metadata
            metadata_path = deployment_dir / f"model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Try to create symlink (optional)
            latest_path = deployment_dir / "latest_model.pkl"
            try:
                if latest_path.exists():
                    latest_path.unlink()
                latest_path.symlink_to(model_path.name)
                logger.info(f"Created symlink: {latest_path}")
            except OSError as e:
                logger.warning(f"Could not create symlink (admin privileges may be needed): {str(e)}")
                # Fallback - just copy the file
                try:
                    import shutil
                    shutil.copy2(model_path, latest_path)
                    logger.info(f"Created copy as fallback: {latest_path}")
                except Exception as copy_error:
                    logger.warning(f"Could not create fallback copy either: {str(copy_error)}")
            
            logger.info(
                f"Successfully saved best model '{self.best_model['name']}'\n"
                f"Model path: {model_path}\n"
                f"Metadata: {metadata_path}"
            )
            
            return model_path
            
        except Exception as e:
            error_msg = f"Failed to save best model: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        if not self.results:
            raise ValueError("No evaluation results available")
            
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison table
        comparison_df = self.compare_models()
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info(f"Generated report in {output_dir}")

    def _generate_visualizations(self) -> None:
        """Generate all evaluation visualizations."""
        output_dir = Path(self.config["paths"]["output_dir"])
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for model_name, metrics in self.results.items():
            if "roc_auc" in metrics and metrics["roc_auc"] is not None:
                fpr, tpr, _ = roc_curve(
                    self.test_data["is_prospect"],
                    self.models[model_name].predict_proba(self.test_data.drop(columns=["is_prospect"]))[:, 1]
                )
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.savefig(output_dir / "roc_curves.png")
        plt.close()
    
        # Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for model_name, metrics in self.results.items():
            if "avg_precision" in metrics and metrics["avg_precision"] is not None:
                precision, recall, _ = precision_recall_curve(
                    self.test_data["is_prospect"],
                    self.models[model_name].predict_proba(self.test_data.drop(columns=["is_prospect"]))[:, 1]
                )
                plt.plot(recall, precision, label=f"{model_name} (AP = {metrics['avg_precision']:.2f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc='upper right')
        plt.savefig(output_dir / "precision_recall_curves.png")
        plt.close()
    
        # Confusion Matrices
        self._plot_confusion_matrices(output_dir)  

    def _plot_confusion_matrices(self, output_dir: Path) -> None:
        """Generate and save confusion matrix plots for all models."""
        plt.figure(figsize=(12, 8))
        
        # Determine grid size based on number of models
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        if n_models == 1:
            axes = np.array([axes])  # Ensure axes is always array-like
        axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            ax = axes[i]
            
            if "confusion_matrix" not in metrics:
                continue
                
            cm = np.array(metrics["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, annot_kws={"size": 14})
            
            ax.set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.2f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Non-Prospect', 'Prospect'])
            ax.set_yticklabels(['Non-Prospect', 'Prospect'])
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices.png", bbox_inches='tight', dpi=300)
        plt.close()

def rank_models_by_each_metric(evaluator: ProspectModelEvaluator) -> dict:
    """
    Print and return the best model for every KPI.
    """
    df = evaluator.compare_models()
    metrics = ["roc_auc", "f1_score", "precision", "recall", "accuracy", "avg_precision"]
    best = {}

    print("\n Best model for each metric")
    for metric in metrics:
        row = df.loc[df[metric].idxmax()]
        best[metric] = row["model"]
        print(f"  • {metric:<14}: {row['model']}  (score = {row[metric]:.4f})")

    return best


def rank_models_by_weighted_score(evaluator: ProspectModelEvaluator,
                                  weights: dict | None = None) -> pd.DataFrame:
    """
    Return a DataFrame ranked by a weighted sum of metrics
    and print the overall winner.
    """
    if weights is None:
        weights = {"roc_auc": 0.3, "f1_score": 0.3, "recall": 0.2, "precision": 0.2}

    df = evaluator.compare_models().copy()
    for m, w in weights.items():
        df[m] = df[m] * w
    df["weighted_score"] = df[list(weights.keys())].sum(axis=1)
    df = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)

    print("\n Overall ranking by weighted score")
    print(df[["model", "weighted_score"] + list(weights.keys())].to_markdown(index=False))
    print(f"\n Overall best model: {df.loc[0, 'model']}")
    return df


def run_full_evaluation(config_path: str = "config.json") -> None:
    """Run complete evaluation pipeline."""
    evaluator = ProspectModelEvaluator(config_path)
    
    try:
        # 1. Load models and test data
        evaluator.load_models()
        X_test, y_test = evaluator.load_test_data()
        
        # 2. Evaluate models
        evaluator.evaluate_models(X_test, y_test)
        
        # 3. Model comparison
        comparison = evaluator.compare_models()
        print("\nModel Comparison:")
        print(comparison.to_markdown())
        
        # Print rankings by each metric
        rank_models_by_each_metric(evaluator)
        
        # 4. Select best model using multiple metrics
        best_model = evaluator.select_best_model(
            priority_metrics=["roc_auc", "f1_score", "precision", "recall"],
            weights={"roc_auc": 0.3, "f1_score": 0.3, "precision": 0.2, "recall": 0.2}
        )
        
        # 5. Generate comprehensive report
        evaluator.generate_report()
        
        # 6. Save best model
        evaluator.save_best_model()
        
        # 7. Demo on new data
        print("\n Evaluation pipeline completed successfully!")

    except Exception as e:
        print(f" An error occurred during evaluation: {e}")

if __name__ == "__main__":
    run_full_evaluation()
