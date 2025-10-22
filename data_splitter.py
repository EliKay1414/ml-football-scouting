import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
import logging
import joblib
import json
from typing import Optional, Tuple, Dict, List
import yaml

class ProspectDataSplitter:
    
    #  parameters for football prospect evaluation
    SCOUTING_PARAMS = {
        'technical': ['skill_moves', 'weak_foot', 'technical_composite'],
        'physical': ['pace', 'stamina', 'strength', 'physical_composite'],
        'mental': ['composure', 'vision', 'work_rate'],
        'potential': ['potential_growth', 'age_adjusted_potential'],
        'position_specific': {
            'GK': ['goalkeeping_score', 'reflexes'],
            'DEF': ['defensive_score', 'standing_tackle'],
            'MID': ['playmaking_score', 'passing'],
            'ATT': ['attacking_score', 'finishing']
        }
    }
    
    def __init__(self, 
                 test_size: float = 0.2, 
                 val_size: float = 0.2, 
                 random_state: int = 42):
                 
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.feature_columns = None
        self.target_column = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prospect_splitting.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_and_validate(self, processed_path: Path) -> pd.DataFrame:
        """Load and validate prospect data with key scouting parameters"""
        try:
            df = pd.read_csv(processed_path)
            self.logger.info(f"Loaded prospect data with shape {df.shape}")
            
            # Validate scouting parameters
            missing_params = []
            for param_group in self.SCOUTING_PARAMS.values():
                if isinstance(param_group, dict):
                    for params in param_group.values():
                        missing_params.extend(p for p in params if p not in df.columns)
                else:
                    missing_params.extend(p for p in param_group if p not in df.columns)
            
            if missing_params:
                self.logger.warning(f"Missing scouting parameters: {set(missing_params)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise

    def _create_position_strata(self, df: pd.DataFrame) -> pd.Series:
        """
        Create strata for stratified splitting based on player's main position and binned potential.
        """
        df = df.copy()
        
        # Extract main position from 'player_positions' (first listed)
        df['position_group'] = df['player_positions'].str.split(',').str[0]
        
        # Map detailed positions to broad groups
        position_map = {
            'GK': 'GK',
            'CB': 'DEF', 'LB': 'DEF', 'RB': 'DEF',
            'CDM': 'MID', 'CM': 'MID', 'CAM': 'MID',
            'LW': 'ATT', 'RW': 'ATT', 'ST': 'ATT'
        }
        df['position_group'] = df['position_group'].map(position_map).fillna('OTHER')
        
        # Bin potential into quantiles (2 or 3 bins)
        if 'potential_growth' in df.columns:
            potential = df['potential_growth']
            try:
                df['potential_bin'] = pd.qcut(potential, q=3, labels=['low', 'medium', 'high'])
            except ValueError:
                # fallback if qcut fails (not enough unique values)
                median = potential.median()
                df['potential_bin'] = np.where(potential > median, 'high', 'low')
        else:
            df['potential_bin'] = 'unknown'
        
        # Combine for strata
        strata = df['position_group'].astype(str) + '_' + df['potential_bin'].astype(str)
        return strata



    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data with stratification based on position and potential bins."""
        try:
            self.feature_columns = df.columns.tolist()
            
            # Use the new strata creation method
            strata = self._create_position_strata(df)
            
            # First split: train+val vs test
            idx_train_val, idx_test = train_test_split(
                df.index,
                test_size=self.test_size,
                stratify=strata,
                random_state=self.random_state
            )
            
            # Create strata for train+val subset
            strata_train_val = strata.loc[idx_train_val]
            
            # Second split: train vs val
            idx_train, idx_val = train_test_split(
                idx_train_val,
                test_size=self.val_size / (1 - self.test_size),
                stratify=strata_train_val,
                random_state=self.random_state
            )
            
            splits = (df.loc[idx_train], df.loc[idx_val], df.loc[idx_test])
            
            # Optionally validate splits without 'is_prospect'
            # self._validate_splits(splits)  # Remove or update this method
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            raise


    def save_splits_with_scouting_metadata(self, splits: Tuple[pd.DataFrame, ...], output_dir: Path) -> None:
        """Save splits with comprehensive scouting metadata"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            train, val, test = splits
            
            # Convert numpy types to native Python types for YAML serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return int(obj) if isinstance(obj, np.integer) else float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(x) for x in obj]
                return obj
            
            # Create detailed metadata with converted types
            metadata = {
                'split_config': {
                    'test_size': float(self.test_size),
                    'val_size': float(self.val_size),
                    'random_state': int(self.random_state)
                    
                },
                'split_stats': convert_numpy_types({
                    'train': self._get_split_stats(train),
                    'validation': self._get_split_stats(val),
                    'test': self._get_split_stats(test)
                }),
                'scouting_params': self.SCOUTING_PARAMS,
                'features': self.feature_columns,
                'split_date': pd.Timestamp.now().isoformat()
            }
            
            # Save metadata as YAML for better readability
            with open(output_dir/'scouting_split_metadata.yaml', 'w') as f:
                yaml.dump(metadata, f, sort_keys=False, default_flow_style=False)
            
            # Save splits
            train.to_csv(output_dir/'train.csv', index=False)
            val.to_csv(output_dir/'validation.csv', index=False)
            test.to_csv(output_dir/'test.csv', index=False)
            
            self.logger.info(f"Saved scouting splits to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save scouting splits: {e}")
            raise

    def _get_split_stats(self, df: pd.DataFrame) -> Dict:
        """Generate scouting-specific statistics for a split"""
        stats = {
            'total_samples': len(df),
            'position_distribution': df['player_positions'].value_counts(normalize=True).head(10).to_dict(),
            'potential_stats': {
                'mean': float(df['potential'].mean()),
                'std': float(df['potential'].std())
            }
        }
        
        # Add scouting parameter stats safely (only numeric columns)
        for param_group, params in self.SCOUTING_PARAMS.items():
            if isinstance(params, dict):
                for pos, pos_params in params.items():
                    stats[f'{param_group}_{pos}'] = {
                        p: float(df[p].mean()) for p in pos_params
                        if p in df.columns and pd.api.types.is_numeric_dtype(df[p])
                    }
            else:
                stats[param_group] = {
                    p: float(df[p].mean()) for p in params
                    if p in df.columns and pd.api.types.is_numeric_dtype(df[p])
                }
        
        return stats
