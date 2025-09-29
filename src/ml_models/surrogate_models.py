"""
Machine Learning Surrogate Models for Real-Time Option Pricing.

This module implements ML models that learn from PDE solver outputs to provide
ultra-fast option pricing for real-time risk management.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import time
from typing import Dict, Tuple, List


class OptionPricingSurrogate:
    """Base class for option pricing surrogate models."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize surrogate model.

        Parameters:
        -----------
        model_type : str
            Type of model: 'random_forest', 'gradient_boosting', or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_time = None
        self.metrics = {}

    def build_model(self, **kwargs):
        """Build the underlying ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                tree_method='hist',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        target_col: str = 'price',
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict:
        """
        Train the surrogate model.

        Parameters:
        -----------
        X : pd.DataFrame
            Features (market parameters)
        y : pd.DataFrame
            Targets (price and Greeks)
        target_col : str
            Column to predict ('price', 'delta', 'gamma', 'theta')
        test_size : float
            Fraction of data for testing
        scale_features : bool
            Whether to scale features

        Returns:
        --------
        metrics : dict
            Training and testing metrics
        """
        print(f"\nTraining {self.model_type} for {target_col}...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[target_col], test_size=test_size, random_state=42
        )

        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Train
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        self.training_time = time.time() - start_time

        print(f"Training completed in {self.training_time:.2f} seconds")

        # Predict
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        self.metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'training_time': self.training_time
        }

        # Print metrics
        print("\nTraining Metrics:")
        print(f"  Train RMSE: {self.metrics['train_rmse']:.6f}")
        print(f"  Test RMSE:  {self.metrics['test_rmse']:.6f}")
        print(f"  Train MAE:  {self.metrics['train_mae']:.6f}")
        print(f"  Test MAE:   {self.metrics['test_mae']:.6f}")
        print(f"  Train R²:   {self.metrics['train_r2']:.6f}")
        print(f"  Test R²:    {self.metrics['test_r2']:.6f}")

        return self.metrics

    def predict(
        self,
        X: pd.DataFrame,
        scale_features: bool = True
    ) -> np.ndarray:
        """
        Predict option prices/Greeks.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        scale_features : bool
            Whether to scale features

        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        return self.model.predict(X_scaled)

    def predict_with_timing(
        self,
        X: pd.DataFrame,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Predict with timing measurement.

        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        inference_time : float
            Time taken for prediction (seconds)
        """
        start_time = time.time()
        predictions = self.predict(X, scale_features)
        inference_time = time.time() - start_time

        return predictions, inference_time

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            return importance_df
        else:
            return None

    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.metrics = model_data.get('metrics', {})
        print(f"Model loaded from: {filepath}")


class MultiOutputSurrogate:
    """Surrogate model that predicts price and all Greeks simultaneously."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize multi-output surrogate.

        Parameters:
        -----------
        model_type : str
            Type of base model
        """
        self.model_type = model_type
        self.models = {
            'price': OptionPricingSurrogate(model_type),
            'delta': OptionPricingSurrogate(model_type),
            'gamma': OptionPricingSurrogate(model_type),
            'theta': OptionPricingSurrogate(model_type)
        }

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train models for all targets.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.DataFrame
            Targets (must have columns: price, delta, gamma, theta)
        test_size : float
            Test set size

        Returns:
        --------
        all_metrics : dict
            Metrics for each target
        """
        all_metrics = {}

        for target in ['price', 'delta', 'gamma', 'theta']:
            print(f"\n{'='*60}")
            print(f"Training model for: {target}")
            print('='*60)

            metrics = self.models[target].train(X, y, target, test_size)
            all_metrics[target] = metrics

        return all_metrics

    def predict_all(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Predict all outputs at once.

        Parameters:
        -----------
        X : pd.DataFrame
            Features

        Returns:
        --------
        predictions : dict
            Dictionary with predictions for each target
        """
        predictions = {}

        for target in ['price', 'delta', 'gamma', 'theta']:
            predictions[target] = self.models[target].predict(X)

        return predictions

    def save_all(self, directory: str):
        """Save all models."""
        import os
        os.makedirs(directory, exist_ok=True)

        for target, model in self.models.items():
            filepath = os.path.join(directory, f'{target}_model.joblib')
            model.save_model(filepath)

    def load_all(self, directory: str):
        """Load all models."""
        import os

        for target in ['price', 'delta', 'gamma', 'theta']:
            filepath = os.path.join(directory, f'{target}_model.joblib')
            if os.path.exists(filepath):
                self.models[target].load_model(filepath)