"""
Machine Learning Model untuk Klasifikasi Tingkat Stres Karyawan Remote
dengan Burnout Score sebagai Proxy Maslach Burnout Inventory (MBI)

Author: Research
Date: March 2026
Description: Comprehensive ML pipeline untuk stress level classification
dengan feature engineering berdasarkan MBI concept
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)

warnings.filterwarnings('ignore')
np.random.seed(42)

print("\n" + "="*90)
print("MACHINE LEARNING CLASSIFICATION: STRESS LEVEL PREDICTION WITH BURNOUT SCORE")
print("="*90)

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
TRAIN_SIZE = 0.80
CV_FOLDS = 5

# =============================================================================
# CLASS: DATA LOADING & EXPLORATION
# =============================================================================
class DataLoader:
    """Load and explore dataset"""
    
    @staticmethod
    def load_data(filepath):
        """Load dataset from CSV"""
        df = pd.read_csv(filepath)
        return df
    
    @staticmethod
    def explore_data(df):
        """Explore dataset structure and content"""
        print("\n" + "="*90)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("="*90)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        print(f"\nData Types:\n{df.dtypes}")
        
        print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        print(f"\nTarget Variable Distribution (Stress_Level):")
        print(df['Stress_Level'].value_counts())
        print(df['Stress_Level'].value_counts(normalize=True))
        
        print(f"\nBasic Statistics:\n{df.describe()}")


# =============================================================================
# CLASS: BURNOUT SCORE CALCULATION
# =============================================================================
class BurnoutScoreEngine:
    """Calculate Burnout Score based on Maslach Burnout Inventory"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def encode_categorical_variables(self):
        """Encode categorical variables to numeric"""
        print("\n" + "="*90)
        print("STEP 2: CATEGORICAL VARIABLE ENCODING")
        print("="*90)
        
        # Sleep Quality: Poor(1) -> Average(2) -> Good(3)
        sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
        self.df['Sleep_Quality_Encoded'] = self.df['Sleep_Quality'].map(sleep_quality_map)
        
        # Productivity Change: Decrease(1) -> No Change(2) -> Increase(3)
        productivity_map = {'Decrease': 1, 'No Change': 2, 'Increase': 3}
        self.df['Productivity_Change_Encoded'] = self.df['Productivity_Change'].map(productivity_map)
        
        print(f"\n[Sleep Quality Encoding]")
        print(f"  Poor -> 1, Average -> 2, Good -> 3")
        print(f"  Encoded values: {self.df['Sleep_Quality_Encoded'].value_counts().sort_index().to_dict()}")
        
        print(f"\n[Productivity Change Encoding]")
        print(f"  Decrease -> 1, No Change -> 2, Increase -> 3")
        print(f"  Encoded values: {self.df['Productivity_Change_Encoded'].value_counts().sort_index().to_dict()}")
        
        return self.df
    
    def normalize_variables(self):
        """Normalize variables using MinMaxScaler"""
        print("\n" + "="*90)
        print("STEP 3: VARIABLE NORMALIZATION (MinMaxScaler 0-1)")
        print("="*90)
        
        variables_to_normalize = [
            'Hours_Worked_Per_Week',
            'Sleep_Quality_Encoded',
            'Social_Isolation_Rating',
            'Work_Life_Balance_Rating',
            'Productivity_Change_Encoded'
        ]
        
        normalized_values = self.scaler.fit_transform(self.df[variables_to_normalize])
        
        for i, var in enumerate(variables_to_normalize):
            self.df[f'{var}_norm'] = normalized_values[:, i]
        
        print(f"\nNormalized Variables Statistics:")
        normalized_cols = [f'{var}_norm' for var in variables_to_normalize]
        print(self.df[normalized_cols].describe().round(4))
        
        return self.df
    
    def calculate_mbi_dimensions(self):
        """Calculate 3 dimensions of Maslach Burnout Inventory"""
        print("\n" + "="*90)
        print("STEP 4: MASLACH BURNOUT INVENTORY (MBI) DIMENSIONS CALCULATION")
        print("="*90)
        
        # Emotional Exhaustion (EE)
        # Proksikasi: High work hours + Low sleep quality
        self.df['Emotional_Exhaustion'] = (
            self.df['Hours_Worked_Per_Week_norm'] + 
            (1 - self.df['Sleep_Quality_Encoded_norm'])
        ) / 2
        
        print(f"\n[1] EMOTIONAL EXHAUSTION (EE)")
        print(f"    Formula: (Hours_Worked_normalized + (1 - Sleep_Quality_normalized)) / 2")
        print(f"    Mean: {self.df['Emotional_Exhaustion'].mean():.4f}")
        print(f"    Std: {self.df['Emotional_Exhaustion'].std():.4f}")
        print(f"    Range: {self.df['Emotional_Exhaustion'].min():.4f} - {self.df['Emotional_Exhaustion'].max():.4f}")
        
        # Depersonalization (DP)
        # Proksikasi: High social isolation + Low work-life balance
        self.df['Depersonalization'] = (
            self.df['Social_Isolation_Rating_norm'] + 
            (1 - self.df['Work_Life_Balance_Rating_norm'])
        ) / 2
        
        print(f"\n[2] DEPERSONALIZATION (DP)")
        print(f"    Formula: (Social_Isolation_normalized + (1 - Work_Life_Balance_normalized)) / 2")
        print(f"    Mean: {self.df['Depersonalization'].mean():.4f}")
        print(f"    Std: {self.df['Depersonalization'].std():.4f}")
        print(f"    Range: {self.df['Depersonalization'].min():.4f} - {self.df['Depersonalization'].max():.4f}")
        
        # Reduced Personal Accomplishment (RPA)
        # Proksikasi: Decreased productivity
        self.df['Reduced_Personal_Accomplishment'] = 1 - self.df['Productivity_Change_Encoded_norm']
        
        print(f"\n[3] REDUCED PERSONAL ACCOMPLISHMENT (RPA)")
        print(f"    Formula: 1 - Productivity_Change_normalized")
        print(f"    Mean: {self.df['Reduced_Personal_Accomplishment'].mean():.4f}")
        print(f"    Std: {self.df['Reduced_Personal_Accomplishment'].std():.4f}")
        print(f"    Range: {self.df['Reduced_Personal_Accomplishment'].min():.4f} - {self.df['Reduced_Personal_Accomplishment'].max():.4f}")
        
        return self.df
    
    def calculate_burnout_score(self):
        """Calculate composite Burnout Score"""
        print("\n" + "="*90)
        print("STEP 5: COMPOSITE BURNOUT SCORE CALCULATION")
        print("="*90)
        
        self.df['Burnout_Score'] = (
            self.df['Emotional_Exhaustion'] + 
            self.df['Depersonalization'] + 
            self.df['Reduced_Personal_Accomplishment']
        ) / 3
        
        print(f"\nFormula: (EE + DP + RPA) / 3")
        print(f"Range: 0.0 - 1.0")
        print(f"\nBurnout Score Statistics:")
        print(f"  Mean: {self.df['Burnout_Score'].mean():.4f}")
        print(f"  Median: {self.df['Burnout_Score'].median():.4f}")
        print(f"  Std Dev: {self.df['Burnout_Score'].std():.4f}")
        print(f"  Min: {self.df['Burnout_Score'].min():.4f}")
        print(f"  Max: {self.df['Burnout_Score'].max():.4f}")
        print(f"  Q1: {self.df['Burnout_Score'].quantile(0.25):.4f}")
        print(f"  Q3: {self.df['Burnout_Score'].quantile(0.75):.4f}")
        
        # Categorize burnout levels
        self.df['Burnout_Category'] = pd.cut(
            self.df['Burnout_Score'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['Low', 'Moderate', 'High'],
            include_lowest=True
        )
        
        print(f"\nBurnout Category Distribution:")
        print(self.df['Burnout_Category'].value_counts().sort_index())
        print(self.df['Burnout_Category'].value_counts(normalize=True).sort_index())
        
        return self.df
    
    def get_data(self):
        """Get processed dataframe"""
        return self.df


# =============================================================================
# CLASS: FEATURE PREPARATION
# =============================================================================
class FeaturePreparation:
    """Prepare features for modeling"""
    
    @staticmethod
    def prepare_features(df):
        """Select and prepare features for classification"""
        print("\n" + "="*90)
        print("STEP 6: FEATURE ENGINEERING AND PREPARATION")
        print("="*90)
        
        # Feature selection
        features = [
            'Burnout_Score',                      # Main feature engineering
            'Emotional_Exhaustion',               # MBI dimension 1
            'Depersonalization',                  # MBI dimension 2
            'Reduced_Personal_Accomplishment',    # MBI dimension 3
            'Hours_Worked_Per_Week',
            'Social_Isolation_Rating',
            'Work_Life_Balance_Rating',
            'Number_of_Virtual_Meetings',
            'Years_of_Experience',
            'Age'
        ]
        
        print(f"\nFeatures Selected ({len(features)} features):")
        for i, feat in enumerate(features, 1):
            print(f"  {i:2d}. {feat}")
        
        X = df[features].copy()
        y = df['Stress_Level'].copy()
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        print(f"Target Vector Shape: {y.shape}")
        
        print(f"\nFeature Statistics:")
        print(X.describe().round(4))
        
        print(f"\nTarget Distribution:")
        print(y.value_counts())
        
        return X, y, features


# =============================================================================
# CLASS: MODEL TRAINING & EVALUATION
# =============================================================================
class ModelEvaluator:
    """Train and evaluate multiple classification models"""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        # Encode target
        self.label_encoder = LabelEncoder()
        self.y_train_enc = self.label_encoder.fit_transform(y_train)
        self.y_test_enc = self.label_encoder.transform(y_test)
        
        # Standardize features for SVM
        self.scaler_features = StandardScaler()
        self.X_train_scaled = self.scaler_features.fit_transform(X_train)
        self.X_test_scaled = self.scaler_features.transform(X_test)
        
        self.models = {}
        self.results = {}
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*90)
        print("STEP 7A: TRAINING LOGISTIC REGRESSION")
        print("="*90)
        
        print(f"\nParameters:")
        print(f"  max_iter: 1000")
        print(f"  solver: auto")
        print(f"  random_state: {RANDOM_STATE}")
        
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
        lr.fit(self.X_train, self.y_train_enc)
        
        self.models['Logistic Regression'] = lr
        
        print(f"Training completed!")
        print(f"Model coefficients shape: {lr.coef_.shape}")
        
        return lr
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*90)
        print("STEP 7B: TRAINING RANDOM FOREST")
        print("="*90)
        
        print(f"\nParameters:")
        print(f"  n_estimators: 100")
        print(f"  max_depth: 15")
        print(f"  random_state: {RANDOM_STATE}")
        print(f"  n_jobs: -1 (parallel)")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train_enc)
        
        self.models['Random Forest'] = rf
        
        print(f"Training completed!")
        
        return rf
    
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("\n" + "="*90)
        print("STEP 7C: TRAINING SUPPORT VECTOR MACHINE (SVM)")
        print("="*90)
        
        print(f"\nParameters:")
        print(f"  kernel: rbf")
        print(f"  C: 1.0")
        print(f"  random_state: {RANDOM_STATE}")
        print(f"  Features: StandardScaled")
        
        svm = SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True)
        svm.fit(self.X_train_scaled, self.y_train_enc)
        
        self.models['SVM'] = svm
        
        print(f"Training completed!")
        print(f"Number of support vectors: {len(svm.support_vectors_)}")
        
        return svm
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*90)
        print("STEP 8: MODEL EVALUATION")
        print("="*90)
        
        for model_name in ['Logistic Regression', 'Random Forest', 'SVM']:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            
            # Use scaled features for SVM, regular features for others
            if model_name == 'SVM':
                X_test_use = self.X_test_scaled
            else:
                X_test_use = self.X_test
            
            # Predictions
            y_pred_enc = model.predict(X_test_use)
            y_pred = self.label_encoder.inverse_transform(y_pred_enc)
            
            # Metrics
            accuracy = accuracy_score(self.y_test_enc, y_pred_enc)
            precision = precision_score(self.y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            recall = recall_score(self.y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            
            cm = confusion_matrix(self.y_test_enc, y_pred_enc)
            
            self.results[model_name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_enc': y_pred_enc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
            print(f"\n{'='*90}")
            print(f"Model: {model_name}")
            print(f"{'='*90}")
            print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            print(f"Confusion Matrix:")
            print(cm)
            
            # Cross-validation
            if model_name != 'SVM':
                cv_scores = cross_val_score(model, self.X_train, self.y_train_enc, cv=CV_FOLDS)
                print(f"\nCross-Validation Scores ({CV_FOLDS}-fold): {cv_scores}")
                print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if 'Random Forest' in self.models:
            rf = self.models['Random Forest']
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance_df
        return None
    
    def get_results_dataframe(self):
        """Get results as dataframe"""
        results_list = []
        for model_name, result in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1']
            })
        return pd.DataFrame(results_list)


# =============================================================================
# CLASS: VISUALIZATION
# =============================================================================
class Visualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, df, y_test, evaluator):
        self.df = df
        self.y_test = y_test
        self.evaluator = evaluator
    
    def plot_stress_distribution(self):
        """Plot Stress Level distribution"""
        print("\n[VIZ-1] Creating Stress Level Distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        stress_counts = self.df['Stress_Level'].value_counts()
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        axes[0].bar(stress_counts.index, stress_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Stress Level', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        for i, v in enumerate(stress_counts.values):
            axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # Pie plot
        axes[1].pie(stress_counts.values, labels=stress_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Stress Level Composition', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('01_Stress_Level_Distribution.png', dpi=150, bbox_inches='tight', format='png')
        plt.close()
        print("      [OK] Saved: 01_Stress_Level_Distribution.png")
    
    def plot_burnout_distribution(self):
        """Plot Burnout Score distribution"""
        print("\n[VIZ-2] Creating Burnout Score Distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(self.df['Burnout_Score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.df['Burnout_Score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].axvline(self.df['Burnout_Score'].median(), color='green', linestyle='--', linewidth=2, label='Median')
        axes[0, 0].set_xlabel('Burnout Score', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Histogram of Burnout Score', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # KDE
        self.df['Burnout_Score'].plot(kind='kde', ax=axes[0, 1], color='darkblue', linewidth=2)
        axes[0, 1].set_xlabel('Burnout Score', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('KDE Plot of Burnout Score', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Box plot by stress level
        stress_order = ['Low', 'Medium', 'High']
        self.df.boxplot(column='Burnout_Score', by='Stress_Level', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Stress Level', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Burnout Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Burnout Score by Stress Level', fontsize=12, fontweight='bold')
        axes[1, 0].get_figure().suptitle('')
        
        # Violin plot
        sns.violinplot(data=self.df, x='Stress_Level', y='Burnout_Score', order=stress_order, ax=axes[1, 1])
        axes[1, 1].set_xlabel('Stress Level', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Burnout Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Burnout Score Distribution (Violin)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('02_Burnout_Score_Distribution.png', dpi=150, bbox_inches='tight', format='png')
        plt.close()
        print("      [OK] Saved: 02_Burnout_Score_Distribution.png")
    
    def plot_burnout_by_job_role(self):
        """Plot Burnout Score by Job Role"""
        print("\n[VIZ-3] Creating Burnout Score by Job Role...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        burnout_by_role = self.df.groupby('Job_Role')['Burnout_Score'].agg(['mean', 'std']).reset_index()
        burnout_by_role = burnout_by_role.sort_values('mean', ascending=False)
        
        bars = ax.bar(burnout_by_role['Job_Role'], burnout_by_role['mean'], 
                      yerr=burnout_by_role['std'], capsize=5, alpha=0.7, 
                      color='steelblue', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Job Role', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Burnout Score', fontsize=11, fontweight='bold')
        ax.set_title('Average Burnout Score by Job Role', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('03_Burnout_by_Job_Role.png', dpi=150, bbox_inches='tight', format='png')
        plt.close()
        print("      [OK] Saved: 03_Burnout_by_Job_Role.png")
    
    def plot_confusion_matrices(self):
        """Plot Confusion Matrices for all models"""
        print("\n[VIZ-4] Creating Confusion Matrices...")
        
        n_models = len(self.evaluator.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.evaluator.results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                       cbar=True, cbar_kws={'label': 'Count'})
            axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'Confusion Matrix - {model_name}\n(Accuracy: {result["accuracy"]:.1%})', 
                              fontsize=12, fontweight='bold')
            
            # Set tick labels
            stress_labels = self.evaluator.label_encoder.classes_
            axes[idx].set_xticklabels(stress_labels, rotation=45)
            axes[idx].set_yticklabels(stress_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig('04_Confusion_Matrices.png', dpi=150, bbox_inches='tight', format='png')
        plt.close()
        print("      [OK] Saved: 04_Confusion_Matrices.png")
    
    def plot_feature_importance(self):
        """Plot Feature Importance"""
        print("\n[VIZ-5] Creating Feature Importance...")
        
        importance_df = self.evaluator.get_feature_importance()
        
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            top_features = importance_df.head(15)
            
            bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
                          color='forestgreen', edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values)
            ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Features', fontsize=11, fontweight='bold')
            ax.set_title('Top 15 Most Important Features (Random Forest)', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('05_Feature_Importance.png', dpi=150, bbox_inches='tight', format='png')
            plt.close()
            print("      [OK] Saved: 05_Feature_Importance.png")
    
    def plot_model_comparison(self):
        """Plot Model Performance Comparison"""
        print("\n[VIZ-6] Creating Model Comparison...")
        
        results_df = self.evaluator.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            bars = ax.bar(results_df['Model'], results_df[metric], 
                         alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                         edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('06_Model_Comparison.png', dpi=150, bbox_inches='tight', format='png')
        plt.close()
        print("      [OK] Saved: 06_Model_Comparison.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution pipeline"""
    
    # 1. Load Data
    loader = DataLoader()
    df = loader.load_data('Impact_of_Remote_Work_on_Mental_Health.csv')
    loader.explore_data(df)
    
    # 2. Feature Engineering: Burnout Score
    burnout_engine = BurnoutScoreEngine(df)
    df = burnout_engine.encode_categorical_variables()
    df = burnout_engine.normalize_variables()
    df = burnout_engine.calculate_mbi_dimensions()
    df = burnout_engine.calculate_burnout_score()
    
    # 3. Prepare Features
    X, y, feature_names = FeaturePreparation.prepare_features(df)
    
    # 4. Train-Test Split
    print("\n" + "="*90)
    print("STEP 7: TRAIN-TEST SPLIT (80:20)")
    print("="*90)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain Set Size: {len(X_train)} ({TRAIN_SIZE*100:.0f}%)")
    print(f"Test Set Size: {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    print(f"\nTrain Set Stress_Level Distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTest Set Stress_Level Distribution:")
    print(y_test.value_counts().sort_index())
    
    # 5. Model Training & Evaluation
    evaluator = ModelEvaluator(X_train, X_test, y_train, y_test, feature_names)
    evaluator.train_logistic_regression()
    evaluator.train_random_forest()
    evaluator.train_svm()
    evaluator.evaluate_all_models()
    
    # 6. Visualizations
    print("\n" + "="*90)
    print("STEP 9: VISUALIZATIONS")
    print("="*90)
    
    visualizer = Visualizer(df, y_test, evaluator)
    visualizer.plot_stress_distribution()
    visualizer.plot_burnout_distribution()
    visualizer.plot_burnout_by_job_role()
    visualizer.plot_confusion_matrices()
    visualizer.plot_feature_importance()
    visualizer.plot_model_comparison()
    
    # 7. Save Results
    print("\n" + "="*90)
    print("STEP 10: SAVING RESULTS")
    print("="*90)
    
    # Save enhanced dataset
    output_columns = [
        'Employee_ID', 'Stress_Level', 'Job_Role', 'Industry',
        'Hours_Worked_Per_Week', 'Sleep_Quality', 
        'Social_Isolation_Rating', 'Work_Life_Balance_Rating', 'Productivity_Change',
        'Emotional_Exhaustion', 'Depersonalization', 'Reduced_Personal_Accomplishment',
        'Burnout_Score', 'Burnout_Category'
    ]
    
    df_output = df[output_columns].copy()
    df_output.to_csv('Dataset_Enhanced_with_Burnout_Score.csv', index=False)
    print(f"\n[OK] Saved: Dataset_Enhanced_with_Burnout_Score.csv")
    
    # Save model results
    results_df = evaluator.get_results_dataframe()
    results_df.to_csv('Model_Performance_Results.csv', index=False)
    print(f"[OK] Saved: Model_Performance_Results.csv")
    
    # Save feature importance
    importance_df = evaluator.get_feature_importance()
    if importance_df is not None:
        importance_df.to_csv('Feature_Importance_Ranking.csv', index=False)
        print(f"[OK] Saved: Feature_Importance_Ranking.csv")
    
    # 8. Final Summary
    print("\n" + "="*90)
    print("FINAL SUMMARY AND CONCLUSIONS")
    print("="*90)
    
    print(f"\n[DATASET ANALYSIS]")
    print(f"  Total Employees: {len(df)}")
    print(f"  Burnout Score Mean: {df['Burnout_Score'].mean():.4f}")
    print(f"  Burnout Category:")
    print(df['Burnout_Category'].value_counts().to_string())
    
    print(f"\n[MODEL PERFORMANCE]")
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"  Best Model: {best_model['Model']}")
    print(f"  Best Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
    print(f"  Best F1-Score: {best_model['F1-Score']:.4f}")
    
    print(f"\n[TOP FEATURES]")
    importance_df = evaluator.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(5).to_string(index=False))
    
    print(f"\n[DELIVERABLES]")
    print(f"  CSV Files: 3")
    print(f"    - Dataset_Enhanced_with_Burnout_Score.csv")
    print(f"    - Model_Performance_Results.csv")
    print(f"    - Feature_Importance_Ranking.csv")
    print(f"  PNG Files: 6")
    print(f"    - 01_Stress_Level_Distribution.png")
    print(f"    - 02_Burnout_Score_Distribution.png")
    print(f"    - 03_Burnout_by_Job_Role.png")
    print(f"    - 04_Confusion_Matrices.png")
    print(f"    - 05_Feature_Importance.png")
    print(f"    - 06_Model_Comparison.png")
    
    print("\n" + "="*90)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
