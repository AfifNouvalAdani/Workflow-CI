import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Jangan set tracking URI dan experiment di sini
# MLflow Project akan mengelolanya

def load_data():
    train_df = pd.read_csv("weather_train_processed.csv")
    test_df = pd.read_csv("weather_test_processed.csv")
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    return X_train, X_test, y_train, y_test

def train_basic_model():
    print("="*60)
    print("CI/CD MODEL TRAINING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # PENTING: Jangan gunakan mlflow.start_run() 
    # karena MLflow Project sudah mengelola run
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics secara manual (opsional, autolog sudah menangani ini)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nTraining completed successfully!")
    
    return model

if __name__ == "__main__":
    print("Starting CI/CD model training...")
    model = train_basic_model()
    print("CI/CD pipeline completed!")