from pathlib import Path
import json, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import build_dataset, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PROCESSED
ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'; RESULTS=ROOT/'outputs/results'; FIGURES=ROOT/'outputs/figures'

def prep():
    df=build_dataset()
    split=int(len(df)*0.8)
    train=df.iloc[:split]; test=df.iloc[split:]
    Xtr,ytr=train[FEATURES],train.stockout_within_7_days
    Xte,yte=test[FEATURES],test.stockout_within_7_days
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',numeric,NUMERIC_FEATURES),('cat',categorical,CATEGORICAL_FEATURES)])
    return df,Xtr,ytr,Xte,yte,pre

def main():
    df,Xtr,ytr,Xte,yte,pre=prep()
    models={
      'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
      'KNN': KNeighborsClassifier(n_neighbors=7,weights='distance'),
      'Naive Bayes': GaussianNB(),
      # LinearSVC is substantially faster than SVC(probability=True) on this
      # one-hot encoded dataset; calibration supplies the required probabilities.
      'SVM': CalibratedClassifierCV(LinearSVC(class_weight='balanced',random_state=42,max_iter=3000),cv=3),
      'Decision Tree': DecisionTreeClassifier(max_depth=8,class_weight='balanced',random_state=42),
      'Random Forest': RandomForestClassifier(n_estimators=120,max_depth=14,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=42)
    }
    rows=[]; fitted={}
    for name,est in models.items():
        pipe=Pipeline([('preprocessor',pre),('model',est)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
        row={'Model':name,'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0),'ROC-AUC':roc_auc_score(yte,prob),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
        rows.append(row); fitted[name]=pipe
        print(name,row)
    results=pd.DataFrame(rows).sort_values(['F1','Recall','ROC-AUC'],ascending=False)
    RESULTS.mkdir(parents=True,exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS/'model_comparison.csv',index=False)
    best_name=results.iloc[0]['Model']; joblib.dump(fitted[best_name],MODEL_DIR/'best_model.joblib')
    with open(MODEL_DIR/'model_metadata.json','w') as f: json.dump({'best_model':best_name,'features':FEATURES,'target':'stockout_within_7_days','risk_thresholds':{'medium':0.35,'high':0.65}},f,indent=2)
    # Visual outputs from actual results.
    import matplotlib.pyplot as plt, seaborn as sns
    sns.set_theme(style='whitegrid'); FIGURES.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(10,5)); results.set_index('Model')[['Accuracy','Precision','Recall','F1','ROC-AUC']].plot(kind='bar',ax=plt.gca()); plt.ylim(0,1); plt.title('Actual holdout model comparison'); plt.tight_layout(); plt.savefig(FIGURES/'model_comparison.png',dpi=150); plt.close()
    for col,title,file in [('current_stock','Current stock distribution','stock_distribution.png'),('sales_velocity','Historical sales velocity','sales_velocity.png'),('demand_variability','Demand variability','demand_variability.png')]:
        plt.figure(figsize=(8,4)); sns.histplot(df[col],bins=30,kde=True); plt.title(title); plt.tight_layout(); plt.savefig(FIGURES/file,dpi=150); plt.close()
    plt.figure(figsize=(7,4)); sns.countplot(data=df,x='promotion_status',hue='stockout_within_7_days'); plt.title('Promotion status vs derived 7-day stockout'); plt.tight_layout(); plt.savefig(FIGURES/'promotion_vs_stockout.png',dpi=150); plt.close()
    plt.figure(figsize=(7,4)); sns.boxplot(data=df,x='stockout_within_7_days',y='current_stock'); plt.title('Current stock vs derived 7-day stockout'); plt.tight_layout(); plt.savefig(FIGURES/'stock_vs_stockout.png',dpi=150); plt.close()
    plt.figure(figsize=(5,4)); sns.countplot(data=df,x='stockout_within_7_days'); plt.title('Derived target distribution'); plt.tight_layout(); plt.savefig(FIGURES/'target_distribution.png',dpi=150); plt.close()
    print(f'Best model: {best_name}')
if __name__=='__main__': main()
