import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def cross_validation(model, X, y):
    """
    Perform cross-validation.
    
    Parameters:
    - model: The machine learning model to evaluate.
    - X: Feature matrix.
    - y: Target variable.
    """
    # 配置重复交叉验证
    cv_strategy = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    # 定义存储指标的列表
    pearsons_r = []
    maes = []
    rmses = []
    medaes = []
    r2s = []

    # 手动实现重复交叉验证以计算多种指标
    for train_idx, test_idx in cv_strategy.split(X, y):
        # 划分训练集和测试集
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 模型训练
        model.fit(X_train, y_train)
        
        # 模型预测
        y_pred = model.predict(X_test)
        
        # 计算 Pearson’s R
        r, _ = pearsonr(y_test, y_pred)
        pearsons_r.append(r)
        
        # 计算 Mean Absolute Error (m.a.e.)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)
        
        # 计算 Root Mean Square Error (r.m.s.e.)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)

        medae = median_absolute_error(y_test, y_pred)
        medaes.append(medae)

        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)

    # 将结果保存到 DataFrame 中
    metrics_df = pd.DataFrame({
        'Pearson_R': pearsons_r,
        'MAE': maes,
        'RMSE': rmses,
        'MEDAE': medaes,
        'R2': r2s
    })

    # 打印结果
    print("\n--- 交叉验证性能报告 ---")
    print(f"平均 Pearson’s R: {np.mean(pearsons_r):.4f} ± {np.std(pearsons_r):.4f}")
    print(f"平均 MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"平均 RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"平均 MEDAE: {np.mean(medaes):.4f} ± {np.std(medaes):.4f}")
    print(f"平均 R2: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    # 可视化结果
    metrics_melted = pd.melt(metrics_df, var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Metric', y='Score', data=metrics_melted)
    plt.title('Distribution of Cross-Validation Scores', fontsize=16)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return metrics_df