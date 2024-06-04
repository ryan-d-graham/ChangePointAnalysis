from sklearn.linear_model import LassoCV

# Use LassoCV to find the optimal alpha through cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5, max_iter=10000)
lasso_cv.fit(V_transformed, np.zeros(V_transformed.shape[0]))

# Extract the best alpha
best_alpha = lasso_cv.alpha_
print("Best alpha found by LassoCV:", best_alpha)

# Fit Lasso with the best alpha
lasso = Lasso(alpha=best_alpha, max_iter=10000, tol=1e-4)
lasso.fit(V_transformed, np.zeros(V_transformed.shape[0]))
V_sparse = lasso.coef_.reshape(1, -1) * V_transformed

# Visualize the results
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Heatmap of the sparse kernel PCA components
plt.figure(figsize=(10, 8))
sns.heatmap(V_sparse, cmap='viridis', annot=True, linewidths=.5)
plt.title('Heatmap of Sparse Kernel-PCA Transformed Data')
plt.xlabel('Components')
plt.ylabel('Samples')
plt.show()

# Pairplot of the sparse kernel PCA components
sparse_df = pd.DataFrame(V_sparse, columns=[f'Component {i+1}' for i in range(V_sparse.shape[1])])
sns.pairplot(sparse_df)
plt.suptitle('Pairplot of Sparse Kernel-PCA Transformed Data', y=1.02)
plt.show()