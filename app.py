import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Create some datasets
datasets = {
	'Circles': make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42),
	'Moons': make_moons(noise=0, random_state=42),
	'Linearly Separable': make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
}

# Custom basis functions
def polyTransform(X, degree):
	return np.power(X, degree)

def sin_basis(X):
	return np.sin(X)

def gaussian_basis(X, N, width_factor=2.0):
	centers = np.linspace(X.min(), X.max(), N)
	width = width_factor * (centers[1] - centers[0])
	return np.exp(-0.5 * np.sum(((X[:, :, np.newaxis] - centers) / width) ** 2, axis=1) )

# Streamlit app
def main():
	st.title("Neural Network Visualization")
	st.sidebar.header("Settings")
	
	# Select dataset
	dataset_name = st.sidebar.selectbox("Choose a dataset", list(datasets.keys()))
	X_init, y = datasets[dataset_name]
	# print(X.shape, y.shape)

	# Basis functions
	basis_functions = {
		'X1': lambda x: polyTransform(x[:, 0], 1),
		'X2': lambda x: polyTransform(x[:, 1], 1),
		'X1^2': lambda x: polyTransform(x[:, 0], 2),
		'X2^2': lambda x: polyTransform(x[:, 1], 2),
		'Sin(X1)': lambda x: sin_basis(x[:, 0]),
		'Sin(X2)': lambda x: sin_basis(x[:, 1]),
		'Gaussian': lambda x: gaussian_basis(x, 5)
	}
	selected_basis = st.sidebar.multiselect("Select Basis Functions", basis_functions.keys(), default=['X1', 'X2'])
	
	# Hyperparameter controls
	layer_sizes = st.sidebar.text_input("Layer Sizes (comma-separated)", "3,3,3")
	layer_sizes = [int(neurons) for neurons in layer_sizes.split(",")]
	
	learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
	epochs = st.sidebar.slider("Epochs", 10, 500, 100, 10)
	batch_size = st.sidebar.slider("Batch Size", 1, 30, 10, 1)
	test_size = st.sidebar.slider("Test Size", 0.0, 1.0, 0.2, 0.05)


	activation = st.sidebar.selectbox("Activation Function", ["tanh", "relu", "logistic"])
	
	# Button to start training
	if st.sidebar.button("Train Model"):
		basis = [basis_functions[basis] for basis in selected_basis  if basis != "Gaussian"]
		X = np.array([basis_function(X_init) for basis_function in basis]).T
		if "Gaussian" in selected_basis:
			print("Gaussian")
			temp = gaussian_basis(X_init, 5)
			X = np.concatenate([X, temp], axis=1)
		stdScale = StandardScaler().fit(X)
		X = stdScale.transform(X)

		# X = np.concatenate([basis_function(X_init) for basis_function in basis], axis=0)
		print(X.shape)
		# Train the neural network
		model = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=epochs, learning_rate_init=learning_rate, activation=activation, solver="adam", batch_size=batch_size, shuffle=True, random_state=42)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
		model.fit(X_train, y_train)
				
		
		# Make predictions on a grid
		x_min, x_max = X_init[:, 0].min() - 1, X_init[:, 0].max() + 1
		y_min, y_max = X_init[:, 1].min() - 1, X_init[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
		X_contour_init = np.c_[xx.ravel(), yy.ravel()];
		X_contour = np.array([fnc(X_contour_init) for fnc in basis]).T
		if "Gaussian" in selected_basis:
			temp = gaussian_basis(X_contour_init, 5)
			X_contour = np.concatenate([X_contour, temp], axis=1)
		X_contour = stdScale.transform(X_contour)
		print(X_contour.shape)
		Z = model.predict_proba(X_contour)
		Z = Z[:, 1].reshape(xx.shape)
		
		# Plot the contour
		st.write(f"### Dataset: {dataset_name}")
		plt.figure(figsize=(8, 6))
		plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
		plt.scatter(X_init[:, 0], X_init[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Probability Contour')
		st.pyplot(plt)
		
		# Show accuracy
		y_pred = model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		st.write(f"Accuracy: {accuracy:.2f}")

		# Print model weights and intercepts
		st.write("### Model Weights")
		for i, (weight, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
			st.write(f"Layer {i+1}:")
			st.write(weight)
			st.write(intercept)
			st.write("")

if __name__ == "__main__":
	main()
