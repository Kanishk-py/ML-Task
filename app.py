import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as t

class MCDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

class NeuralNetwork:
	def __init__(self, hidden_layers, learning_rate, batch_size, epochs, input_size, activation, mc_dropout=False, dropout_rate=0.2):
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.mc_dropout = mc_dropout
		self.input_size = input_size
		self.output_size = 2
		self.dropout_rate = dropout_rate
		self.activation = activation
		self.model = self._build_model()

	def _build_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.InputLayer(input_shape=(self.input_size,)))
		
		for neurons in self.hidden_layers:
			model.add(tf.keras.layers.Dense(neurons, activation=self.activation))
			if self.mc_dropout:
				model.add(MCDropout(self.dropout_rate))

		model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))
		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		return model

	def train(self, x_train, y_train):
		history = self.model.fit(
			x_train, y_train,
			batch_size=self.batch_size,
			epochs=self.epochs,
			validation_split=0.1
		)
		return history

	def test(self, x_test, y_test):
		loss, accuracy = self.model.evaluate(x_test, y_test)
		return loss, accuracy

# Create some Datasets
datasets = {
	'Circles': make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=42),
	'Moons': make_moons(n_samples=400, noise=0.1, random_state=42),
	'Linearly Separable': make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
}

# Custom basis transformation functions
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
	st.sidebar.header("Tune Nueral Network")
	
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
		# 'Gaussian': lambda x: gaussian_basis(x, 5)
	}
	selected_basis = st.sidebar.multiselect("Select Basis Functions", list(basis_functions.keys()) + ['Gaussian'], default=['X1', 'X2'])
	
	# Hyperparameter controls
	layer_sizes = st.sidebar.text_input("Layer Sizes (comma-separated)", "3,3,3")
	layer_sizes = [int(neurons) for neurons in layer_sizes.split(",")]
	
	learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
	epochs = st.sidebar.slider("Epochs", 10, 500, 40, 10)
	batch_size = st.sidebar.slider("Batch Size", 1, 30, 10, 1)
	test_size = st.sidebar.slider("Test Size", 0.0, 1.0, 0.2, 0.05)
	activation = st.sidebar.selectbox("Activation Function", ["tanh", "relu", "logistic"])
	
	mc_dropout = st.sidebar.checkbox("MC Dropout", value=False)
	dropout_val = 0.2
	if mc_dropout:
		dropout_val = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, 0.05)


	# Button to start training
	trainBtn = st.sidebar.button("Train Model")
	if trainBtn:
		basis = [basis_functions[basis] for basis in selected_basis  if basis != "Gaussian"]
		X = np.array([basis_function(X_init) for basis_function in basis]).T
		print(X.shape)
		if "Gaussian" in selected_basis:
			if X.shape[0] == 0:
				X = gaussian_basis(X_init, 5)
			else:
				X = np.concatenate([X, gaussian_basis(X_init, 5)], axis=1)
			print(X.shape)
		stdScale = StandardScaler().fit(X)
		X = stdScale.transform(X)


		model = NeuralNetwork(hidden_layers=layer_sizes, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, input_size=X.shape[1], activation=activation, mc_dropout=mc_dropout, dropout_rate=dropout_val)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
		history = model.train(X_train, y_train)
		loss, accuracy = model.test(X_test, y_test)

		
		# Train the neural network
		# model = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=epochs, learning_rate_init=learning_rate, activation=activation, solver="adam", batch_size=batch_size, shuffle=True, random_state=42)
		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
		# model.fit(X_train, y_train)
				
		
		# Make predictions on a grid
		x_min, x_max = X_init[:, 0].min() - 1, X_init[:, 0].max() + 1
		y_min, y_max = X_init[:, 1].min() - 1, X_init[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
		X_contour_init = np.c_[xx.ravel(), yy.ravel()];
		X_contour = np.array([fnc(X_contour_init) for fnc in basis]).T
		if "Gaussian" in selected_basis:
			if X_contour.shape[0] == 0:
				X_contour = gaussian_basis(X_contour_init, 5)
			else:
				X_contour = np.concatenate([X_contour, gaussian_basis(X_contour_init, 5)], axis=1)
		X_contour = stdScale.transform(X_contour)
		
		# Z = model.model.predict(X_contour)
		Z = np.array([model.model(X_contour, training=True).numpy()[:,1] for _ in range(30)]).T.mean(axis=1)
		Z = Z.reshape(xx.shape)
		
		# Plot the contour
		st.write(f"### Dataset: {dataset_name}")
		plt.figure(figsize=(8, 6))
		plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
		plt.colorbar()
		plt.scatter(X_init[:, 0], X_init[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Probability Contour')
		st.pyplot(plt)

		# Print the loss and accuracy
		st.write(f"Accuracy: {accuracy:.2f}")
		st.write(f"Loss: {loss:.2f}")

		
		# Print model weights and intercepts
		st.write("Model Weights:")
		for layer in model.model.layers:
			if hasattr(layer, 'weights') and layer.weights:  # Check if the layer has weights and they are not empty
				weights = layer.get_weights()[0]
				st.write(f"Weights shape for Layer {layer.name}: {weights.shape}")
				st.write(weights)

			if hasattr(layer, 'bias') and layer.weights:  # Check if the layer has biases and they are not empty
				bias = layer.get_weights()[1]
				st.write(f"Bias shape for Layer {layer.name}: {bias.shape}")
				st.write(bias)

			# if isinstance(layer, tf.keras.layers.Dropout):  # Handle dropout layers separately
				# st.write(f"Dropout rate for Layer {layer.name}: {layer.rate}")

if __name__ == "__main__":
	main()
