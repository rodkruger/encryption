from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression, SGDClassifier
from concrete.ml.common.utils import FheMode

# Lets create a synthetic data-set
x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

# Encrypt data
# ??
# AES/256 (?) -> Mais usado no mundo
# RSA -> Inviolável (custoso)

# Avaliar diferentes algoritmos de criptografia (custosos e não custosos)
# Gerar chave (privada e pública)
# Medir em termos de tempo o impacto
# Olhar as bibliotecas de criptografia homórfica (C/C++)

# Split the data-set into a train and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Now we train in the clear and quantize the weights
model = SGDClassifier(n_bits=8, parameters_range=, fit_encrypted=True)
model.fit(X_train, y_train)

# We can simulate the predictions in the clear
y_pred_clear = model.predict(X_test)

# We then compile on a representative set
model.compile(X_train)

# Finally we run the inference on encrypted inputs !
y_pred_fhe = model.predict(X_test, fhe="execute")

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print(f"Similarity: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")