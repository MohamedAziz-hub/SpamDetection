# train_bilingual.py - VERSION SIMPLE ET EFFICACE
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=== ENTRAÎNEMENT SIMPLE DU MODÈLE BILINGUE ===")

# Charger les données
df = pd.read_csv('data/mixed/bilingual_dataset_processed.csv')
print(f"📊 Données chargées: {len(df)} messages")

# Préparer les données
X = df['cleaned_message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Vectorizer SIMPLE
vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(X)

print(f"📈 Features créées: {X_tfidf.shape}")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🎯 Division: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# Modèles SIMPLES
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Entraînement et évaluation
best_model = None
best_score = 0

print("\n🤖 ENTRAÎNEMENT...")
for name, model in models.items():
    print(f"🔧 {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   ✅ Accuracy: {accuracy:.4f}")

    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

# Évaluation finale
y_pred_best = best_model.predict(X_test)
print(f"\n🏆 MEILLEUR MODÈLE: {best_name}")
print(f"📈 Accuracy finale: {best_score:.2%}")

print("\n📋 RAPPORT DÉTAILLÉ:")
print(classification_report(y_test, y_pred_best, target_names=['HAM', 'SPAM']))

# Sauvegarde
joblib.dump(best_model, 'models/bilingual_spam_model.pkl')
joblib.dump(vectorizer, 'models/bilingual_vectorizer.pkl')

print("💾 Modèles sauvegardés!")

# Tests rapides
print("\n🧪 TESTS RAPIDES:")
test_messages = [
    "Félicitations ! Vous avez gagné 1 000 000 € !",  # SPAM FR
    "Bonjour, comment allez-vous aujourd'hui ?",  # HAM FR
    "Congratulations! You won $1000000!",  # SPAM EN
    "Hello, how are you doing today?"  # HAM EN
]

for msg in test_messages:
    cleaned = msg.lower()
    vectorized = vectorizer.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    proba = best_model.predict_proba(vectorized)[0]

    result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    confidence = max(proba)

    print(f"📨 '{msg[:30]}...' → {result} ({confidence:.2%})")

print(f"\n🎉 TERMINÉ! Précision: {best_score:.2%}")