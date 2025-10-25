# app_bilingual.py - VERSION CORRIGÉE
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import os

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Spam Bilingue",
    page_icon="🌍",
    layout="centered"
)

# Titre bilingue
st.title("🌍 Détecteur de Spam Intelligent")
st.markdown("""
**Détecte les spams en Français et Anglais** 🇫🇷 🇬🇧

Cette application utilise l'**Intelligence Artificielle** pour identifier les messages indésirables.
""")


class BilingualSpamDetector:
    def __init__(self):
        # DEBUG: Afficher le chemin actuel
        current_dir = os.getcwd()
        st.sidebar.write(f"📁 Dossier actuel: {current_dir}")

        # Chemins complets vers les modèles
        model_path = os.path.join('models', 'bilingual_spam_model.pkl')
        vectorizer_path = os.path.join('models', 'bilingual_vectorizer.pkl')

        st.sidebar.write(f"🔍 Recherche modèle: {model_path}")
        st.sidebar.write(f"🔍 Recherche vectorizer: {vectorizer_path}")

        # Vérifier si les fichiers existent
        model_exists = os.path.exists(model_path)
        vectorizer_exists = os.path.exists(vectorizer_path)

        st.sidebar.write(f"📄 Modèle existe: {model_exists}")
        st.sidebar.write(f"📄 Vectorizer existe: {vectorizer_exists}")

        # Charger le modèle et vectorizer
        try:
            if model_exists and vectorizer_exists:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.model_loaded = True
                st.sidebar.success("✅ Modèles chargés avec succès!")
            else:
                st.error("❌ Fichiers de modèle manquants")
                self.model_loaded = False

        except Exception as e:
            st.error(f"❌ Erreur de chargement: {e}")
            self.model_loaded = False

        # Initialiser les outils de prétraitement
        try:
            nltk.data.find('corpora/stopwords')
            self.stopwords_fr = set(stopwords.words('french'))
            self.stopwords_en = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords_fr = set(stopwords.words('french'))
            self.stopwords_en = set(stopwords.words('english'))

        self.stemmer_fr = SnowballStemmer("french")
        self.stemmer_en = PorterStemmer()

    def detect_language(self, text):
        """Détecte la langue du texte"""
        text_lower = text.lower()

        french_indicators = ['le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'à', 'vous', 'nous']
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'you', 'we']

        french_count = sum(1 for word in french_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)

        return 'fr' if french_count > english_count else 'en'

    def clean_text(self, text, language='auto'):
        """Nettoie le texte pour l'analyse"""
        if not text or text.strip() == '':
            return ''

        if language == 'auto':
            language = self.detect_language(text)

        # Nettoyage de base
        text = text.lower()

        if language == 'fr':
            text = re.sub(r'[^a-zA-Zàâäéèêëîïôöùûüç\s]', '', text)
            stop_words = self.stopwords_fr
            stemmer = self.stemmer_fr
        else:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            stop_words = self.stopwords_en
            stemmer = self.stemmer_en

        # Tokenization et stemming
        words = text.split()
        cleaned_words = []

        for word in words:
            if word not in stop_words and len(word) > 2:
                stemmed_word = stemmer.stem(word)
                cleaned_words.append(stemmed_word)

        return ' '.join(cleaned_words)

    def predict(self, text):
        """Fait une prédiction sur le texte"""
        if not self.model_loaded:
            return None, None, None, None

        # Nettoyer le texte
        language = self.detect_language(text)
        cleaned_text = self.clean_text(text, language)

        # Vectoriser
        text_vectorized = self.vectorizer.transform([cleaned_text])

        # Prédire
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]

        return prediction, probability, language, cleaned_text


# Initialiser le détecteur
detector = BilingualSpamDetector()

# Interface utilisateur
if detector.model_loaded:
    # Zone de texte
    user_input = st.text_area(
        "📝 Entrez le message à analyser:",
        placeholder="Collez votre email ou message ici (Français ou Anglais)...",
        height=150
    )

    # Options
    col1, col2 = st.columns(2)
    with col1:
        auto_detect = st.checkbox("Détection automatique de langue", value=True)
    with col2:
        if not auto_detect:
            language_choice = st.radio("Choisir la langue:", ["Français", "Anglais"])
        else:
            language_choice = "Auto"

    # Bouton d'analyse
    if st.button("🔍 Analyser le message", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyse en cours..."):
                # Prédiction
                lang_param = 'auto' if auto_detect else language_choice.lower()[:2]
                prediction, probability, detected_lang, cleaned_text = detector.predict(user_input)

                if prediction is not None:
                    # Affichage des résultats
                    st.markdown("---")
                    st.subheader("📊 Résultats de l'analyse")

                    # Résultat principal
                    if prediction == 1:  # SPAM
                        st.error(f"🚨 **SPAM DÉTECTÉ**")
                        confidence = probability[1]
                        st.metric("Niveau de confiance", f"{confidence:.2%}")
                    else:  # HAM
                        st.success(f"✅ **MESSAGE NORMAL**")
                        confidence = probability[0]
                        st.metric("Niveau de confiance", f"{confidence:.2%}")

                    # Détails techniques
                    with st.expander("🔧 Détails techniques"):
                        st.write(f"**Langue détectée:** {detected_lang.upper()}")
                        st.write(f"**Probabilité SPAM:** {probability[1]:.4f}")
                        st.write(f"**Probabilité HAM:** {probability[0]:.4f}")
                        st.write(f"**Texte nettoyé:** {cleaned_text}")
        else:
            st.warning("⚠️ Veuillez entrer un message à analyser.")
else:
    st.error("""
    ❌ **Modèle non chargé**

    Pour résoudre ce problème:

    1. **Vérifiez que les fichiers existent:**
    ```bash
    ls models/
    ```

    2. **Si les fichiers manquent, ré-entraînez le modèle:**
    ```bash
    python train_bilingual.py
    ```

    3. **Redémarrez l'application:**
    ```bash
    streamlit run app_bilingual.py
    ```
    """)

# Section d'information
with st.sidebar:
    st.header("🎯 Exemples à tester")

    st.subheader("🇫🇷 Spams Français")
    st.code("Félicitations ! Vous avez gagné 1 000 000 € !")

    st.subheader("🇬🇧 English Spams")
    st.code("Congratulations! You won $1000000!")

    st.subheader("📧 Messages normaux")
    st.code("Bonjour, comment allez-vous ?")
    st.code("Hello, how are you doing?")