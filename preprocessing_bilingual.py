# preprocessing_bilingual.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import os

# Télécharger les stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Téléchargement des stopwords...")
    nltk.download('stopwords')


class BilingualTextPreprocessor:
    def __init__(self):
        # Initialiser les stemmers et stopwords pour les deux langues
        self.stemmer_fr = SnowballStemmer("french")
        self.stemmer_en = PorterStemmer()
        self.stopwords_fr = set(stopwords.words('french'))
        self.stopwords_en = set(stopwords.words('english'))

    def detect_language(self, text):
        """Détecte si le texte est français ou anglais"""
        text_lower = text.lower()

        # Mots caractéristiques du français
        french_indicators = ['le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'à', 'dans', 'pour', 'vous', 'nous',
                             'je', 'tu', 'il', 'elle', 'on']
        french_count = sum(1 for word in french_indicators if word in text_lower)

        # Mots caractéristiques de l'anglais
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'you', 'we', 'i', 'he', 'she', 'it', 'they']
        english_count = sum(1 for word in english_indicators if word in text_lower)

        if french_count > english_count:
            return 'fr'
        else:
            return 'en'

    def clean_text(self, text, language='auto'):
        """Nettoie le texte selon la langue"""
        if pd.isna(text) or text.strip() == '':
            return ''

        # Détection automatique de la langue si non spécifiée
        if language == 'auto':
            language = self.detect_language(text)

        # 1. Convertir en minuscules
        text = text.lower()

        # 2. Supprimer la ponctuation (conserver les accents pour le français)
        if language == 'fr':
            text = re.sub(r'[^a-zA-Zàâäéèêëîïôöùûüç\s]', '', text)
        else:
            text = re.sub(r'[^a-zA-Z\s]', '', text)

        # 3. Tokenization
        words = text.split()

        # 4. Choisir les stopwords et stemmer selon la langue
        if language == 'fr':
            stop_words = self.stopwords_fr
            stemmer = self.stemmer_fr
        else:
            stop_words = self.stopwords_en
            stemmer = self.stemmer_en

        # 5. Supprimer les stopwords et appliquer le stemming
        cleaned_words = []
        for word in words:
            if word not in stop_words and len(word) > 2:
                stemmed_word = stemmer.stem(word)
                cleaned_words.append(stemmed_word)

        return ' '.join(cleaned_words)

    def prepare_dataset(self, df, text_column='message', label_column='label', lang_column='language'):
        """Prépare un dataset complet avec nettoyage"""
        print("Nettoyage des données...")

        # Appliquer le nettoyage selon la langue
        df['cleaned_message'] = df.apply(
            lambda row: self.clean_text(row[text_column], row.get(lang_column, 'auto')),
            axis=1
        )

        # Convertir les labels en numérique
        df['label_num'] = df[label_column].map({'ham': 0, 'spam': 1})

        # Supprimer les messages vides après nettoyage
        initial_count = len(df)
        df = df[df['cleaned_message'].str.strip() != '']
        final_count = len(df)

        print(f"Messages après nettoyage: {final_count}/{initial_count}")

        return df


def main():
    preprocessor = BilingualTextPreprocessor()

    # Traiter le dataset mixte
    print("=== PRÉPARATION DU DATASET BILINGUE ===")
    df_mixed = pd.read_csv('data/mixed/bilingual_dataset.csv')
    df_processed = preprocessor.prepare_dataset(df_mixed)

    # Sauvegarder
    df_processed.to_csv('data/mixed/bilingual_dataset_processed.csv', index=False)
    print("✅ Dataset bilingue nettoyé sauvegardé!")

    # Aperçu
    print("\n=== APERÇU DES DONNÉES NETTOYÉES ===")
    print("Exemples français:")
    french_samples = df_processed[df_processed['language'] == 'fr'].head(3)
    for _, row in french_samples.iterrows():
        print(f"Original: {row['message'][:50]}...")
        print(f"Nettoyé:  {row['cleaned_message']}")
        print(f"Label: {row['label']} | Langue: {row['language']}")
        print()

    print("Exemples anglais:")
    english_samples = df_processed[df_processed['language'] == 'en'].head(3)
    for _, row in english_samples.iterrows():
        print(f"Original: {row['message'][:50]}...")
        print(f"Nettoyé:  {row['cleaned_message']}")
        print(f"Label: {row['label']} | Langue: {row['language']}")
        print()


if __name__ == "__main__":
    main()