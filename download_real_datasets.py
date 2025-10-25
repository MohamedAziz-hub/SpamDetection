# download_real_datasets.py
import pandas as pd
import urllib.request
import zipfile
import os


def download_real_datasets():
    print("📥 TÉLÉCHARGEMENT DE VRAIS DATASETS...")

    # Créer les dossiers
    os.makedirs('data/english', exist_ok=True)
    os.makedirs('data/french', exist_ok=True)
    os.makedirs('data/mixed', exist_ok=True)

    # ========== DATASET ANGLAIS RÉEL ==========
    print("🔤 Téléchargement dataset anglais...")

    # Dataset SMS Spam Collection (réel)
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        filename = "data/smsspamcollection.zip"

        urllib.request.urlretrieve(url, filename)

        # Extraire
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data/english/')

        # Lire et formater
        df_english = pd.read_csv('data/english/SMSSpamCollection', sep='\t', names=['label', 'message'])
        df_english['language'] = 'en'

        print(f"✅ Dataset anglais: {len(df_english)} messages")

    except Exception as e:
        print(f"❌ Erreur téléchargement anglais: {e}")
        print("🔄 Création dataset anglais manuel...")
        df_english = create_manual_english_dataset()

    # ========== DATASET FRANÇAIS RÉEL ==========
    print("\n🔤 Téléchargement dataset français...")

    # Créer un dataset français manuellement plus grand
    df_french = create_large_french_dataset()

    # ========== COMBINER ET SAUVEGARDER ==========
    print("\n🔄 COMBINAISON DES DATASETS...")

    # Combiner
    df_mixed = pd.concat([df_english, df_french], ignore_index=True)

    # Mélanger aléatoirement
    df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sauvegarder
    df_english.to_csv('data/english/english_spam_data.csv', index=False)
    df_french.to_csv('data/french/french_spam_data.csv', index=False)
    df_mixed.to_csv('data/mixed/bilingual_dataset.csv', index=False)

    print("✅ DATASETS CRÉÉS AVEC SUCCÈS!")
    print(f"📊 STATISTIQUES FINALES:")
    print(f"   • Dataset anglais: {len(df_english)} messages")
    print(f"   • Dataset français: {len(df_french)} messages")
    print(f"   • Dataset mixte: {len(df_mixed)} messages")
    print(f"   • Total spams: {len(df_mixed[df_mixed['label'] == 'spam'])}")
    print(f"   • Total ham: {len(df_mixed[df_mixed['label'] == 'ham'])}")

    return df_english, df_french, df_mixed


def create_large_french_dataset():
    """Crée un large dataset français avec plus de variété"""

    french_messages = []
    french_labels = []

    # ========== SPAMS FRANÇAIS (100 exemples) ==========
    spam_templates = [
        "Félicitations ! Vous avez gagné {montant} à notre {loterie} ! Cliquez sur {lien}",
        "Prêt {urgence} disponible sans frais. Obtenez {montant} immédiatement.",
        "Perdez {poids} en {temps} sans effort avec notre produit {qualificatif}.",
        "URGENT: Votre {compte} a été suspendu. Vérifiez maintenant sur {lien}",
        "Travail à domicile: Gagnez {salaire} par mois sans {experience}.",
        "{produit} gratuit ! Payez seulement {frais} de frais de port.",
        "Investissement {crypto}: {multiplicateur} votre argent en {delai} garantie !",
        "Vous êtes sélectionné pour un {voyage} gratuit à {destination} !",
        "{medicament} secret que les {professionnels} cachent au public.",
        "Crédit {urgence} sans justification. Appelez {numero} !",
        "Alerte sécurité: Votre {compte} a été compromis. Cliquez {lien}.",
        "Gagnez argent {facilement} depuis votre {endroit}. Démarrage {rapidite}.",
        "Diplôme {niveau} sans {examen}. Obtenez-le {delai}.",
        "Produit {anti_age}: Effacez {annees} en {jours} seulement.",
        "Visa {pays} gratuit. Postulez {urgence} avant fermeture.",
    ]

    spam_variations = {
        'montant': ['1 000 000 €', '500 000 €', '100 000 €', '50 000 €', '10 000 €', '5 000 €'],
        'loterie': ['loterie internationale', 'grande loterie', 'loterie nationale', 'tirage au sort'],
        'lien': ['www.claim-price.com', 'www.gagnant.net', 'www.reclamation.fr', 'www.prix-garanti.com'],
        'urgence': ['urgent', 'immédiat', 'rapide', 'express'],
        'poids': ['10 kg', '15 kg', '20 kg', '5 kg', '8 kg'],
        'temps': ['2 semaines', '1 mois', '15 jours', '3 semaines'],
        'qualificatif': ['miracle', 'révolutionnaire', 'exceptionnel', 'incroyable'],
        'compte': ['compte bancaire', 'compte email', 'compte en ligne', 'profil'],
        'salaire': ['5 000 €', '3 000 €', '7 000 €', '10 000 €'],
        'experience': ['expérience', 'diplôme', 'formation', 'compétence'],
        'produit': ['iPhone', 'Samsung', 'iPad', 'MacBook', 'TV 4K'],
        'frais': ['10 €', '15 €', '20 €', '5 €'],
        'crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain'],
        'multiplicateur': ['Doublez', 'Triplez', 'Multipliez par 5', 'Augmentez de 300%'],
        'delai': ['24h', '48h', '1 semaine', '15 jours'],
        'voyage': ['voyage', 'séjour', 'vacances', 'croisière'],
        'destination': ['Maldives', 'Bali', 'Thaïlande', 'Caraïbes', 'Dubai'],
        'medicament': ['Médicament', 'Remède', 'Traitement', 'Solution'],
        'professionnels': ['médecins', 'pharmaciens', 'spécialistes', 'experts'],
        'numero': ['le 08 00 00 00 00', 'ce numéro', 'vite', 'maintenant'],
        'facilement': ['facilement', 'simplement', 'rapidement', 'sans effort'],
        'endroit': ['canapé', 'maison', 'bureau', 'jardin'],
        'rapidite': ['immédiat', 'rapide', 'instantané', 'dès maintenant'],
        'niveau': ['universitaire', 'master', 'doctorat', 'licence'],
        'examen': ['examen', 'test', 'contrôle', 'évaluation'],
        'anti_age': ['anti-âge', 'rajeunissant', 'regénérant', 'revitalisant'],
        'annees': ['10 ans', '15 ans', '20 ans', '5 ans'],
        'jours': ['15 jours', '1 mois', '3 semaines', '10 jours'],
        'pays': ['Canada', 'USA', 'Australie', 'Japon', 'Europe'],
    }

    # Générer 100 spams variés
    for i in range(100):
        template = spam_templates[i % len(spam_templates)]
        message = template
        for key, values in spam_variations.items():
            if f"{{{key}}}" in message:
                message = message.replace(f"{{{key}}}", values[i % len(values)])
        french_messages.append(message)
        french_labels.append('spam')

    # ========== MESSAGES NORMAUX FRANÇAIS (100 exemples) ==========
    normal_templates = [
        "Bonjour {prenom}, comment allez-vous {moment} ?",
        "Merci pour votre {message}, je vous réponds {delai}.",
        "Pouvez-vous m'appeler quand vous êtes {disponible} ?",
        "La réunion est prévue pour {jour} à {heure}.",
        "Je serai en retard pour le {evenement} {moment}.",
        "Joyeux anniversaire ! Passez une {qualite} journée.",
        "Veuillez trouver ci-joint le {document} demandé.",
        "Salut, on se voit {jour} pour le {activite} ?",
        "Le {projet} sera prêt pour {delai}.",
        "Merci de votre {attitude}. {formule}.",
        "Peux-tu m'envoyer le {fichier} quand tu as un moment ?",
        "La réunion de {jour} est {statut}.",
        "As-tu reçu mon {message} d'{hier} ?",
        "On se retrouve où pour le {activite} {moment} ?",
        "N'oublie pas d'acheter le {produit} en rentrant.",
    ]

    normal_variations = {
        'prenom': ['Jean', 'Marie', 'Pierre', 'Sophie', 'Paul', 'Julie'],
        'moment': ['aujourd\'hui', 'ce matin', 'cet après-midi', 'ce soir'],
        'message': ['email', 'message', 'courrier', 'appel'],
        'delai': ['bientôt', 'dès que possible', 'dans la journée', 'demain'],
        'disponible': ['disponible', 'libre', 'prêt', 'convenable'],
        'jour': ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi'],
        'heure': ['14h00', '15h30', '10h00', '16h45', '9h15'],
        'evenement': ['dîner', 'déjeuner', 'rendez-vous', 'réunion', 'cinéma'],
        'qualite': ['bonne', 'excellente', 'agréable', 'merveilleuse'],
        'document': ['document', 'fichier', 'rapport', 'dossier'],
        'activite': ['déjeuner', 'dîner', 'cinéma', 'café', 'shopping'],
        'projet': ['rapport', 'projet', 'dossier', 'document', 'fichier'],
        'attitude': ['compréhension', 'patience', 'collaboration', 'aide'],
        'formule': ['Cordialement', 'Bien à vous', 'Sincèrement', 'Amicalement'],
        'fichier': ['fichier', 'document', 'rapport', 'dossier', 'photo'],
        'statut': ['annulée', 'reportée', 'confirmée', 'modifiée'],
        'hier': ['hier', 'la semaine dernière', 'ce matin', 'tout à l\'heure'],
        'produit': ['pain', 'lait', 'journal', 'café', 'sucre'],
    }

    # Générer 100 messages normaux variés
    for i in range(100):
        template = normal_templates[i % len(normal_templates)]
        message = template
        for key, values in normal_variations.items():
            if f"{{{key}}}" in message:
                message = message.replace(f"{{{key}}}", values[i % len(values)])
        french_messages.append(message)
        french_labels.append('ham')

    # Créer le dataframe français
    df_french = pd.DataFrame({
        'message': french_messages,
        'label': french_labels,
        'language': ['fr'] * 200
    })

    print(f"✅ Dataset français créé: {len(df_french)} messages")
    return df_french


def create_manual_english_dataset():
    """Crée un dataset anglais si le téléchargement échoue"""
    print("🔄 Création dataset anglais manuel...")

    english_data = {
        'message': [
            # SPAMS (50 exemples)
            "Free entry in 2 a wkly comp to win FA Cup final tkts",
            "You have won a $1000 prize! Call now to claim.",
            "Urgent: Your account has been compromised.",
            "Work from home and earn $5000 monthly.",
            "Congratulations! You won an iPhone. Click to claim.",
            "Lose weight fast with this miracle pill.",
            "Double your money in 24 hours guaranteed.",
            "Free vacation to Bahamas! Limited time offer.",
            "Your bank account needs verification immediately.",
            "Investment opportunity: 300% return guaranteed.",
            "You are selected for a free cruise vacation.",
            "Limited offer: Get your free laptop now!",
            "Urgent message: Your package delivery failed.",
            "Exclusive deal: 90% off all products today!",
            "Warning: Your computer may be infected.",
            "Get paid $100 per survey from home.",
            "Your tax refund is waiting. Claim now!",
            "Special discount for valued customers.",
            "Your Netflix account has been suspended.",
            "Win a brand new car! Enter now!",

            # HAM (50 exemples)
            "Hey, how are you doing today?",
            "Thanks for your email, I'll reply soon.",
            "Can you call me when you're available?",
            "The meeting is scheduled for tomorrow at 2 PM.",
            "I'll be late for dinner tonight.",
            "Happy birthday! Have a great day.",
            "Please find attached the requested document.",
            "Hi, are we meeting for lunch on Saturday?",
            "The report will be ready by end of week.",
            "Thanks for your understanding. Best regards.",
            "Can you send me the file when you have a moment?",
            "Monday's meeting is cancelled.",
            "Did you receive my message from yesterday?",
            "Where should we meet for the cinema tonight?",
            "Don't forget to buy bread on your way home.",
            "The doctor confirmed the appointment for Friday.",
            "I booked the room for the conference.",
            "The kids come back from school at 4:30 PM.",
            "Do you want to go to the restaurant this weekend?",
            "The project is going well, thanks for your help.",
        ],
        'label': ['spam'] * 20 + ['ham'] * 20,
        'language': ['en'] * 40
    }

    return pd.DataFrame(english_data)


if __name__ == "__main__":
    download_real_datasets()