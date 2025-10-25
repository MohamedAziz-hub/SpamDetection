# download_datasets.py - VERSION ENRICHIE
import pandas as pd
import os


def create_enriched_datasets():
    print("Création des datasets enrichis...")

    # ========== DATASET FRANÇAIS ENRICHIE ==========
    french_messages = [
        # SPAMS (30 exemples)
        "Félicitations ! Vous avez gagné 1 000 000 € à notre loterie !",
        "Prêt urgent sans frais. Obtenez 5000 € immédiatement.",
        "Perdez 10 kg en 2 semaines sans effort. Produit miracle !",
        "URGENT: Votre compte bancaire a été suspendu. Vérifiez maintenant.",
        "Travail à domicile: Gagnez 5000 € par mois sans expérience.",
        "iPhone gratuit ! Payez seulement 10 € de frais de port.",
        "Investissement crypto: Doublez votre argent en 24h garantie !",
        "Vous êtes sélectionné pour un voyage gratuit aux Maldives !",
        "Médicament secret que les médecins cachent au public.",
        "Crédit immédiat sans justification. Appelez vite !",
        "Alerte sécurité: Votre compte a été compromis. Cliquez ici.",
        "Gagnez argent facilement depuis votre canapé. Démarrage immédiat.",
        "Diplôme universitaire sans examen. Obtenez-le rapidement.",
        "Produit anti-âge: Effacez 20 ans en 15 jours seulement.",
        "Visa Canada gratuit. Postulez maintenant avant fermeture.",
        "Héritage surprise: 250 000 € vous attendent. Répondez vite !",
        "Cartes crédit pré-approuvées sans vérification de revenu.",
        "Logiciel trading: 300% profit mensuel garanti contrat signé.",
        "Stage rémunéré 200 €/jour. Travaillez de chez vous.",
        "Épargne garantie 15% rendement. Sans risque capital.",
        "Téléphone portable dernier cri gratuit. Offre limitée.",
        "Maison à 1 €. Programme gouvernemental secret. Découvrez !",
        "Assurance vie gratuite. Protégez votre famille maintenant.",
        "Énergie libre: Réduisez facture électricité 90%. Méthode secrète.",
        "Recrutement: Ambassadeurs marque. Salaire fixe + primes.",
        "Cryptomonnaie: Investissez maintenant et devenez millionnaire.",
        "Formation trading offerte. Devenez expert en 7 jours.",
        "Achat groupé: iPhone 50% moins cher. Stocks limités.",
        "Bonus exceptionnel: 500 € offerts pour ouverture compte.",
        "Parfum de luxe gratuit. Payez seulement l'expédition.",

        # MESSAGES NORMAUX (30 exemples)
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Merci pour votre email, je vous réponds bientôt.",
        "Pouvez-vous m'appeler quand vous êtes disponible ?",
        "La réunion est prévue pour demain à 14 heures.",
        "Je serai en retard pour le dîner ce soir.",
        "Joyeux anniversaire ! Passez une bonne journée.",
        "Veuillez trouver ci-joint le document demandé.",
        "Salut, on se voit samedi pour le déjeuner ?",
        "Le rapport sera prêt pour la fin de semaine.",
        "Merci de votre compréhension. Cordialement.",
        "Peux-tu m'envoyer le fichier quand tu as un moment ?",
        "La réunion de lundi est annulée.",
        "As-tu reçu mon message d'hier ?",
        "On se retrouve où pour le cinéma ce soir ?",
        "N'oublie pas d'acheter le pain en rentrant.",
        "Le médecin a confirmé le rendez-vous pour vendredi.",
        "J'ai réservé la salle pour la conférence.",
        "Les enfants rentrent de l'école à 16h30.",
        "Tu veux qu'on aille au restaurant ce weekend ?",
        "Le projet avance bien, merci pour ton aide.",
        "La météo annonce de la pluie pour demain.",
        "J'ai terminé la lecture du livre que tu m'as prêté.",
        "Le train part à 18h45 de la gare centrale.",
        "Penses-tu venir à la fête d'anniversaire de Paul ?",
        "J'ai acheté les billets pour le concert.",
        "La réparation de la voiture coûtera 350 €.",
        "Super idée pour les vacances ! J'adore le plan.",
        "Mon frère arrive en visite ce weekend.",
        "Le colis est arrivé ce matin, merci !",
        "On se voit à l'entrée du métro à 19h ?"
    ]

    french_labels = ['spam'] * 30 + ['ham'] * 30
    french_languages = ['fr'] * 60

    # ========== DATASET ANGLAIS ENRICHIE ==========
    english_messages = [
        # SPAMS (30 exemples)
        "Congratulations! You won $1,000,000 in our lottery!",
        "Urgent loan without fees. Get $5000 immediately.",
        "Lose 10 kg in 2 weeks without effort. Miracle product!",
        "URGENT: Your bank account has been suspended. Verify now.",
        "Work from home: Earn $5000 monthly without experience.",
        "Free iPhone! Pay only $10 shipping fees.",
        "Crypto investment: Double your money in 24h guaranteed!",
        "You are selected for a free trip to Maldives!",
        "Secret medicine that doctors hide from the public.",
        "Instant credit without verification. Call now!",
        "Security alert: Your account has been compromised. Click here.",
        "Make money easily from your couch. Start immediately.",
        "University degree without exams. Get it quickly.",
        "Anti-aging product: Erase 20 years in just 15 days.",
        "Free Canada visa. Apply now before closing.",
        "Surprise inheritance: $250,000 waiting for you. Reply fast!",
        "Pre-approved credit cards without income check.",
        "Trading software: 300% monthly profit guaranteed signed contract.",
        "Paid internship $200/day. Work from home.",
        "Guanteed savings 15% return. No capital risk.",
        "Latest smartphone free. Limited offer.",
        "House for $1. Secret government program. Discover!",
        "Free life insurance. Protect your family now.",
        "Free energy: Reduce electricity bill 90%. Secret method.",
        "Recruitment: Brand ambassadors. Fixed salary + bonuses.",
        "Cryptocurrency: Invest now and become millionaire.",
        "Free trading training. Become expert in 7 days.",
        "Group purchase: iPhone 50% cheaper. Limited stocks.",
        "Exceptional bonus: $500 free for account opening.",
        "Luxury perfume free. Pay only shipping.",

        # NORMAL MESSAGES (30 exemples)
        "Hello, how are you doing today?",
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
        "The weather forecast says rain for tomorrow.",
        "I finished reading the book you lent me.",
        "The train departs at 6:45 PM from central station.",
        "Are you coming to Paul's birthday party?",
        "I bought the tickets for the concert.",
        "The car repair will cost $350.",
        "Great idea for the holidays! I love the plan.",
        "My brother is visiting this weekend.",
        "The package arrived this morning, thank you!",
        "Shall we meet at the metro entrance at 7 PM?"
    ]

    english_labels = ['spam'] * 30 + ['ham'] * 30
    english_languages = ['en'] * 60

    # Créer les dataframes
    df_french = pd.DataFrame({
        'message': french_messages,
        'label': french_labels,
        'language': french_languages
    })

    df_english = pd.DataFrame({
        'message': english_messages,
        'label': english_labels,
        'language': english_languages
    })

    # Combiner
    df_mixed = pd.concat([df_french, df_english], ignore_index=True)

    # Sauvegarder
    os.makedirs('data/french', exist_ok=True)
    os.makedirs('data/english', exist_ok=True)
    os.makedirs('data/mixed', exist_ok=True)

    df_french.to_csv('data/french/french_spam_data.csv', index=False)
    df_english.to_csv('data/english/english_spam_data.csv', index=False)
    df_mixed.to_csv('data/mixed/bilingual_dataset.csv', index=False)

    print("✅ Datasets enrichis créés !")
    print(f"📊 Statistiques:")
    print(f"   Français: {len(df_french)} messages")
    print(f"   Anglais: {len(df_english)} messages")
    print(f"   Total: {len(df_mixed)} messages")
    print(f"   Spams: {len(df_mixed[df_mixed['label'] == 'spam'])}")
    print(f"   Normaux: {len(df_mixed[df_mixed['label'] == 'ham'])}")


if __name__ == "__main__":
    create_enriched_datasets()