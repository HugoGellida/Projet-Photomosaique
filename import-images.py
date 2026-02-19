import os
import requests
from tqdm import tqdm
import time

'''
  ======== TUTORIEL ========

Avant de pouvoir exécuter ce code, il faut que vous puissiez être capable de faire les préparations nécessaires. Nous utilisons
la base de données d'image appelée Pexels, contenant des dizaines de milliers d'images

Tout d'abord, sur votre terminal, exécuter la commande suivante: pip install requests tqdm
    -> Cela vous permettra d'avoir une jolie barre de progression qui s'affiche lors du téléchargement
    
Puis, installez sur Visual Studio Code (ou VSCodium) l'extension: Better Comments
    -> Cela vous permettra d'avoir une manière de personnaliser vos commentaires

Exemple:
    - #! Ce commentaire apparaît en rouge
    - #? Celui-ci en bleu
    - #* Celui-ci en vert doux
    - #TODO Celui-ci en orange

Ensuite, allez sur le site Pexels, et connectez/inscrivez-vous. Une fois cela fais, il vous faut une clé API. Il y aura remplir
un questionnaire ou quelque chose dans le genre, un peu relou. Une fois cela fait, copiez cette clé. Vous devez la collez
dans la variable globale API_KEY dans l'onglet CONFIGURATION

#* Bravo, vous avez terminez les préparations et pouvez exécuter librement le code. Si vous voulez des explications, n'hésitez
#* pas à regardez mes commentaires !

  ===============================
'''

# ======== CONFIGURATION ========
#? Nous avons besoin de communiquer quelles données nous voulons et de spécifier chaque paramètres, afin d'avoir le même dataset!
API_KEY = "wz57tsZ18vrNp421HZirfczUxgnk3pUH1KDuIMCNc3hb5TFon5uIOr76"    #! COLLEZ VOTRE CLE ICI
QUERY = "nature"                                                        # Le thème d'images que vous voulez. Chaque images ont des tags, on met un tag ici pour filtrer notre recherche.
PER_PAGE = 80                                                           # Maximum autorisé par Pexels. C'est le nombre d'images par pages que nous voulons prendre.
MAX_PAGES = 10                                                          # Le nombre de pages que vous voulez. 
SAVE_DIR = "pexels_images"                                              # Le repertoire dans lequel nous voulons mettre nos images. #! ATTENTION SI VOUS CHANGEZ LE NOM, SOYEZ SUR DE CHANGER LE NOM AUSSI DANS LE FICHIER .gitignore!!!
# ===============================

# ======== FUNCTIONS ========
def download_image(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200: # Si on à l'autorisation
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                
# ===============================

# ======== MAIN ========

headers = {
    "Authorization": API_KEY
}

os.makedirs(SAVE_DIR, exist_ok=True) # On créer le repertoire s'il n'existe pas


for page in range(1, MAX_PAGES + 1):
    print(f"Fetching page {page}...")

    url = f"https://api.pexels.com/v1/search?query={QUERY}&per_page={PER_PAGE}&page={page}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.text)
        break

    data = response.json()
    photos = data.get("photos", [])

    if not photos:
        print("No more photos found.")
        break

    for photo in tqdm(photos):
        img_url = photo["src"]["original"]
        img_id = photo["id"]
        filename = os.path.join(SAVE_DIR, f"{img_id}.jpg")
        download_image(img_url, filename)

    time.sleep(1)  # Ici, j'ai eu beaucoup de mal, mais l'API à un certaine limite à respecter. Donc je met un delai pour éviter qu'il nous crache dessus

print("Download complete!")