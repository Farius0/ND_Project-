from PIL import Image
import numpy as np, os, matplotlib.pyplot as plt, cv2, random
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from skimage.segmentation import flood_fill
from collections import deque, defaultdict
import heapq  # File de priorité optimisée
from pathlib import Path

# Définition des chemins

images_dir = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"

def resize_image(image_np, size=(256, 256)):
    """
    Redimensionne une image NumPy en un certain taille.

    Parameters:
    ----------
    image_np : np.ndarray
        Image sous forme de tableau NumPy.
    size : tuple, optional  (default=(256, 256))
        Taille de l'image redimensionnée.

    Returns:
    -------
    np.ndarray
        Image redimensionnée.
    """
    image_pil = Image.fromarray(image_np.astype(np.uint8))  # Convertir en PIL
    image_pil = image_pil.resize(size, Image.LANCZOS)  # Redimensionnement
    return np.array(image_pil)  # Convertir à nouveau en numpy array


# Charger une image aléatoire
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
rand = np.random.randint(0, len(image_files))
# rand = 903

# Ouvrir l'image et la convertir en niveaux de gris
img = Image.open(image_files[rand]).convert('L')  # Conversion en niveaux de gris
# img_color = Image.open(image_files[rand])  # Image couleur
img = resize_image(np.array(img))  # Conversion en tableau NumPy
# img_color = resize_image(np.array(img_color))  # Conversion en tableau NumPy

################################################################################################# Seuillage #################################################################################################

# # Définition du seuil
# threshold = 125

# # Application du seuillage
# binary_img = (img > threshold).astype(np.uint8) * 255  # Seuillage binaire (0 ou 255)

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].imshow(binary_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Seuillage avec seuil {threshold}', fontsize=15)

# plt.show()

################################################################################################# Seuillage avec histogramme #################################################################################################

# # Calcul de l'histogramme
# histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255)) # Calcul de l'histogramme

# # Affichage de l'image et de son histogramme
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].plot(bin_edges[:-1], histogram, color='black')
# ax[1].set_title('Histogramme des niveaux de gris', fontsize=15)
# ax[1].set_xlabel('Niveau de gris')
# ax[1].set_ylabel('Nombre de pixels')

# plt.show()

################################################################################################# Seuillage adaptatif #################################################################################################
# # Initialisation du seuil
# S_old = np.mean(img)  # Moyenne des intensités
# epsilon = 1e-3  # Critère d'arrêt

# while True:
#     # Séparation en deux groupes
#     G1 = img[img <= S_old]
#     G2 = img[img > S_old]

#     # Calcul des nouvelles moyennes
#     if len(G1) == 0 or len(G2) == 0:
#         break  # Évite la division par zéro
#     m1 = np.mean(G1)
#     m2 = np.mean(G2)

#     # Nouveau seuil
#     S_new = (m1 + m2) / 2

#     # Vérification du critère d'arrêt
#     if abs(S_new - S_old) < epsilon:
#         break
#     S_old = S_new

# # Seuillage final
# binary_img = (img > S_new).astype(np.uint8) * 255  # Seuillage binaire

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].imshow(binary_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Seuillage automatique (S={S_new:.2f})', fontsize=15)

# plt.show()

################################################################################################# Seuillage Multiple #################################################################################################
# # Définition de plusieurs seuils (choisis manuellement ou à partir de l'histogramme)
# thresholds = [80, 160]  # Exemple de seuillage en 3 classes

# # Application du seuillage multiple
# segmented_img = np.zeros_like(img)

# segmented_img[img <= thresholds[0]] = 0  # Première classe (0)
# segmented_img[(img > thresholds[0]) & (img <= thresholds[1])] = 127  # Deuxième classe (gris)
# segmented_img[img > thresholds[1]] = 255  # Troisième classe (blanc)

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].imshow(segmented_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Seuillage multiple : seuils {thresholds}', fontsize=15)

# plt.show()

################################################################################################# Seuillage avec entropie (Shannon) #################################################################################################
# # Calcul de l'histogramme normalisé    
# histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))  # Calcul de l'histogramme
# histogram_norm = histogram / np.sum(histogram)  # Normalisation de l'histogramme

# # Affichage de l'histogramme
# plt.figure(figsize=(10, 5))
# plt.plot(bin_edges[:-1], histogram, color='black')
# plt.title("Histogramme des niveaux de gris")
# plt.xlabel("Niveau de gris")
# plt.ylabel("Densité de probabilité")
# plt.show()


# # Calcul de l'entropie pour chaque seuil    
# entropies = []
# for threshold in range(256):
#     p1 = np.sum(histogram_norm[:threshold])
#     p2 = np.sum(histogram_norm[threshold:])
#     if p1 == 0 or p2 == 0:
#         entropies.append(0)  # Évite la division par zéro
#         continue
#     entropies.append(-p1 * np.log2(p1) - p2 * np.log2(p2))

# # Seuillage d'Otsu
# threshold = np.argmax(entropies)

# # Application du seuillage    
# binary_img = (img > threshold).astype(np.uint8) * 255  # Seuillage binaire

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].imshow(binary_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Seuillage d\'Otsu (S={threshold})', fontsize=15)

# plt.show()

################################################################################################# Seuillage d'Otsu (Binaire) #################################################################################################

# # Étape 1 : Calcul de l'histogramme normalisé
# histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))  
# histogram_norm = histogram / np.sum(histogram)  # Normalisation de l'histogramme

# # Affichage de l'histogramme
# plt.figure(figsize=(10, 5))
# plt.plot(bin_edges[:-1], histogram, color='black')
# plt.title("Histogramme des niveaux de gris")
# plt.xlabel("Niveau de gris")
# plt.ylabel("Densité de probabilité")
# plt.show()

# # Initialisation des variables pour Otsu
# total_mean = np.sum(np.arange(256) * histogram_norm)  # Moyenne globale
# w1, mean1 = 0, 0  # Poids et moyenne pour la classe C1
# sigma_b_squared = np.zeros(256)  # Tableau pour stocker la variance inter-classe

# # Calcul de la variance inter-classe pour chaque seuil
# for threshold in range(256):
#     w1 += histogram_norm[threshold]  # Mise à jour de w1(T)
#     if w1 == 0:
#         continue

#     w2 = 1 - w1  # Probabilité de la classe 2
#     mean1 += threshold * histogram_norm[threshold]  # Mise à jour de mu1(T)
#     mean2 = (total_mean - mean1) / w2 if w2 > 0 else 0  # Moyenne de la classe C2
    
#     # Calcul de la variance inter-classe
#     sigma_b_squared[threshold] = w1 * w2 * (mean1 / w1 - mean2) ** 2

# # Seuil optimal
# optimal_threshold = np.argmax(sigma_b_squared)

# # Application du seuillage
# binary_img = (img > optimal_threshold).astype(np.uint8) * 255  # Seuillage binaire

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image originale', fontsize=15)

# ax[1].imshow(binary_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Seuillage d\'Otsu (S={optimal_threshold})', fontsize=15)

# plt.show()

################################################################################################# Seuillage d'Otsu avec OpenCV #################################################################################################
# # Seuillage d'Otsu avec OpenCV
# _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Affichage avec OpenCV
# cv2.imshow('Image originale', img)
# cv2.imshow('Seuillage d\'Otsu avec OpenCV', binary_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################################################################################# Seuillage multiple Otsu #################################################################################################

# def otsu_multi_thresholding(img, num_classes=3):
#     """
#     Algorithme de Otsu généralisé pour trouver plusieurs seuils optimaux.
#     :param img: Image en niveaux de gris sous forme de numpy array.
#     :param num_classes: Nombre de classes souhaitées (k).
#     :return: Liste des seuils optimaux.
#     """
#     assert num_classes >= 2, "Il faut au moins 2 classes (1 seuil)."

#     # Étape 1 : Calcul de l'histogramme normalisé
#     histogram, _ = np.histogram(img, bins=256, range=(0, 255))
#     histogram_norm = histogram / np.sum(histogram)  # Normalisation

#     # Étape 2 : Moyenne totale
#     total_mean = np.sum(np.arange(256) * histogram_norm)  # Moyenne globale

#     # Étape 3 : Génération de toutes les combinaisons possibles de seuils
#     seuils_possibles = range(1, 255)  # On évite 0 et 255
#     seuils_combinations = list(combinations(seuils_possibles, num_classes - 1))

#     # Étape 4 : Recherche des meilleurs seuils
#     best_thresholds = None
#     max_sigma_b_squared = -1

#     for seuils in seuils_combinations:
#         seuils = (0,) + seuils + (255,)  # Ajout des extrémités (0 et 255)
#         w, mu = [], []

#         for i in range(num_classes):
#             hist_range = range(seuils[i], seuils[i + 1])
#             w_i = np.sum(histogram_norm[hist_range])
#             mu_i = np.sum(np.arange(seuils[i], seuils[i + 1]) * histogram_norm[hist_range]) / w_i if w_i > 0 else 0
#             w.append(w_i)
#             mu.append(mu_i)

#         # Calcul de la variance inter-classe
#         sigma_b_squared = sum(w[i] * (mu[i] - total_mean) ** 2 for i in range(num_classes))

#         # Mise à jour des meilleurs seuils
#         if sigma_b_squared > max_sigma_b_squared:
#             max_sigma_b_squared = sigma_b_squared
#             best_thresholds = seuils[1:-1]  # On enlève les bornes 0 et 255

#     return best_thresholds

# # Détection automatique des seuils multiples
# num_classes = 3  # Ex: 3 classes -> 2 seuils
# seuils = otsu_multi_thresholding(img, num_classes)

# # Application du seuillage multiple
# segmented_img = np.zeros_like(img)
# for i, s in enumerate(seuils):
#     segmented_img[img > s] = (i + 1) * 255 // (num_classes - 1)

# # Affichage des résultats
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title("Image originale", fontsize=15)

# ax[1].plot(np.histogram(img, bins=256, range=(0, 255))[1][:-1], np.histogram(img, bins=256, range=(0, 255))[0], color='black')
# for s in seuils:
#     ax[1].axvline(s, color='red', linestyle='dashed', label=f"Seuil: {s}")
# ax[1].legend()
# ax[1].set_title("Histogramme et Seuils", fontsize=15)

# ax[2].imshow(segmented_img, cmap='gray')
# ax[2].axis('off')
# ax[2].set_title(f"Seuillage Otsu Multiple ({num_classes} classes)", fontsize=15)

# plt.show()

################################################################################################# kmeans #################################################################################################

# # Conversion de l'image en un tableau 1D pour le clustering
# pixels = img.reshape(-1, 1)  # Image en une colonne (1 seule dimension : intensité)

# # Définition du nombre de clusters (classes)
# k = 3  # Par exemple, 3 classes (fond, objet 1, objet 2)

# # Appliquer K-Means
# kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# kmeans.fit(pixels)

# # Récupérer les étiquettes (classe de chaque pixel) et recréer l'image segmentée
# segmented_img = kmeans.labels_.reshape(img.shape)

# # Affichage des résultats
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title("Image originale", fontsize=15)

# ax[1].imshow(segmented_img, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f"Segmentation K-Means ({k} classes)", fontsize=15)

# plt.show()

# # Conversion de l'image couleur en un tableau de pixels (chaque pixel = [R, G, B])
# pixels = img_color.reshape(-1, 3)  # Chaque pixel devient un vecteur (R, G, B)

# # Nombre de clusters (segments de couleur)
# k = 4  # Par exemple, 4 classes

# # Appliquer K-Means sur les couleurs
# kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# kmeans.fit(pixels)

# # Récupérer les étiquettes (classe de chaque pixel)
# segmented_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
# segmented_img = segmented_pixels.reshape(img_color.shape)  # Reconstruction de l'image

# # Affichage des résultats
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img_color)
# ax[0].axis('off')
# ax[0].set_title("Image originale", fontsize=15)

# ax[1].imshow(segmented_img)
# ax[1].axis('off')
# ax[1].set_title(f"Segmentation K-Means ({k} classes)", fontsize=15)

# plt.show()


################################################################################################ Choix du nombre de clusters ################################################################################################

# # Conversion de l'image en une liste de pixels (grayscale ou couleur)
# pixels = img.reshape(-1, 1)  # Pour une image en niveaux de gris
# # pixels = img_color.reshape(-1, 3)  # Pour une image couleur (décommenter)

# # Tester plusieurs valeurs de k
# inerties = []
# k_values = range(1, 10)  # Tester k de 1 à 10

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(pixels)
#     inerties.append(kmeans.inertia_)  # Stocker l’inertie

# # Tracer la courbe du coude
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inerties, marker='o', linestyle='dashed', color='blue')
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Inertie intra-classe")
# plt.title("Méthode du Coude pour déterminer k")
# plt.show()

# from sklearn.metrics import silhouette_score

# silhouette_scores = []
# k_values = range(2, 10)  # k = 1 est exclu car silhouette_score n'est pas défini pour k=1

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(pixels)
#     score = silhouette_score(pixels, labels)  # Calcul de la silhouette
#     silhouette_scores.append(score)

# # Tracer la silhouette en fonction de k
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, silhouette_scores, marker='o', linestyle='dashed', color='green')
# plt.xlabel("Nombre de clusters (k)")
# plt.ylabel("Indice de Silhouette")
# plt.title("Indice de Silhouette pour déterminer k")
# plt.show()

################################################################################################# Croissance de région #################################################################################################
# def region_growing(img, seed, threshold=10):
#     """
#     Implémente la croissance de région avec un point germe.
#     :param img: Image en niveaux de gris sous forme de numpy array.
#     :param seed: Coordonnées (x, y) du point de départ.
#     :param threshold: Seuil d'homogénéité pour inclure les pixels voisins.
#     :return: Image segmentée.
#     """
#     seed_value = img[seed]
#     mask = np.abs(img - seed_value) < threshold  # Pixels proches de la valeur du point germe
#     segmented_img = np.where(mask, 255, 0).astype(np.uint8)
    
#     return segmented_img

# def generate_colors(num_colors, is_color):
#     """
#     Génère une liste de couleurs distinctes adaptées au type d'image.
#     :param num_colors: Nombre de couleurs à générer.
#     :param is_color: Booléen indiquant si l'image est en couleur ou non.
#     :return: Liste de couleurs (valeurs uniques pour gris, tuples RGB pour couleur).
#     """
#     random.seed(42)  # Assure une reproductibilité des couleurs
#     if is_color:
#         return [tuple(random.randint(50, 255) for _ in range(3)) for _ in range(num_colors)]
#     else:
#         return [random.randint(50, 255) for _ in range(num_colors)]  # Valeurs uniques en niveaux de gris

# def region_growing(image, seeds, threshold=15, distance_threshold=5, distance_tolerance=5):
#     """
#     Algorithme de croissance de région avec file de priorité pour une propagation optimisée.
    
#     :param image: Image (niveaux de gris ou couleur) sous forme de np.array.
#     :param seeds: Liste de tuples [(y1, x1), (y2, x2), ...] définissant les points germes.
#     :param threshold: Seuil d'homogénéité pour accepter un pixel dans la région.
#     :param distance_threshold: Distance maximale pour qu'un pixel soit directement assigné à un germe.
#     :param distance_tolerance: Seuil pour limiter le recalcul des distances.
#     :return: Image segmentée avec propagation colorée.
#     """
#     # Dimensions et initialisation des structures de données
#     h, w = image.shape[:2]
#     is_color = len(image.shape) == 3
#     segmented = np.zeros((h, w, 3), dtype=np.uint8) if is_color else np.zeros((h, w), dtype=np.uint8)
#     distance_map = np.full((h, w), np.inf)  # Stockage des distances minimales
#     priority_queue = []  # Utilisation d'une file de priorité

#     # Générer des couleurs adaptées au type d'image
#     num_seeds = len(seeds)
#     colors = generate_colors(num_seeds, is_color)

#     # Initialisation des points germes dans la file de priorité
#     for i, seed in enumerate(seeds):
#         heapq.heappush(priority_queue, (0, seed, colors[i]))  # Distance initiale = 0
#         distance_map[seed] = 0  # Distance minimale = 0

#     # Définition des voisins pour la propagation (8-connectivité)
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

#     # Exploration des pixels voisins
#     while priority_queue:
#         current_dist, (y, x), color = heapq.heappop(priority_queue)  # Extraire le pixel le plus proche

#         # Ajouter le pixel à la segmentation
#         segmented[y, x] = color if is_color else int(color)

#         # Explorer les voisins
#         for dy, dx in neighbors:
#             ny, nx = y + dy, x + dx
#             if 0 <= ny < h and 0 <= nx < w:  # Vérifier les limites
#                 best_seed = None
#                 min_spatial_dist = np.inf
#                 min_intensity_diff = np.inf

#                 # Vérifier chaque germe
#                 for j, seed in enumerate(seeds):
#                     spatial_dist = np.sqrt((ny - seed[0]) ** 2 + (nx - seed[1]) ** 2)

#                     if is_color:
#                         intensity_diff = np.linalg.norm(image[ny, nx] - image[seed])
#                     else:
#                         intensity_diff = abs(float(image[ny, nx]) - float(image[seed]))

#                     # Vérifier si le pixel peut être immédiatement assigné
#                     if intensity_diff < threshold and spatial_dist <= distance_threshold:
#                         best_seed = j
#                         break  # Plus besoin de continuer, on a trouvé un bon candidat immédiatement

#                     # Sinon, on garde en mémoire le meilleur candidat trouvé jusqu'à présent
#                     if intensity_diff < threshold and spatial_dist < min_spatial_dist:
#                         min_spatial_dist = spatial_dist
#                         min_intensity_diff = intensity_diff
#                         best_seed = j

#                 # Si on a trouvé un germe valable, on l'affecte
#                 if best_seed is not None and (min_spatial_dist < distance_map[ny, nx] - distance_tolerance):
#                     distance_map[ny, nx] = min_spatial_dist  # Mise à jour de la distance minimale
#                     heapq.heappush(priority_queue, (min_spatial_dist, (ny, nx), colors[best_seed]))  # Ajout à la file

#     return segmented

# # # Exemple d'utilisation de la croissance de région

# # Liste de points germes
# seeds = [(50, 50), (200, 50), (125, 125), (200, 200)]

# # Application sur une image en niveaux de gris ou couleur
# segmented_img = region_growing(img_color, seeds, threshold=150, distance_threshold=1, distance_tolerance=0)

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(img_color)
# ax[0].axis('off')
# ax[0].set_title("Image originale", fontsize=15)

# ax[1].imshow(segmented_img)
# ax[1].axis('off')
# ax[1].set_title("Croissance de Région Optimale", fontsize=15)

# plt.show()

# ################################################################################################# Division et fusion de régions #################################################################################################
# def is_homogeneous(region, threshold):
#     """
#     Vérifie si une région est homogène selon un critère donné.
    
#     :param region: Région de l'image sous forme de matrice numpy.
#     :param threshold: Seuil d'homogénéité.
#     :return: True si la région est homogène, False sinon.
#     """
#     return np.std(region) < threshold  # Critère basé sur l'écart-type

# def compute_gradient(image):
#     """
#     Calcule le gradient de l'image pour détecter les contours forts.
#     :param image: Image d'entrée en niveaux de gris.
#     :return: Image du gradient.
#     """
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     return np.sqrt(grad_x ** 2 + grad_y ** 2)

# def split(image, x, y, size, min_size, threshold):
#     """
#     Divise l'image récursivement en sous-blocs homogènes.
    
#     :param image: Image d'entrée.
#     :param x, y: Coordonnées du coin supérieur gauche de la région.
#     :param size: Taille de la région actuelle.
#     :param min_size: Taille minimale d'une région.
#     :param threshold: Seuil d'homogénéité.
#     :return: Liste des régions détectées.
#     """
#     regions = []

#     # Extraire la région courante
#     region = image[y:y+size, x:x+size]

#     # Vérifier si la région est homogène ou si elle est trop petite pour être divisée
#     if is_homogeneous(region, threshold) or size <= min_size:
#         regions.append((x, y, size))  # Ajouter la région finale
#     else:
#         # Division en 4 sous-régions
#         half_size = size // 2
#         regions += split(image, x, y, half_size, min_size, threshold)  # Haut gauche
#         regions += split(image, x + half_size, y, half_size, min_size, threshold)  # Haut droit
#         regions += split(image, x, y + half_size, half_size, min_size, threshold)  # Bas gauche
#         regions += split(image, x + half_size, y + half_size, half_size, min_size, threshold)  # Bas droit

#     return regions

# def color_distance(region1, region2):
#     """
#     Calcule la distance Euclidienne entre deux couleurs moyennes (RGB).
#     :param region1: Moyenne de la première région (tuple R, G, B).
#     :param region2: Moyenne de la seconde région (tuple R, G, B).
#     :return: Distance entre les deux couleurs.
#     """
#     return np.linalg.norm(np.array(region1) - np.array(region2))

# def merge(regions, image, threshold, gradient_threshold=50):
#     """
#     Fusionne les régions adjacentes similaires en optimisant les recalculs et la gestion des contours.
    
#     :param regions: Liste des régions détectées.
#     :param image: Image d'entrée (peut être en niveaux de gris ou en couleur).
#     :param threshold: Seuil de fusion (distance maximale entre deux couleurs pour fusionner).
#     :param gradient_threshold: Seuil du gradient pour éviter de fusionner à travers des contours forts.
#     :return: Image segmentée.
#     """
#     h, w = image.shape[:2]
#     is_color = len(image.shape) == 3  # Vérifier si l'image est en couleur
#     segmented = np.zeros((h, w, 3), dtype=np.uint8) if is_color else np.zeros((h, w), dtype=np.uint8)
#     labels = np.zeros((h, w), dtype=int)  # Carte des labels des régions

#     # Calcul du gradient pour éviter la fusion à travers des contours
#     gradient_map = compute_gradient(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_color else image)

#     # Stocker les couleurs moyennes des régions
#     region_colors = {}

#     # Initialiser un graphe d'adjacence pour stocker les connexions des régions
#     adjacency_graph = defaultdict(set)

#     label = 1  # Compteur de labels uniques

#     # Trier les régions pour fusionner des blocs de grande taille en premier
#     regions.sort(key=lambda r: r[2], reverse=True)

#     # Affectation initiale des régions
#     for x, y, size in regions:
#         if np.all(labels[y:y+size, x:x+size] == 0):  # Vérifier si la région est encore non attribuée
#             mean_color = np.mean(image[y:y+size, x:x+size], axis=(0, 1)) if is_color else np.mean(image[y:y+size, x:x+size])
#             segmented[y:y+size, x:x+size] = mean_color
#             labels[y:y+size, x:x+size] = label  # Marquer la région fusionnée
#             region_colors[label] = mean_color  # Stocker la couleur moyenne
#             label += 1

#     # Construire le graphe d’adjacence
#     for x, y, size in regions:
#         region_label = labels[y, x]
#         if region_label == 0:
#             continue

#         # Vérifier les voisins immédiats
#         for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             ny, nx = y + dy, x + dx
#             if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != 0 and labels[ny, nx] != region_label:
#                 adjacency_graph[region_label].add(labels[ny, nx])

#     # Fusionner les régions adjacentes
#     for region_label, neighbors in adjacency_graph.items():
#         for neighbor_label in neighbors:
#             if neighbor_label in region_colors and region_label in region_colors:
#                 # Vérifier la différence de couleur
#                 diff = color_distance(region_colors[region_label], region_colors[neighbor_label]) if is_color else abs(region_colors[region_label] - region_colors[neighbor_label])

#                 # Vérifier le gradient pour éviter de fusionner à travers des contours forts
#                 if diff < threshold and np.mean(gradient_map[labels == neighbor_label]) < gradient_threshold:
#                     # Fusionner la région avec son voisin
#                     labels[labels == neighbor_label] = region_label
#                     new_color = np.mean(image[labels == region_label], axis=0) if is_color else np.mean(image[labels == region_label])
#                     segmented[labels == region_label] = new_color
#                     region_colors[region_label] = new_color  # Mettre à jour la couleur fusionnée

#     return segmented


# def split_and_merge(image, min_size=8, threshold_split=15, threshold_merge=10, gradient_threshold=50):
#     """
#     Implémente l'algorithme de segmentation Split-and-Merge.
    
#     :param image: Image d'entrée.
#     :param min_size: Taille minimale des blocs.
#     :param threshold_split: Seuil d'homogénéité pour la division.
#     :param threshold_merge: Seuil de similarité pour la fusion.
#     :return: Image segmentée.
#     """
#     h, w = image.shape[:2]
#     regions = split(image, 0, 0, min(h, w), min_size, threshold_split)
#     return merge(regions, image, threshold_merge, gradient_threshold)

# # Application de l'algorithme 
# segmented_img = split_and_merge(img_color, min_size=8, threshold_split=20, threshold_merge=15)

# # Affichage
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(img_color)
# ax[0].axis('off')
# ax[0].set_title("Image originale", fontsize=15)

# ax[1].imshow(segmented_img)
# ax[1].axis('off')
# ax[1].set_title("Segmentation Split-and-Merge (Optimisée)", fontsize=15)

# plt.show()

############################################################################################################### Watershed ##############################################################

def preprocess_image(image):
    """
    Prépare l'image pour la segmentation Watershed.
    
    :param image: Image d'entrée (RGB ou niveaux de gris).
    :return: Image prétraitée (gradient de l'image).
    """
    # Vérifier si l’image est en couleur
    is_color = len(image.shape) == 3

    # Convertir en niveaux de gris si nécessaire
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_color else image

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calcul du gradient de l'image (carte d'élévation)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient, blurred

# Prétraitement
gradient_img, blurred_img = preprocess_image(img)

# Affichage des résultats
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap="jet")
ax[0].axis('off')
ax[0].set_title("Image originale", fontsize=15)

ax[1].imshow(blurred_img, cmap='gray')
ax[1].axis('off')
ax[1].set_title("Image après flou gaussien", fontsize=15)

ax[2].imshow(gradient_img, cmap='gray')
ax[2].axis('off')
ax[2].set_title("Gradient de l'image (Carte d'élévation)", fontsize=15)

plt.show()

def detect_basins(gradient):
    """
    Détecte les bassins versants et initialise les marqueurs.
    
    :param gradient: Carte d’élévation (gradient de l'image).
    :return: Matrice des marqueurs.
    """
    h, w = gradient.shape
    markers = np.zeros((h, w), dtype=int)

    # Détection des minima locaux (les bassins)
    minima = (gradient < np.percentile(gradient, 35))  # Seuil bas (5% des pixels les plus bas)
    
    label = 1
    for y in range(h):
        for x in range(w):
            if minima[y, x]:  # Détecter les bassins
                markers[y, x] = label
                label += 1

    return markers

# def detect_basins(gradient):
#     """
#     Détecte les bassins versants et leur attribue des labels connexes.
    
#     :param gradient: Carte d’élévation (gradient de l'image).
#     :return: Matrice des marqueurs avec des labels regroupés.
#     """
#     h, w = gradient.shape
#     markers = np.zeros((h, w), dtype=int)

#     # Détection des minima locaux (les bassins)
#     minima = (gradient < np.percentile(gradient, 5)).astype(np.uint8)  # Seuillage bas (5% des pixels les plus bas)

#     # Appliquer `connectedComponents()` pour regrouper les bassins
#     num_labels, labels = cv2.connectedComponents(minima)

#     return labels


def watershed_propagation(gradient, markers):
    """
    Implémente l'algorithme Watershed en propageant les marqueurs par inondation.
    
    :param gradient: Carte d’élévation (gradient de l'image).
    :param markers: Matrice des marqueurs des bassins.
    :return: Image segmentée avec lignes de partage des eaux.
    """
    h, w = gradient.shape
    segmented = np.copy(markers)
    priority_queue = []  # File de priorité
    
    # Initialiser la file de priorité avec les bassins marqués
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:  # Si c'est un bassin identifié
                heapq.heappush(priority_queue, (gradient[y, x], (y, x), markers[y, x]))

    # Définition des voisins (8-connectivité)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Propagation par inondation
    while priority_queue:
        _, (y, x), label = heapq.heappop(priority_queue)

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and segmented[ny, nx] == 0:
                segmented[ny, nx] = label  # Affecter le marqueur
                heapq.heappush(priority_queue, (gradient[ny, nx], (ny, nx), label))

    return segmented

def extract_watershed_lines(segmented):
    """
    Extrait les lignes de partage des eaux.
    
    :param segmented: Image segmentée après propagation.
    :return: Image avec les lignes de partage des eaux marquées.
    """
    h, w = segmented.shape
    watershed_lines = np.zeros((h, w), dtype=np.uint8)

    # Définition des voisins (8-connectivité)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            label = segmented[y, x]
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and segmented[ny, nx] != label:
                    watershed_lines[y, x] = 255  # Marquer la ligne

    return watershed_lines

# Étape 1 : Prétraitement
gradient_img, blurred_img = preprocess_image(img)

# Étape 2 : Détection des bassins
markers = detect_basins(gradient_img)

# Étape 3 : Propagation des marqueurs (Algorithme Watershed)
segmented_img = watershed_propagation(gradient_img, markers)

# Étape 4 : Extraction des lignes de partage des eaux
watershed_lines = extract_watershed_lines(segmented_img)

# Affichage des résultats
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(markers, cmap='nipy_spectral')
ax[0].axis('off')
ax[0].set_title("Marqueurs Initiaux")

ax[1].imshow(segmented_img, cmap='nipy_spectral')
ax[1].axis('off')
ax[1].set_title("Segmentation Watershed")

ax[2].imshow(watershed_lines, cmap='gray')
ax[2].axis('off')
ax[2].set_title("Lignes de Partage des Eaux")

plt.show()
