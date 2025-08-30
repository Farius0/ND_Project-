import warnings; warnings.filterwarnings('ignore')
from skimage.morphology import disk, square, octagon, rectangle, diamond, erosion, dilation,\
    opening, closing, remove_small_objects, label, binary_closing, remove_small_holes, skeletonize, convex_hull_image, binary_erosion
from skimage import measure
from skimage.data import binary_blobs
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops

# print(disk(3))       # cercle de rayon 3
# print(square(5))      # carré 5x5
# print(octagon(3,1))   # octogone
# print(rectangle(3,5)) # rectangle 3x5
# print(diamond(3))    # diamant

image = np.zeros((100, 100), dtype=bool)
# image[30:70, 30:70] = 1              # un carré plein
image[20:40, 20:90] = 1    
# image[35, 35] = 0                    # trou à l'intérieur
# image[72, 72] = 1                    # bruit isolé
# image[50, 50] = 0                    # un trou au centre

# Structurant
se = disk(3)
# print(se)

# # Erosion
# eroded = erosion(image, se)

# # Affichage côte à côte
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Image originale")
# axes[1].imshow(eroded, cmap='gray')
# axes[1].set_title("Après érosion (disk(3))")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# # Dilatation avec le même structurant
# dilated = dilation(image, se)

# # Affichage côte à côte
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Image originale")
# axes[1].imshow(dilated, cmap='gray')
# axes[1].set_title("Après dilatation (disk(3))")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# # Ouverture (érosion suivie de dilatation)
# opened = opening(image, se)

# # Affichage côte à côte
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Image originale")
# axes[1].imshow(opened, cmap='gray')
# axes[1].set_title("Après ouverture (erosion → dilation)")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# # Fermeture (dilatation suivie d'érosion)
# closed = closing(image, se)

# # Affichage côte à côte
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Image originale")
# axes[1].imshow(closed, cmap='gray')
# axes[1].set_title("Après fermeture (dilation → érosion)")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# # Extraction des contours sur l'image originale
# contours = measure.find_contours(image, level=0.5)

# # Visualisation
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(image, cmap='gray')
# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
# ax.set_title("Contours extraits (find_contours)")
# ax.axis('off')
# plt.tight_layout()
# plt.show()


# # Générer les 16 cas possibles de 2x2 pixels (0 ou 1)
# from itertools import product

# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# level = 0.5

# for idx, config in enumerate(product([0, 1], repeat=4)):
#     row, col = divmod(idx, 4)
#     ax = axes[row, col]

#     # Construire la cellule 2x2
#     cell = np.array([
#         [config[0], config[1]],
#         [config[2], config[3]]
#     ], dtype=float)

#     # Créer une version agrandie pour affichage
#     big_cell = np.zeros((10, 10))
#     big_cell[2:4, 2:4] = cell

#     # Affichage
#     ax.imshow(big_cell, cmap='gray', interpolation='nearest')

#     # Affiche les valeurs
#     for i in range(2):
#         for j in range(2):
#             val = cell[i, j]
#             ax.text(2 + j, 2 + i, f"{int(val)}", color='red', ha='center', va='center', fontsize=12)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f"{''.join(map(str, config))}")

# plt.suptitle("Les 16 configurations possibles d'une cellule 2x2 (Marching Squares)", fontsize=16)
# plt.tight_layout()
# plt.show()


# # Cas concret : un carré avec trou au centre
# image_example = np.zeros((10, 10), dtype=float)
# image_example[3:7, 3:7] = 1.0
# image_example[4, 4] = 0.0  # petit trou

# # Extraire les contours
# contours_example = measure.find_contours(image_example, level=0.5)

# # Affichage
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(image_example, cmap='gray', interpolation='nearest')
# ax.set_title("Contour détecté via marching squares (niveau 0.5)")

# # Tracer les segments
# for contour in contours_example:
#     ax.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=2)

# ax.set_xticks(np.arange(0, 10))
# ax.set_yticks(np.arange(0, 10))
# ax.grid(True, color='lightgray')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()


# # Création d'une image binaire contenant plusieurs petits objets + 1 grand
# image_objects = np.zeros((100, 100), dtype=bool)
# image_objects[20:40, 20:40] = 1              # grand carré
# image_objects[10, 10] = 1                    # pixel isolé
# image_objects[80, 80] = 1                    # autre petit bruit
# image_objects[70:72, 30:32] = 1              # petit patch 2x2
# image_objects[50:52, 60:63] = 1              # patch 2x3

# # Étiquetage des objets
# labeled = label(image_objects)

# # Suppression des objets de moins de 10 pixels
# filtered = remove_small_objects(labeled, min_size=10)

# # Affichage
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(image_objects, cmap='gray')
# axes[0].set_title("Image avec bruit (petits objets)")
# axes[1].imshow(filtered > 0, cmap='gray')
# axes[1].set_title("Après remove_small_objects(min_size=10)")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# # Appliquer une dilatation sur l'image filtrée
# dilated_filtered = dilation(filtered > 0, se)

# # Affichage
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(filtered > 0, cmap='gray')
# axes[0].set_title("Image après suppression des petits objets")
# axes[1].imshow(dilated_filtered, cmap='gray')
# axes[1].set_title("Après dilation (disk(3))")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# # Application de la fermeture morphologique (dilatation → érosion)
# closed_image = binary_closing(dilated_filtered, footprint=se)

# # Affichage
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(dilated_filtered, cmap='gray')
# axes[0].set_title("Avant fermeture (dilatation seule)")
# axes[1].imshow(closed_image, cmap='gray')
# axes[1].set_title("Après binary_closing (disk(3))")
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# # On part d'une image binaire contenant des formes avec trous
# image_with_holes = np.zeros((100, 100), dtype=bool)
# image_with_holes[20:70, 20:70] = 1                     # carré principal
# image_with_holes[30:35, 30:35] = 0                     # trou 1
# image_with_holes[50:55, 50:55] = 0                     # trou 2
# image_with_holes[80, 80] = 1                           # bruit
# image_with_holes[85:87, 30:32] = 1                     # petit patch

# # On applique :
# # 1. remove_small_holes
# # 2. skeletonize
# # 3. convex_hull_image

# filled = remove_small_holes(image_with_holes, area_threshold=30)
# skeleton = skeletonize(image_with_holes)
# convexed = convex_hull_image(image_with_holes)

# # Affichage
# fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# axes[0].imshow(image_with_holes, cmap='gray')
# axes[0].set_title("Image originale\n(trous + bruit)")

# axes[1].imshow(filled, cmap='gray')
# axes[1].set_title("remove_small_holes\n(30 pixels)")

# axes[2].imshow(skeleton, cmap='gray')
# axes[2].set_title("skeletonize")

# axes[3].imshow(convexed, cmap='gray')
# axes[3].set_title("convex_hull_image")

# for ax in axes:
#     ax.axis('off')

# plt.tight_layout()
# plt.show()



# # Générer une image binaire réaliste (amas de formes organiques)
# segmented_example = binary_blobs(length=128, blob_size_fraction=0.1, volume_fraction=0.4, rng=1)

# # Appliquer le squelette
# skeleton = skeletonize(segmented_example)

# # Affichage
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(segmented_example, cmap='gray')
# axes[0].set_title("Image segmentée (binary_blobs)")

# axes[1].imshow(skeleton, cmap='gray')
# axes[1].set_title("Squelette (skeletonize)")

# for ax in axes:
#     ax.axis('off')

# plt.tight_layout()
# plt.show()



# # Création d'une nouvelle image avec objets au bord
# image_with_border_objs = np.zeros((100, 100), dtype=bool)
# image_with_border_objs[10:30, 10:30] = 1        # objet 1 (interne)
# image_with_border_objs[0:20, 70:90] = 1         # objet 2 (touche bord)
# image_with_border_objs[80:100, 0:20] = 1        # objet 3 (touche bord)
# image_with_border_objs[45:65, 45:65] = 1        # objet 4 (trou au centre)
# image_with_border_objs[50, 50] = 0              # trou dans l'objet 4

# # Suppression des objets au bord
# no_border = clear_border(image_with_border_objs)

# # Remplissage des trous internes
# filled_holes = binary_fill_holes(image_with_border_objs)

# # Propriétés des régions
# labels = label(image_with_border_objs)
# props = regionprops(labels)

# # Affichage
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(image_with_border_objs, cmap='gray')
# axes[0].set_title("Image originale (bord + trou)")

# axes[1].imshow(no_border, cmap='gray')
# axes[1].set_title("clear_border")

# axes[2].imshow(filled_holes, cmap='gray')
# axes[2].set_title("binary_fill_holes")

# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show(), props

# # Affichage des propriétés des régions détectées
# props_data = []

# for p in props:
#     props_data.append({
#         "Label": p.label,
#         "Aire (pixels)": p.area,
#         "Centroid (y, x)": p.centroid,
#         "Périmètre": p.perimeter,
#         "Excentricité": p.eccentricity,
#         "Orientation (rad)": p.orientation
#     })

# import pandas as pd
# # import ace_tools as tools

# df_props = pd.DataFrame(props_data)
# # df_props = tools.add_percentages(df_props)
# df_props


# ========================================================================================================================================================================================================================================================================================================

# # Réextraction des contours depuis le masque nettoyé
# sorted_contours = sorted(contours, key=len, reverse=True)
# contour_top = sorted_contours[0]

# # Étape : conversion en dictionnaire colonne → lignes
# col_indices = contour_top[:, 1].astype(int)
# row_indices = contour_top[:, 0]

# # On regroupe les lignes par colonne
# contour_map = {}
# for x, y in zip(col_indices, row_indices):
#     if x not in contour_map:
#         contour_map[x] = []
#     contour_map[x].append(y)

# # Calcul de l'épaisseur locale à partir de cette version simplifiée
# thickness_fast_map = {col: max(rows) - min(rows) for col, rows in contour_map.items() if len(rows) >= 2}

# # Visualisation
# cols = list(thickness_fast_map.keys())
# vals = list(thickness_fast_map.values())
# mean_thickness_fast = np.mean(vals)

# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(cols, vals, label="Épaisseur rapide (pixels)", color='teal')
# ax.axhline(mean_thickness_fast, color='red', linestyle='--', label=f"Moyenne = {mean_thickness_fast:.2f}")
# ax.set_title("Épaisseur du SC à partir du contour (rapide via np.unique + regroupement)")
# ax.set_xlabel("Colonne (x)")
# ax.set_ylabel("Épaisseur (pixels)")
# ax.legend()
# plt.tight_layout()
# plt.show()

# ==========================================================================================================================================================================================================================================================================================================
def estimate_thickness_single_contour(
    contour,
    method="mean",
    min_points_per_column=2,
    min_distance=1.0,
    smooth=False,
    kernel_size=5
):
    """
    Estime une épaisseur locale à partir d'un seul contour, en regroupant les points par colonne.

    Paramètres
    ----------
    contour : np.ndarray
        Contour (N x 2), coordonnée [y, x].

    method : str, optional
        Méthode de réduction pour regrouper les lignes ('mean', 'min', 'max', 'median'), par défaut 'mean'.

    min_points_per_column : int, optional
        Nombre minimal de points requis dans une colonne, par défaut 2.

    min_distance : float, optional
        Épaisseur minimale acceptée pour garder une colonne, par défaut 1.0.

    smooth : bool, optional
        Applique un lissage 1D sur le profil d'épaisseur, par défaut False.

    kernel_size : int, optional
        Taille du noyau de lissage (impair), par défaut 5.

    Retourne
    -------
    thickness_map : dict
        Dictionnaire {colonne : épaisseur}

    mean_thickness : float
        Moyenne de l'épaisseur sur les colonnes valides
    """
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    reducers = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "median": np.median
    }
    reducer = reducers.get(method, np.mean)

    col_indices = contour[:, 1].astype(int)
    row_indices = contour[:, 0]

    contour_map = {}
    for x, y in zip(col_indices, row_indices):
        contour_map.setdefault(x, []).append(y)

    thickness_map = {}
    for col, rows in contour_map.items():
        if len(rows) >= min_points_per_column:
            d = max(rows) - min(rows)
            if d >= min_distance:
                thickness_map[col] = d

    if smooth and len(thickness_map) >= kernel_size:
        keys = list(thickness_map.keys())
        values = list(thickness_map.values())
        smooth_vals = uniform_filter1d(values, size=kernel_size, mode='nearest')
        thickness_map = dict(zip(keys, smooth_vals))

    mean_thickness = reducer(list(thickness_map.values())) if thickness_map else float('nan')
    return thickness_map, mean_thickness



contours = measure.find_contours(image, 0.5)
sorted_contours = sorted(contours, key=len, reverse=True)

# Application sur la forme sinueuse
raw_profile, mean_raw = estimate_thickness_single_contour(sorted_contours[0], smooth=False)
smooth_profile, mean_smooth = estimate_thickness_single_contour(sorted_contours[0], smooth=True, kernel_size=9)

# Visualisation
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(raw_profile.keys(), raw_profile.values(), label="Sans lissage", color='grey', alpha=0.6)
ax.plot(smooth_profile.keys(), smooth_profile.values(), label="Avec lissage", color='green', linewidth=2)
ax.axhline(mean_raw, linestyle='--', color='red', label=f"Moyenne brute = {mean_raw:.2f}")
ax.axhline(mean_smooth, linestyle='--', color='blue', label=f"Moyenne lissée = {mean_smooth:.2f}")
ax.set_title("Profil d’épaisseur — Comparaison Sans vs Avec lissage")
ax.set_xlabel("Colonne (x)")
ax.set_ylabel("Épaisseur (pixels)")
ax.legend()
plt.tight_layout()
plt.show()

# # Visualisation superposée sur l’image du masque avec le profil d’épaisseur
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.imshow(image, cmap='gray')
# ax.set_title("Visualisation de l'épaisseur projetée sur le masque")

# # Affichage des barres d'épaisseur sur les colonnes valides
# for col, ep in smooth_profile.items():
#     y0 = min(sorted_contours[0][:, 0][sorted_contours[0][:, 1].astype(int) == col], default=0)
#     ax.plot([col, col], [y0, y0 + ep], color='lime', linewidth=1)

# ax.set_xlabel("Colonne (x)")
# ax.set_ylabel("Ligne (y)")
# plt.tight_layout()
# plt.show()


#====================================================================================================================================================================
from scipy.spatial.distance import cdist

for w in range(10, 17) :
    ma = contours[0]
    n = int(np.round(ma[:,1].min() + w))
    l = int(np.round(ma[:,1].max() - w))

    contours2 = measure.find_contours(image[:, n:l])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image, cmap='gray')
    ax.set_title("Visualisation de l'épaisseur projetée sur le masque")
    for c in contours2:
        plt.plot(c[:, 1] + n, c[:, 0], linewidth=5, color='green')
        
    plt.tight_layout()
    plt.show()
    
    if len(contours2) == 2 :
        break

c_dist = cdist(contours2[0], contours2[1])
c_dist_min = np.min(c_dist, axis=1)
c_dist_mean = np.mean(c_dist_min)
split_mean = list(filter(lambda x: x > 0, c_dist_mean))
split_mean = list(filter(lambda x: x < 30, split_mean))

# print(c_dist_mean)

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.imshow(image, cmap='gray')
# ax.set_title("Visualisation de l'épaisseur projetée sur le masque")
# plt.plot(ma[:,1], np.full_like(ma[:,1], ma[:,0].min()), linewidth=5, color='red')
# plt.plot(ma[:,1], np.full_like(ma[:,1], ma[:,0].max()), linewidth=5, color='red')
    
# plt.tight_layout()
# plt.show()

