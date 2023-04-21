
<h1>Prédire les ventes d'un jeu vidéo à l'aide du machine learning. En collaboration avec : C.Anselmo, K.Minatchy, D.Rivet</h1>

A partir du scrapping du site [vgchartz](https://www.vgchartz.com/gamedb/) nous avons analysé le marché du jeu vidéo et tenter de prédire les ventes à travers un modèle de regression.

Pour prédire les ventes nous avons tenté de collecter le plus de variables possible en scrappant également les sites [jeuxvideo.com](https://www.jeuxvideo.com/tous-les-jeux/) et [metacritic](https://www.metacritic.com/browse/games/score/metascore/all/all/filtered).
La spécificité de ce jeu de données et que celui-ci contient beaucoup d'outlier et n'a pas de temporalité de ventes (c'est à dire qu'elles sont toutes cumulées).

Nous avons donc effectué une analyse grâce aux variables explicatives suivantes (notre variable cible étant Golbal_Sales) : 
numériques : 
- l'année
- la note

catégorielles : 
- la plateforme
- le genre
- le studio (développeur), publisher (éditeurs) que nous avons tout deux clusterisé.
En l'état les performances de nos modèles n'étaient pas assez fiable pour le déployer.


<h2>A posteriori voici les difficultés rencontrées ainsi que les approches que nous aurions pu prendre :</h2>

1. Les difficultés principales identifiées sont: 
- le match de la collecte de données via scrapping
- pas de partitionnement de la temporalité des ventes 

2. Les approches que nous pourrions proposer pour pallier ces difficultés: 
- trouver un positionnement sur le marché : point de vue du studio, du publisher, 
- une période de temps, 
- prédire pour une saga de jeux (ex: assassins creed 2 par rapport aux résultats de ventes assassins creed 1)

3. Ce qu'il faut également prendre en compte pour ce type d'analyse de prédiction des ventes: 
- analyse de sentiment pour prédire l'engouement avant la sortie des jeux (commentaires sur le trailer, nb de vues avant sa sortie) 
- concurrence: cours de la bourse sur les acteurs du jeu vidéo
- économique: pouvoir d'achat sur le marché (ex: crise économique et sa variation des postes de dépense)
- socioculturel: étude sur les comportements des consommateurs (ex: genre attendu par âge, pays)
- technologique: innovations sur le marché (disruption du marché du jeu vidéo avec l'amplification des jeux en ligne, sur mobile mais encore les nouvelles technologie telles que les casque de RA, le metaverse...)
- saisonnalité: trend à certaines périodes de l'année (ex: un jeu qui sortira pour Noël aura forcement plus de succès que si il est lancé à une autre période de l'année).

Pour en savoir plus sur notre projet, voici le [rapport complet](https://www.canva.com/design/DAFdomYMGpE/fGqe4W_wGX5-QixbEYUfmw/edit?utm_content=DAFdomYMGpE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).
