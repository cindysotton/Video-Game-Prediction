import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from bokeh.plotting import figure, output_notebook, show, curdoc, save
from bokeh.models import BoxAnnotation, ColumnDataSource, Range1d
from bokeh.models.tools import HoverTool
from bokeh.layouts import row
from bokeh.models import Range1d, LabelSet, Tabs
from bokeh.palettes import inferno

from sklearn import model_selection
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



#[theme]
#base="light"
#primaryColor="#9400d3"

def set_theme():
    st.set_page_config(
        page_title="Game Project Prediction",
        page_icon="🎮",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Personnalisation du thème
    st.markdown(
        """
        <style>
        .css-1aumxhk {  /* Exemple de sélecteur CSS */
            background-color: #0E1117;
            primaryColor="#9400d3";
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# fichiers 
df = pd.read_csv('vgsales_all1.csv')
df_ml = pd.read_csv('df_ml.csv')

flierprops = dict(marker="X", markerfacecolor='darkviolet', markersize=12,
                  linestyle='none')

# 2. horizontal menu
selected = option_menu(None, ['Projet','Méthodologie','Analyse','Modélisation','Conclusion'],
    icons=['book',"eyedropper",'calculator','clipboard','controller'],
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
    styles={
            "icon": {"color": "#blank", "font-size": "13px"},
            "nav-link": {
                "font-size": "12px",
                "text-align": "center",
                }})

# Présentation
if selected == "Projet":
    image = Image.open('media/manette.jpeg')
    st.image(image)
    st.title("La Data Science sera t elle le \"cheat code\" de la vente d'un jeu vidéo ?")
    st.markdown("Estimer les ventes d'un produit avant son lancement peut être une véritable force pour la rentabilité d'une entreprise. Dans le cadre de ce projet nous allons essayer de déployer un modèle qui permettra de prédire les ventes d'un jeu.")
    st.markdown("""Ce projet a permis de tirer parti des connaissances apprises lors du cursus de formation Datascientest en gérant un projet complet pour répondre à une problématique donnée.
* Techniquement :  Les données étant incomplètes il faut avant tout venir compléter et actualiser le jeu de données pour affiner au mieux l'analyse et les tendances. 
* Economiquement: Le projet prend son intérêt dans l'estimation des ventes en quantité. Grâce a ce dernier,  en cas de réussite, nous pourrons expliquer et prévoir les ventes et spécifier a quoi le succès est dû (genre, notes, divers).  Pour les équipes en charge de la prise de décision du lancement d'un jeu vidéo, cette prédiction aura énormément de valeur, car elle permettra de savoir si il est pertinent ou pas de lancer sa production en fonction des objectifs de rentabilité fixés.""")
    
    st.subheader("Contacter l'équipe")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        linkedin_celine = "https://www.linkedin.com/in/celine-anselmo/"
        image_celine = "./app/static/Celine_Anselmo.png"
        
        st.markdown(f'<a href="{linkedin_celine}" target="_blank"><img src="{image_celine}" width="200"></a>', unsafe_allow_html=True)

    with col2:
        linkedin_cindy = "https://www.linkedin.com/in/cindysotton/"
        image_cindy = "./app/static/Cindy_Sotton.png"
        
        st.markdown(f'<a href="{linkedin_cindy}" target="_blank"><img src="{image_cindy}" width="200"></a>', unsafe_allow_html=True)

    
    with col3:
        linkedin_karine = "https://www.linkedin.com/in/karine-minatchy/"
        image_karine = "./app/static/Karine_Minatchy.png"
        
        st.markdown(f'<a href="{linkedin_karine}" target="_blank"><img src="{image_karine}" width="200"></a>', unsafe_allow_html=True)

    with col4:
        linkedin_dorian = "https://www.linkedin.com/in/dorian-rivet/"
        image_dorian = "./app/static/Dorian_Rivet.png"
        
        st.markdown(f'<a href="{linkedin_dorian}" target="_blank"><img src="{image_dorian}" width="200"></a>', unsafe_allow_html=True)

    st.subheader("Objectifs")
    st.markdown("""Nous devons,  à partir du dataset fourni et des variables collectées, prédire les ventes de la sortie d'un jeu vidéo. Pour ce faire nous avons identifié les principaux objectifs et étapes qui vont jalonner la réalisation du projet.
Les principales étapes identifiées dans le cadre du projet sont:  
* Extraction des données et nouvelles données
* Data processing
* Data profiling
* Modélisation des données
* Conclusion et préconisations
Afin de répondre, au mieux, à la problématique, nous avons pris contact avec un expert métier (ancien Développeur d’Ubisoft) qui aura pu nous partager ses connaissances sur le secteur, mais aussi son expertise en Data Science étant lui même Data Scientist.""")

    # texte violet
    st.markdown("""
    <style>
    .purple {
        color : darkviolet;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="purple">Entretien métier :</p>', unsafe_allow_html=True)
    st.markdown("""
Voici les points essentiels qui nous aurons aidé dans notre approche pour répondre à la problématique :

1. Les variables qui sont intéressantes à analyser pour la prédiction d’un jeu aurait pu être les suivantes :
* les tendances marché
* le cours de la bourse
* le mode de commercialisation (abonnement, pass...)
* la popularité en ligne (nombre de commentaires sur un trailer, commentaires Redit)

2. Avoir une approche avec de la temporalité est également indispensable.

En l’état notre dataset ne nous permettait pas d’ajouter une notion de temporalité, de part notre avancé et les difficultés que nous avons rencontré, nous n’avons pu tirer pleinement parti de ces recommandation mais néanmoins, elle nous ont donné des idées pour apporter des éléments de réponses parallèle à la problématique dans la conclusion.""")

# Méthodologie
if selected == "Méthodologie":
    image = Image.open('media/manette.jpeg')
    st.image(image)
    tab1, tab2, statistiques = st.tabs(["Extraction des données", "Data processing",'Statistiques'])
    # texte violet
    st.markdown("""
    <style>
    .purple {
        color : darkviolet;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with tab1:
        st.markdown('<p class="purple">Nos données - Dataset Vgchart:</p>', unsafe_allow_html=True)
        st.markdown("""

Le dataset Vgchart est le jeu initial fourni dans le cadre du projet disponible sur Kaggle. les variables se répartissent comme suit:
* Nom
* Date de sortie
* Genre
* Plateforme
* Publisher
* Ventes par région""")
        st.markdown('<p class="purple">Complétion des données - Utilisation du scrapping: </p>', unsafe_allow_html=True)
        st.markdown("""
Nous avons scrappé le site Vgchart pour récupérer : 
* des données plus récentes
* les variables Critic Score et Studio. 

Ce nouveau jeu annule et remplace le premier jeu de données de Vgchart.

Pour compléter les données à disposition nous avons scrappé le site Metacritic et le site jeuvidéo.com avec les librairies Sélenium et BeautifulSoup pour collecter les notes utilisateurs et presse sur une communauté orientée mondiale et française.

Le scrapping a permis de récupérer les informations suivantes:
* Description
* Studio
* Note utilisateur
* Nombre de note utilisateur
* Note presse
* Nombre de vote presse
* Genre

Les résultats n'étant pas à la hauteur autant pour Metacritic que pour jeuxvideo.com (% de Nan trop élevé) nous n'avons pas conservé les données. Pour jeuxvideo.com, les données avaient encore plus de Nan. En effet, les jeux scrappés avec un id français ne pouvaient communiquer avec le dataset initial.""")
            
        st.markdown ('Voici notre nouveau dataset:') 
        st.write(df.head())

    with tab2:
        # texte violet
        st.markdown("""
        <style>
        .purple {
            color : darkviolet;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="purple">1 - Nettoyage de données:  </p>', unsafe_allow_html=True)
        st.markdown("""
* formater le type de la variable date
* supprimer les données après 2022
* supprimer les données NaN pour Global_Sales (notre variable cible)
* supprimer les données NaN pour Critic_score (la variable qui a été l'objet du scrapping)
* remplacer les NaN par 0 pour les sales des régions 
* renommer les colonnes pour la compréhension
* formater des colonnes en str lorsque nécessaire""")

        st.markdown('<p class="purple">2 - Transformation des données: </p>', unsafe_allow_html=True)
        st.markdown("""
* remplacer dans un premier temps les NaN par 0 afin d’éviter la création d’array lors de notre transformation du dataset en table pivot
* supprimer les variables extrêmes pour l'année, le genre et la plateforme
* encoder la variable plateforme afin qu'elle soit en colonne (ex: colonne wii[0,1]) et non en ligne. Une fois le pivot réalisé, nous reconstituons la variable en précisant "Multi_plateforme" si le jeux est sur plusieurs plateformes.
* modifier le dataset en table pivot afin d’avoir une ligne par jeu
* organiser l'ordre des colonnes pour les avoir dans un ordre plus pertinent 

Le dataset obtenu est celui utilisé pour la visualisation et les statistiques""")
    
    with statistiques:
        #Analyses statistiques

        # texte violet
        st.markdown("""
        <style>
        .purple {
            color : darkviolet;
        }
        </style>
        """, unsafe_allow_html=True)

        #Global_Sales
        st.markdown('<p class="purple">Corrélation avec la variable cible: Global_Sales</p>', unsafe_allow_html=True)
        st.markdown("L'analyse de la variance ANOVA a été utilisée pour mettre en relation nos différentes variables explicatives: Platform, Genre, Studio, Publisher et notre variable cible Global_Sales.")

        #image stats1
        image_path = "media/stats1.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Anova")
        except:
            st.error("Impossible d'afficher l'image")

        st.markdown("L'objectif du test est de comparer les moyennes des deux échantillons et de conclure sur l'influence d'une variable explicative catégorielle sur la loi dune variable continue à expliquer. Lorsque la p-value (PR(>F)) est inférieur à 5%, on rejette l'hypothèse selon laquelle la variable n'influe pas sur Global_Sales")

        st.markdown("Nous notons que les autres variables explicatives influent sur la valeur cible. Nous procèderons donc à une analyse complémentaire pour identifier le poids des variables lors de la modélisation.")

        #variables explicatives
        st.markdown('<p class="purple">Corrélation entre les variables explicatives:</p>', unsafe_allow_html=True)

        st.markdown("Nous avons utilisé la méthode statistique V de Cramer pour mesurer le niveau de corrélation entre nos variables explicatives de type qualitatives")
        
        #image stats2
        image_path = "media/stats2.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="V de Cramer")
        except:
            st.error("Impossible d'afficher l'image")

        st.markdown("Cette analyse a permis de mettre en valeur des relations fortes comme le genre et le publisher et de mettre de côté des analyses comme platform et genre où il n'y a pas de corrélation.")


if selected == "Analyse":
    option = st.radio(
    "Menu:",
    ('Le marché','Plateformes', 'Genres', 'Studios','Publishers','Notes'))
    st.divider()
    
    if option == "Le marché":
        st.title('VgChartz : Analyse des données')
        
        st.subheader('Evolution des ventes en millions')
        # EVOLUTION DES VENTES
        data_NA = df.groupby(by=['Year'])['NA_Sales'].sum().reset_index()
        data_EU = df.groupby(by=['Year'])['EU_Sales'].sum().reset_index()
        data_JP = df.groupby(by=['Year'])['JP_Sales'].sum().reset_index()
        data_Others = df.groupby(by=['Year'])['Other_Sales'].sum().reset_index()
        data_globales = df.groupby(by=['Year']).sum().reset_index()
        fig = px.line(data_frame=data_globales, x='Year', y='Global_Sales', labels={'Year': 'Year', 'Global_Sales': 'Sales'})
        fig.add_scatter(x=data_NA['Year'], y=data_NA['NA_Sales'], mode='lines', name='NA_Sales', line_color='darkviolet')
        fig.add_scatter(x=data_EU['Year'], y=data_EU['EU_Sales'], mode='lines', name='EU_Sales', line_color='royalblue')
        fig.add_scatter(x=data_JP['Year'], y=data_JP['JP_Sales'], mode='lines', name='JP_Sales', line_color='hotpink')
        fig.add_scatter(x=data_Others['Year'], y=data_Others['Other_Sales'], mode='lines', name='Other_Sales', line_color='aqua')
        fig.add_scatter(x=data_globales['Year'], y=data_globales['Global_Sales'], mode='lines', name='Global_Sales', line_color='gray')
        fig.update_layout(
            xaxis_title='Années',
            yaxis_title='Ventes',
            legend_title='Zones géographiques',
            width=800,
            height=600,
            yaxis_range=[0, 600],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("> Survolez le graphique pour faire apparaître le détail.")
        st.markdown(
            """Le marché du jeu vidéo a commencé sa croissance à partir de la seconde moitié des années 90 dynamisé par le lancement de nouvelles plateformes:
* Sortie de la PlayStation en 1995
* Nouvel élan dans les années 2000 avec la sortie de la Nintendo 64

Après une forte croissance (2005 à 2010), le marché revient à sa tendance initiale. """)

        # PIE DE LA REPARTITION
        
        st.subheader("Répartions des ventes par zones géographiques")
        df_areas = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
        df_areas = df_areas.sum().reset_index()
        df_areas = df_areas.rename(columns={"index": "Areas", 0: "Sales"})
        labels = df_areas['Areas']
        sizes = df_areas['Sales']
        colors = ['darkviolet', 'royalblue', 'hotpink', 'aqua']
        fig = px.pie(df_areas,
                values=sizes,
                names=labels,
                color_discrete_sequence=colors
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("> Survolez le graphique pour faire apparaître le détail.")
        
        st.markdown(
            """Nos ventes se concentrent sur trois principaux marchés : North America, Europe, Japon (≈90%). 
Les ventes sur d'autres marchés sont inférieures à 10%. 

A noter la concentration particulière d'une part avec :
North Amercia qui réalise près de la moitié des ventes
Le Japon qui réalise plus de 10% des ventes à mettre en perspecitive avec le nombre d'habitants.""")
        
        st.subheader("Ventes par jeux vidéo")
        source = ColumnDataSource(df)
        hover = HoverTool(
        tooltips=[
            ("Nom", "@Name"),
            ("Genre", "@Genre"),
            ("Année", "@Year"),
            ("Platform", "@Platform"),
            ("Studio", "@Studio"),
            ("Publisher", "@Publisher"),
            ("Note", '@Critic_Score'),
            ("Ventes", '@Global_Sales') ])
        p98 = figure(plot_width=800, plot_height=700,x_axis_label='Années', y_axis_label='Ventes')
        doc = curdoc()
        doc.theme = 'dark_minimal'
        doc.add_root(p98)
        p98.circle(x='Year',y='Global_Sales',source = source,color='darkviolet',size=10)
        p98.add_tools(hover)
        st.bokeh_chart(p98, use_container_width=True)
        st.markdown("> Survolez les points du graphique pour faire apparaître le détail.")
        st.markdown("Certains jeux ont connu un succès exceptionnel, c'est notamment le cas pour Wii Sport sorti en 2006 chez Nintendo.")
        st.markdown("Ce graphique nous permet d'identifier les jeux qui se sont démarqués en terme de ventes et apprécier les variables en lien. On peut identifier des \"sagas\" qui ont bien marché (Mario, Pokemon, Grand Theft Auto).")

    if option == 'Plateformes':
            # Replace smaller values with 'Autre' in the 'Platform' column
            df['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne', 'XB', 'DC'], ['Autre', 'Autre', 'Autre', 'Autre', 'Autre'])

            option_plateforme = st.multiselect(
                '**Sélectionner les plateformes que vous souhaitez comparer :**',
                options=df['Platform'].unique(),
                default=df['Platform'].unique(),
                key='plateformes_options'
            )

            # Selected platforms
            if len(option_plateforme) == 0:
                st.warning('Merci de sélectionner au moins une plateforme')

            # Filter the data based on the selected platforms
            df_filtered = df[df['Platform'].isin(option_plateforme)]

            st.subheader('Répartition des ventes par plateformes')

            # Dictionnaire des couleurs par modalités pour retrouver les mêmes sur l'ensemble des graphiques
            DICT_PLAT = {'Multi_Plateforme': 'dodgerblue',
                        'PSP': 'tomato',
                        'GBA': 'mediumaquamarine',
                        'PC': 'mediumpurple',
                        'DS': 'sandybrown',
                        'PS3': 'lightskyblue',
                        'GC': 'hotpink',
                        'PS': 'palegreen',
                        'Wii': 'violet',
                        'PS2': 'gold',
                        'Autre': 'lavender',
                        'X360': 'salmon',
                        '3DS': 'aquamarine',
                        'NS': 'plum',
                        'N64': 'peachpuff'}

            fig = px.pie(df_filtered,
                        values='Global_Sales',
                        names='Platform',
                        color='Platform',
                        color_discrete_map=DICT_PLAT)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            Nous constatons que les parts de marché se répartissent de manière équilibré entre les plateformes.

A noter que certaines plateformes tendent à disparaitre car remplacer par leur upgrade (PS2 qui devient la PS3).""")

            
            st.subheader('Analyse des valeurs extrêmes par plateformes')
            fig = px.box(df_filtered,
             x='Platform',
             y='Global_Sales',
             color='Platform',
             color_discrete_map=DICT_PLAT,
             hover_data=['Name', 'Genre', 'Year', 'Studio', 'Publisher', 'Critic_Score', 'Global_Sales'])

            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                xaxis_tickangle=75,
                height=600,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
Cette représentation graphique met en évidence le constat effectué précédemment à savoir que la plateforme Wii à un outlier. Il s'agit de Wii Sport qui fait des records de ventes par rapport aux autres jeux de Wii. Nos recherches nous ont indiqué que ce jeu est sorti en 2006 en même temps que la console Wii ce qui a participé à l'engouement et l'explosion des ventes. Le jeu faisait parti d'une offre bundle avec la Wii. 

La DS a également des valeurs extrêmes qu'il sera intéressant de regarder avec New Super Mario.""")


            
            st.subheader("Analyse de la corrélation des plateformes par zones géographiques")
            comp_platform = df_filtered[['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
            comp_map = comp_platform.groupby(by=['Platform']).sum()

            plt.figure(figsize=(15, 10))
            sns.set(font_scale=1)
            ht = sns.heatmap(comp_map, annot=True, cmap="cool", fmt='.1f', cbar=False)
            fig2 = ht.get_figure()
            ax = ht.axes
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            st.pyplot(fig2, use_container_width=True)

            st.markdown("""Element intéressant : nous pouvons constater que pour le Japon les plateformes qui ressortent le plus sont "nomades" (3DS et DS)""")

    if option == 'Genres':
            # Remplacer les modalités peu nombreuse par Autre
            #df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
            
            option_genre = st.multiselect(
                '**Sélectionner les plateformes que vous souhaitez comparer :**',
                options=df['Genre'].unique(),
                default=df['Genre'].unique(),
                key='genre_options'
            )

            # Selected platforms
            if len(option_genre) == 0:
                st.warning('Merci de sélectionner au moins un genre')

            # Filter the data based on the selected platforms
            df_filtered = df[df['Genre'].isin(option_genre)]

            st.subheader('Evolution du nombre de jeux par genre')

            fig = px.line(df_filtered, x='Year', y='Genre', color='Genre',
                          title='Nombre de jeux par genre et par année',
                          labels={'Year': 'Année', 'Genre': 'Genre'})
            
            st.plotly_chart(fig, use_container_width=True)


            st.subheader('Répartition des ventes par genre')

            # création d'un dictionnaire pour avoir les même couleurs
            DICT_GENRE = {'Role-Playing': 'dodgerblue',
            'Action': 'tomato',
            'Shooter': 'mediumaquamarine',
            'Sports': 'mediumpurple',
            'Platform': 'sandybrown',
            'Racing': 'lightskyblue',
            'Adventure': 'hotpink',
            'Fighting': 'palegreen',
            'Misc': 'violet',
            'Strategy': 'gold',
            'Simulation': 'lavender',
            'Puzzle': 'salmon',
            'Autre': 'aquamarine'}

            fig = px.pie(df_filtered,
                        values='Global_Sales',
                        names='Genre',
                        color='Genre',
                        color_discrete_map=DICT_GENRE)

            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader('Analyse des valeurs extrêmes par genres')
            fig = px.box(df_filtered,
             x='Genre',
             y='Global_Sales',
             color='Genre',
             color_discrete_map=DICT_GENRE,
             hover_data=['Name', 'Genre', 'Year', 'Studio', 'Publisher', 'Critic_Score', 'Global_Sales'])

            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                xaxis_tickangle=75,
                height=600,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
On observe globalement que tous les genres ont plusieurs plusieurs valeurs extrêmes.

On note que les plus significatives sont: 
* Mario pour "Platform"
* Wii Sport pour "Sport"
* Grand Theft Auto pour "Action"
* Call of Duty pour "Shooter"
* Pokemon "Role-Playing"
* Mario Kart pour "Racing" """)


            
            st.subheader("Analyse de la corrélation des genres par zones géographiques")
            comp_genre = df_filtered[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
            comp_map = comp_genre.groupby(by=['Genre']).sum()

            plt.figure(figsize=(15, 10))
            sns.set(font_scale=1)
            ht = sns.heatmap(comp_map, annot=True, cmap="cool", fmt='.1f', cbar=False)
            fig2 = ht.get_figure()
            ax = ht.axes
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            st.pyplot(fig2, use_container_width=True)

            st.markdown("""
            Nous constatons qu'il y'a des préférences de genre en fonction des zones géographiques. Comme pour les plateformes le Japon a un genre qui se distingue plus que les autres.""")





# Modelisation
if selected == "Modélisation":
    tab1, tab2, tab3 = st.tabs(["Introduction", "Etapes de la Modélisation", 'Résultats'])
    with tab1:
        st.header('Dataset Modélisation:')
        st.subheader('Variable Cible')
        st.markdown('-  Global_Sales')
        st.subheader('Variables Explicatives')
        st.markdown("-  L'année de sortie (Year)  \n-  Le genre (Genre)  \n-  Le studio l’ayant développé (Studio)  \n-  L’éditeur l’ayant publié (Publisher)  \n-  La plateforme sur laquelle le jeu est sortie (Platform)  \n-  Les notes utilisateurs (Critic_Score)")  
        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire). ")
        
    with tab2:
        # texte violet
        st.markdown("""
        <style>
        .purple {
            color : darkviolet;
        }
        </style>
        """, unsafe_allow_html=True)
        
        #Préprocessing Etapes
        st.header('Pre-processing:')
        st.markdown('<p class="purple">Etape 1: Clustering des variables studio et publisher</p>', unsafe_allow_html=True)
        st.markdown('-  Suite au nombre important de modalités dans ces trois colonnes (+700 pour studio), nous avons simplifié les variables en utilisant la méthode du clustering. Studio, Publisher: 1 à 4 suivant leur montant de Global_Sales')
        st.markdown('<p class="purple">Etape 2: Suppression des variables non pertinentes pour la modélisation (name, region sales, description)</p>', unsafe_allow_html=True)
        st.markdown('<p class="purple">Etape 3: Encoding des variables catégorielles</p>', unsafe_allow_html=True)
        
        #Modélisation Etapes
        st.header('Modélisation:')
        st.markdown('<p class="purple">Etape 4: Analyse de l importance des variables, Itération 2</p>', unsafe_allow_html=True)
        
        st.header('Résultats:')
        # Graph importance 
        v1 = ([0.33011073, 0.21852045, 0.00776532, 0.062745  , 0.0032006 ,
        0.00506196, 0.00514901, 0.01860793, 0.00530342, 0.00532782,
        0.01765334, 0.00209605, 0.01434875, 0.01864197, 0.02456184,
        0.01448861, 0.02698887, 0.00310434, 0.00044673, 0.03785596,
        0.00462839, 0.00362745, 0.09522153, 0.00242704, 0.00349292,
        0.00649888, 0.00348894, 0.00684423, 0.0031338 , 0.00686099,
        0.00162643, 0.02738152, 0.0014133 , 0.01008641, 0.00066476,
        0.00062469])

        v2 = (['Critic_Score', 'Year', 'cat_publi', 'cat_studio',
        'Genre_Action-Adventure', 'Genre_Adventure', 'Genre_Fighting',
        'Genre_Misc', 'Genre_Music', 'Genre_Party', 'Genre_Platform',
        'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing', 'Genre_Shooter',
        'Genre_Simulation', 'Genre_Sports', 'Genre_Strategy', 'Platform_DC',
        'Platform_DS', 'Platform_GBA', 'Platform_GC',
        'Multi_Plateforme', 'Platform_N64', 'Platform_NS',
        'Platform_PC', 'Platform_PS', 'Platform_PS2', 'Platform_PS3',
        'Platform_PS4', 'Platform_PSP', 'Platform_Wii', 'Platform_WiiU',
        'Platform_X360', 'Platform_XB', 'Platform_XOne'])

        importances = v1
        feat_importances = pd.Series(importances, index=v2)

        # Trier les variables par ordre d'importance décroissante
        feat_importances = feat_importances.sort_values(ascending=False).head(10)

        # Afficher le graphique en barres
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(feat_importances.index, feat_importances.values, color="darkviolet")
        ax.set_title("Importance de chaque variable")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=75)
        

        # Afficher le graphique dans Streamlit
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        st.pyplot(fig)
        st.markdown('Notre analyse nous indique que les variables Critic_score et Year sont celles qui ont le plus de poids.')
        st.markdown('<p class="purple">Etape 5: Calcul des meilleurs hyperparamètres via une GridSearch.</p>', unsafe_allow_html=True)
        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire).")
        
        
    with tab3:
        
        # Graph Résultats
        x = ["Régression Logistique","Arbre de Décision","Random Forest"]
        y = [0.11884109217929484,-0.2,0.10571851988597436]
        fig, ax = plt.subplots(figsize=(12, 6))
        color = ['#EEA2AD','#87CEFA','#8470FF']
        ax.bar(x, y, color=color, width=0.6)
        ax.set_ylim(-0.5, 1)
        ax.grid(axis='y')
        ax.set_title("Résultats")
        ax.tick_params(axis="x", rotation=55)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        st.pyplot(fig)
        st.markdown('>Les performances des modèles ne sont pas bonnes.  \nNous ne pouvons donc prévoir les ventes !')







if selected == "Conclusion":
    st.markdown("""Des difficultés ont été rencontrées pour prédire les ventes en quantité en partant de notre dataset. 

**Les difficultés principales identifiées sont:**
* Le match de la collecte de données via scrapping
* Le non partitionnement par la durée sur les ventes

Les approches que nous pourrions proposer pour pallier ces difficultés:
* Trouver un positionnement sur le marché avec le :
    * Point de vue du studio
    * Point de vue du publisher
    * Point de vue sur une période de temps

* Interpréation différente entre les séries de jeux (GTA) et les one-shoot""")

    st.markdown("""
    **Il faut également prendre en compte pour ces projections les éléments suivants:**
    
    * Analyse de sentiment pour prédire l'engouement avant la sortie des jeux (commentaires sur le trailer, nombre de vues) 
    * Analyse du marché:
        * Concurrence: cours de la bourse sur les acteurs du jeu vidéo
        * Economique: pouvoir d'achat sur le marché (ex: crise économique)
        * Socioculturel: étude sur les comportements des consommateurs (ex: genre attendu par âge, pays)
        * Technologique: innovations sur le marché
        * Saisonnalité""")
