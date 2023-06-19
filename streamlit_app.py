import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from bokeh.plotting import figure, output_notebook, show, curdoc
from bokeh.models import BoxAnnotation, ColumnDataSource, Range1d
from bokeh.models.tools import HoverTool
from bokeh.layouts import row
from bokeh.models import Range1d, LabelSet, TabPanel, Tabs
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
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from PIL import Image
import streamlit.components.v1 as components

#[theme]
#base="light"
#primaryColor="#9400d3"

def set_theme():
    st.set_page_config(
        page_title="Game Project Prediction",
        page_icon=":video_game:",
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

# dictionnaire de couleurs
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
flierprops = dict(marker="X", markerfacecolor='darkviolet', markersize=12,
                  linestyle='none')
embed_component_CS= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class="badge-base LI-profile-badge" data-locale="fr_FR" data-size="medium" data-theme="dark" data-type="VERTICAL" data-vanity="cindysotton" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://fr.linkedin.com/in/cindysotton?trk=profile-badge"></a></div>"""}

embed_component_CA= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class=“badge-base LI-profile-badge” data-locale=“fr_FR” data-size=“medium” data-theme=“dark” data-type=“VERTICAL” data-vanity=“celine-anselmo” data-version=“v1”><a class=“badge-base__link LI-simple-link” href=“https://fr.linkedin.com/in/celine-anselmo?trk=profile-badge”>Céline ANSELMO</a></div>"""}

embed_component_DR= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class=“badge-base LI-profile-badge” data-locale=“fr_FR” data-size=“medium” data-theme=“dark” data-type=“VERTICAL” data-vanity=“dorian-rivet” data-version=“v1”><a class=“badge-base__link LI-simple-link” href=“https://fr.linkedin.com/in/dorian-rivet?trk=profile-badge”>Dorian Rivet</a></div>"""}

embed_component_KM= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class=“badge-base LI-profile-badge” data-locale=“fr_FR” data-size=“medium” data-theme=“dark” data-type=“VERTICAL” data-vanity=“karine-minatchy-6644a5136” data-version=“v1”><a class=“badge-base__link LI-simple-link” href=“https://www.linkedin.com/in/karine-minatchy-6644a5136?trk=profile-badge”>Karine Minatchy</a></div>"""}


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
    
    st.subheader("Contacter l'équipe")
    col1, col2 = st.columns(2)

    with col1:
        components.html(embed_component_CA['linkedin'],height=310)
  
    with col2:
        components.html(embed_component_CS['linkedin'],height=310)

    col3, col4 = st.columns(2)
    with col3:
        components.html(embed_component_KM['linkedin'],height=310)
    with col4:
        components.html(embed_component_DR['linkedin'],height=310)
        



# Méthodologie
if selected == "Méthodologie":
    image = Image.open('media/manette.jpeg')
    st.image(image)
    tab1, tab2, statistiques = st.tabs(["Extraction des données", "Data processing",'Statistiques'])
    with tab1:
        st.markdown('Nous avons scrappé le site Vgchart pour récupérer :  \n - Des données plus récentes  \n - Variables Critic Score et Studio en plus')
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
        st.markdown('<p class="purple">1 - Nettoyage de données:  </p>'"\n  \n - Formater le type des varibles si nécessaire (str,int,date)  \n  \n- Remplacement et ou suppression des Nans      \n  \n   \n", unsafe_allow_html=True)
        st.markdown('<p class="purple">2 - Transformation des données:</p>'  "\n \n - Supprimer les outliers des variables explicatives \n \n - Encodage de la variable plateforme \n \n    - Application d'un get dummies sur la variable plateforme \n    - Reverse du get dummies pour obtenir un résultat tel que Multi-Plateforme ou le nom de la plateforme \n - Pivot du dataset pour obtention d'une ligne par jeu", unsafe_allow_html=True)
    
    with statistiques:
        #Analyses statistiques

        # header violet
        st.markdown("""
        <style>
        .bigpurple {
            color : darkviolet;
            font-size : 35px;
        }
        </style>
        """, unsafe_allow_html=True)

        #Global_Sales
        st.markdown('<p class="bigpurple">Corrélation avec la variable cible: Global_Sales:</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="bigpurple">Corrélation entre les variables explicatives:</p>', unsafe_allow_html=True)
        st.markdown("Nous avons utilisé la méthode statistique V de Cramer pour mesurer le niveau de corrélation entre nos variables explicatives de type qualitatives")
        
        #image stats2
        image_path = "media/stats2.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="V de Cramer")
        except:
            st.error("Impossible d'afficher l'image")

        st.markdown("Cette analyse a permis de mettre en valeur des relations fortes comme le genre et le publisher et de mettre de côté des analyses comme platform et genre où il n'y a pas de corrélation.")

# Modelisation
if selected == "Modélisation":
    tab1, tab2, tab3 = st.tabs(["Introduction", "Etapes de la Modélisation", 'Résultats'])
    with tab1:
        st.markdown('### Dataset Modélisation:')
        st.text('Variable Cible')
        st.markdown('-  Global_Sales')
         
        
        st.text('Variables Explicatives')
            

        st.markdown("-  L'année de sortie (Year)  \n-  Le genre (Genre)  \n-  Le studio l’ayant développé (Studio)  \n-  L’éditeur l’ayant publié (Publisher)  \n-  La plateforme sur laquelle le jeu est sortie (Platform)  \n-  Les notes utilisateurs (Critic_Score)")  
        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire). ")
        
    with tab2:
        #Préprocessing Etapes
        st.markdown('### Pre-processing:')
        st.markdown('Etape 1: Clustering des variables studio et publisher')
        st.markdown('-  Suite au nombre important de modalités dans ces trois colonnes (+700 pour studio), nous avons simplifié les variables en utilisant la méthode du clustering. Studio, Publisher: 1 à 4 suivant leur montant de Global_Sales')
        st.markdown('Etape 2: Suppression des variables non pertinentes pour la modélisation (name, region sales, description)  \n Etape 3: Encoding des variables catégorielles')
        #Modélisation Etapes
        st.markdown('### Modélisation:')
        #st.markdown('Etape 1: Clustering des variables studio et publisher')
        st.markdown("Etape 2: Analyse de l'importance des variables, Itération 2  \n")
        st.markdown('### Résultats:')
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(feat_importances.index, feat_importances.values, color="darkviolet")
        ax.set_title("Importance de chaque variable")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=75)
        

        # Afficher le graphique dans Streamlit
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        st.pyplot(fig)
        st.markdown('Notre analyse nous indique que les variables Critic_score et Year sont celles qui ont le plus de poids.')
        st.markdown("Etape 3: Calcul des meilleurs hyperparamètres via une GridSearch.")
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        


        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire).")
        
        
    with tab3:
        st.markdown("### Résultats")
        
        # Graph Résultats
        x = ["Régression Logistique","Arbre de Décision","Random Forest"]
        y = [0.11884109217929484,-0.2,0.10571851988597436]
        fig, ax = plt.subplots(figsize=(6, 6))
        color = ['#EEA2AD','#87CEFA','#8470FF']
        ax.bar(x, y, color=color, width=0.6)
        ax.set_ylim(-0.5, 1)
        ax.grid(axis='y')
        ax.set_title("Résultats")
        ax.tick_params(axis="x", rotation=55)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        st.pyplot(fig)
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('>Les performances des modèles ne sont pas bonnes.  \nNous ne pouvons donc prévoir les ventes !')
if selected == "Conclusion":
    st.markdown("Des difficultés ont été rencontrées pour prédire les ventes en quantité en partant de notre dataset.  \n  Les difficultés principales identifiées sont:\n- Le match de la collecte de données via scrapping  \n - Le non partitionnement par la durée sur les ventes \n  \nLes approches que nous pourrions proposer pour pallier ces difficultés: \n- Trouver un positionnement sur le marché avec  \n     -  Point de vue du studio  \n   - Point de vue du publisher \n   - Point de vue sur une période de temps \n   - Interpréation différente entre les séries de jeux (GTA) et les one-shoot")
    st.markdown("Il faut également prendre en compte pour ces projections les éléments suivants:  \n - Analyse de sentiment pour prédire l'engouement avant la sortie des jeux (commentaires sur le trailer, nombre de vues)  \n - Analyse du marché: \n     - Concurrence: cours de la bourse sur les acteurs du jeu vidéo \n     -  Economique: pouvoir d'achat sur le marché (ex: crise économique) \n     -  Socioculturel: étude sur les comportements des consommateurs (ex: genre attendu par âge, pays) \n     - Technologique: innovations sur le marché  \n     - Saisonnalité")
if selected == "Analyse":
    genre = st.radio(
    "Menu:",
    ('Le marché','Plateformes', 'Publishers', 'Studios','Genres','Notes'))
    st.divider()
    if genre == "Le marché":
        st.title('VgChartz : Analyse des données')
        st.markdown("Estimer les ventes d'un produit avant de le lancer est une étape essentielle dans la vie d'un produit. C'est ce que nous allons essayer de faire dans le cadre de ce projet.  \n Notre étude nous portera dans l'univers du jeu vidéo.")
        st.subheader("Ventes globales par années")
        data_NA = df.groupby(by=['Year'])['NA_Sales'].sum()
        data_NA = data_NA.reset_index()
        data_EU = df.groupby(by=['Year'])['EU_Sales'].sum()
        data_EU = data_EU.reset_index()
        data_JP = df.groupby(by=['Year'])['JP_Sales'].sum()
        data_JP = data_JP.reset_index()
        data_Others = df.groupby(by=['Year'])['Other_Sales'].sum()
        data_Others = data_Others.reset_index()
        data_globales = df.groupby(by=['Year']).sum()
        data_globales = data_globales.reset_index()
        p1 = figure(width = 600, height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
        p2 = figure(width = 600, height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
        p3 = figure(width = 600, height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
        p4 = figure(width = 600, height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
        p5 = figure(width = 600, height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
        source1 = ColumnDataSource(data_NA)
        source2 = ColumnDataSource(data_EU)
        source3 = ColumnDataSource(data_JP)
        source4 = ColumnDataSource(data_Others)
        source5= ColumnDataSource(data_globales)
        p1.line(x = "Year",
            y = "NA_Sales",
            line_width = 3,
            color = "darkviolet",
            source = source1)
        p5.line(x = "Year",
            y = "NA_Sales",
            line_width = 3,
            color = "darkviolet",
            source = source1,
        legend_label="NA_Sales")
        p2.line(x = "Year",
            y = "EU_Sales",
            line_width = 3,
            color = "royalblue",
            source = source2)
        p5.line(x = "Year",
            y = "EU_Sales",
            line_width = 3,
            color = "royalblue",
            source = source2,
        legend_label="EU_Sales")
        p3.line(x = "Year",
        y = "JP_Sales",
        line_width = 3,
        color = "hotpink",
        source = source3)
        p5.line(x = "Year",
        y = "JP_Sales",
        line_width = 3,
        color = "hotpink",
        source = source3,
        legend_label="JP_Sales")
        p4.line(x = "Year",
        y = "Other_Sales",
        line_width = 3,
        color = "aqua",
        source = source4)
        p5.line(x = "Year",
        y = "Other_Sales",
        line_width = 3,
        color = "aqua",
        source = source4,
        legend_label="Other_Sales")
        p5.line(x = "Year",
        y='Global_Sales',
        line_width = 3,
        color = "gray",
        source = source5,
        legend_label="Global_Sales")
        labels = LabelSet(x='weight', y='height', text='names',
                x_offset=5, y_offset=5)
        p5.add_layout(labels)
        tab1 = TabPanel(child = p1,
                title = "NA_Sales")
        tab2 = TabPanel(child = p2,
                title = "EU_Sales")
        tab3 = TabPanel(child = p3,
                title = "JP_Sales")
        tab4 = TabPanel(child = p4,
                title = "Others_Sales")
        tab5 = TabPanel(child = p5,
                title = "Globales")
        tabs = Tabs(tabs = [tab5, tab2, tab3, tab4, tab1])
        doc = curdoc()
        doc.theme = 'dark_minimal'
        doc.add_root(tabs)
        st.bokeh_chart(tabs, use_container_width=True)
        st.markdown(
            """Le marché du jeu vidéo a commencé sa croissance à partir de la seconde moitié des années 90 dynamisé par le lancement de nouvelles plateformes: \n\n -   Sortie de la PlayStation en 1995 \n\n - Nouvel élan dans les années 2000 avec la sortie de la Nintendo 64. \n\n - Après une forte croissance (2005 à 2010), le marché revient à sa tendance initiale. """)
        
        st.subheader("Ventes globales par zones géographiques")
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
        st.markdown(
            """Nos ventes se concentrent sur trois principaux marchés : North America, Europe, Japon (≈90%). Les ventes sur d'autres marchés sont inférieures à 10%. A noter la concentration particulière d'une part avec : \n North Amercia qui réalise près de la moitié des ventes. \n Le Japon qui réalise plus de 10% des ventes à mettre en perspecitive avec le nombre d'habitants.""")
        
        st.subheader("Ventes globales par jeux")
        source = ColumnDataSource(df)
        hover = HoverTool(
        tooltips=[
            ("name", "@Name"),
            ("Genre", "@Genre"),
            ("Platform", "@Platform"),
            ("Studio", "@Studio"),
            ("Note", '@Critic_Score') ])
        p98 = figure(plot_width=800, plot_height=700,x_axis_label='Year', y_axis_label='Global_Sales')
        doc = curdoc()
        doc.theme = 'dark_minimal'
        doc.add_root(p98)
        p98.circle(x='Year',y='Global_Sales',source = source,color='darkviolet',size=10)
        p98.add_tools(hover)
        st.bokeh_chart(p98, use_container_width=True)
        st.markdown("Certains jeux ont connu un succès exceptionnel, c'est notamment le cas pour Wii Sport sorti en 2006 chez Nintendo.")
        st.markdown("Ce graphique nous permet d'identifier les jeux qui se sont démarqués en terme de ventes et apprécier les variables en lien. On peut identifier des \"sagas\" qui ont bien marché (Mario, Pokemon, Grand Theft Auto).")

    if genre == 'Notes':
        
        st.subheader("Répartition des notes")
        df['cat_Notes'] = pd.cut(df['Critic_Score'], bins = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['0 à 1','1 à 2','2 à 3','3 à 4','4 à 5','5 à 6','6 à 7', '7 à 8', '8 à 9', '9 à 10'])
        # Remplacer les modalités peu nombreuse par Autre
        df['cat_Notes'] = df['cat_Notes'].replace(['2 à 3', '1 à 2', '0 à 1'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df3 = df['cat_Notes']
        df3.str.split(',', expand=True).stack().reset_index(drop=True)
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                                'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                                'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df3,
                    values=df3.value_counts(),
                    names=df3.value_counts().index,
                    color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("Nous avons représenté ici la répartition des notes de notre liste de jeux.Il sera intéressant d'observer par la suite si l'on peut s'appuyer sur les notes pour valider nos analyses sur les autres variables. On observe près de 75% des jeux qui ont une note comprise entre 5 et 8.")
        
    
        st.markdown("## Zoom sur les Notes")
        df['cat_Notes'] = pd.cut(df['Critic_Score'], bins = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['0 à 1','1 à 2','2 à 3','3 à 4','4 à 5','5 à 6','6 à 7', '7 à 8', '8 à 9', '9 à 10'])
        # Remplacer les modalités peu nombreuse par Autre
        df['cat_Notes'] = df['cat_Notes'].replace(['2 à 3', '1 à 2', '0 à 1'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df3 = df['cat_Notes']
        df3.str.split(',', expand=True).stack().reset_index(drop=True)
        DICT_NOTE = {'7 à 8': 'dodgerblue',
                '8 à 9': 'tomato',
                '6 à 7': 'mediumaquamarine',
                '5 à 6': 'mediumpurple',
                '4 à 5': 'sandybrown',
                '9 à 10': 'lightskyblue',
                '3 à 4': 'hotpink',
                'Autre': 'palegreen'}
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### Réparition des notes par notes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["cat_Notes"].value_counts().head(10).index,
                    x=df["cat_Notes"].value_counts().head(10).values,
                        palette = DICT_NOTE)
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        with col2:
            st.markdown("### Répartition des notes par ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['cat_Notes', 'Global_Sales']]
            df_publisher = df_publisher.groupby('cat_Notes')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="cat_Notes", x="Global_Sales",data=df_publisher, palette = DICT_NOTE);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("On peut donc considérer que les jeux sont en règle générale qualitatifs. De manière logique, les jeux les mieux notés sont ceux qui se vendent le plus.")
        st.subheader("Evolution des notes par genres et par années")
        fig, ax = plt.subplots(5, 1, sharex=True,sharey=True,
                            figsize=(10, 5))
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        liste=['Misc','Action','Shooter','Adventure','Sports']
        color=['violet','tomato','mediumaquamarine','hotpink','mediumpurple']
        for index, i in enumerate(liste):
            dfsource =pd.DataFrame(df[df.Genre ==i].groupby(['Genre', 'Year']).mean()).reset_index()
            source = ColumnDataSource(dfsource)
            sns.lineplot(x = "Year",
                y = "Critic_Score",
                data=dfsource,
                color=color[index],
                label=i,
                        ax=ax[index])
            ax[index].set_ylabel('')
        st.pyplot(fig, theme="streamlit", use_container_width=True) 
        st.markdown("Nous observons sur le graphique ci-contre l'évolution de la note par année sur les genres que nous avions commenté dans la partie Genre. Les genres Misc, Action, Sports semblent afficher une continuité. Nous observons pour Shooter et Adventures des pic de décroissances qu'il pourra être intéressant d'analyser plus en détails. Il y a-t-il un jeu qui aurait été mal reçu du public sur ces pics.")
    if genre == 'Plateformes':
        st.markdown('### Zoom sur les plateformes')
        df1 = df[df.columns[11:]]
        # Remplacer les petites valeurs par autre
        df1['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
            'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        # Remplacer les petites valeurs par autre aussi dans df
        df['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
            'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        df1 = pd.Series(df1["Platform"])
        df1.str.split(',', expand=True).stack().reset_index(drop=True)
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
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                        'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                        'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df1,
            values=df1.value_counts(),
            names=df1.value_counts().index,
            color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("Nous constatons que les parts de marché se répartissent de manière équilibré entre les plateformes.A noter que certaines plateformes tendent à disparaitre car remplacer par leur upgrade (PS2 qui devient la PS3).")

        col1,col2 = st.columns(2)
        with col1: 
            st.subheader("Top 10 modalités")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Platform"].value_counts().head(10).index,
                    x=df["Platform"].value_counts().head(10).values, palette=DICT_PLAT);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Nombre de ventes median")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            df_publisher = df_publisher.head(10)
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            
        st.markdown("Les jeux vendus sur plusieurs plateformes ont le meilleur ratio de vente par jeu. PSP et GBA sont les deux premiers en terme de quantités de jeux mais perdent leur place lorsque l'on regarde le nombre de vente total et médian. A part pour la PS3, l'ensemble des autres plateforme à le même nombre de vente médian. Néanmoins, à restituer avec le nombre de plateformes en exploitation à ce moment.  \n Certaines plateformes vont avoir des jeux qui auront des ventes disproportionnées par rapport au reste de leur catalogue d'où un écart important entre la moyenne et la médiane des ventes (ex : Wii). Le graphique suivant nous permettra de mieux observer  ")
        with col2: 
            st.subheader("Top 10 ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Nombre de ventes moyen")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False)
            df_publisher = pd.DataFrame(df_publisher).reset_index().head(10)
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les plateformes")
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 10))
        
        df1 = df[df.columns[11:]]
        # Remplacer les petites valeurs par autre
        df1['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
        'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        # Remplacer les petites valeurs par autre aussi dans df
        df['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
        'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        df1 = pd.Series(df1["Platform"])
        df1.str.split(',', expand=True).stack().reset_index(drop=True)
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
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Platform',
                    y='Global_Sales',
                    palette = DICT_PLAT,
                    flierprops=flierprops,
                    data=df[df.Platform.isin(list(df.Platform.value_counts().index))])
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, use_container_width=True)
        st.markdown("Cette représentation graphique met en évidence le constat effectué précédemment à savoir que la plateforme Wii à un outlier. Il s'agit de Wii Sport qui fait des records de ventes par rapport aux autres jeux de Wii. Nos recherches nous ont indiqué que ce jeu est sorti en 2006 en même temps que la console Wii ce qui a participé à l'engouement et l'explosion des ventes. Le jeu faisait parti d'une offre bundle avec la Wii. La DS a également des valeurs extrêmes qu'il sera intéressant de regarder avec New Super Mario.")
        st.subheader("Analyse de la corrélation de la variable Platform")
        comp_platform = df[['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',"Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Platform']).sum().sort_values(by="Global_Sales",ascending=False)
            # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)

        
    if genre == "Publishers":
        st.markdown("### Zoom sur les Publishers")
        st.markdown("Avant propos :  \n Nous avons plus de 200 modalités pour cette variable.Il sera intéressant d'observer par la suite les évolutions au cours du temps et les différences par pays.Les publishers les plus importants en terme de production de jeux sont Nintendo, Sony CE, Ubisoft ou encore Electronics Arts.")
        DICT_PUBLISHER = {'Nintendo' : 'dodgerblue','Sony Computer Entertainment':'tomato','Ubisoft':'mediumaquamarine','Electronic Arts':'mediumpurple',
            'Sega':'sandybrown','Konami':'lightskyblue','Activision':'hotpink','THQ':'palegreen','Capcom':'violet','Atlus':'gold',"Rockstar Games":'lavender',
            'Mojang':'salmon','RedOctane':'aquamarine','EA Sports' : 'darkseagreen','Microsoft Game Studios':'moccasin','Sony Computer Entertainment America':'rosybrown','Broderbund':'blue',
            'MTV Games':'plum','ASC Games':"turquoise",'Valve':'indianred','Hello Games':'peachpuff','Microsoft Studios':'lemonchiffon','Valve Corporation':'lightcoral',
            'Bethesda Softworks':'paleturquoise','LucasArts':'steelblue','Virgin Interactive':'chocolate','Sony Interactive Entertainment':'mediumorchid',
            'Blizzard Entertainment':'yellowgreen','City Interactive':'slategrey','Rare':'cornsilk','Square':'cadetblue','Warner Bros. Interactive':'pink'}
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 modalités")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Publisher"].value_counts().head(10).index,
                x=df["Publisher"].value_counts().head(10).values, palette = DICT_PUBLISHER );
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            
            st.subheader("Nombre de ventes median")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].median().sort_values(ascending=False).head(20)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("Le top 4 est globalement constant en terme de quantité de jeux vendus et de production de jeux. Mais les 4 publishers perdent leur place lorsque l'on regarde le ratio ventes/quantités median. Nintendo se démarque en terme de référence sorties et de ventes. Mojang et RedOctane ont quand à eux un très bon ratio de vente médian alors qu'ils sont absents des deux premiers graphiques. On peut constater que les deux premiers Publisher ne sont pas sensibles à des valeurs extrêmes.En revanche Rockstar G, Sony IE, Bethesda S ou encore Nitendo ont quand à eux des jeux qui auront eu des ventes peu représentatives des ventes médianes de leur catalogue. ")
        with col2:
            st.subheader("Top 10 ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)


        

            st.subheader("Nombre de ventes moyen")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].mean().sort_values(ascending=False).head(20)
            df_publisher = pd.DataFrame(df_publisher).reset_index().head(20)
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les publisher")
        plt.figure(figsize=(15, 10))
        sns.set(style="dark", context="talk")
        plt.style.use("dark_background")
        df5= df.loc[(df['Publisher'] == 'Mojang') 
        | (df['Publisher'] =='RedOctane')
        | (df['Publisher'] =='Rockstar Games')
        | (df['Publisher'] =='Sony Interactive Entertainment')
        | (df['Publisher'] =='Bethesda Softworks')
        | (df['Publisher'] =='Nintendo')]
        # création du dico
        DICT_PUBLISHER = {'Nintendo' : 'dodgerblue','Sony Computer Entertainment':'tomato','Ubisoft':'mediumaquamarine','Electronic Arts':'mediumpurple',
                    'Sega':'sandybrown','Konami':'lightskyblue','Activision':'hotpink','THQ':'palegreen','Capcom':'violet','Atlus':'gold',"Rockstar Games":'lavender',
                    'Mojang':'salmon','RedOctane':'aquamarine','EA Sports' : 'darkseagreen','Microsoft Game Studios':'moccasin','Sony Computer Entertainment America':'rosybrown','Broderbund':'blue',
                    'MTV Games':'plum','ASC Games':"turquoise",'Valve':'indianred','Hello Games':'peachpuff','Microsoft Studios':'lemonchiffon','Valve Corporation':'lightcoral',
                    'Bethesda Softworks':'paleturquoise','LucasArts':'steelblue','Virgin Interactive':'chocolate','Sony Interactive Entertainment':'mediumorchid',
                    'Blizzard Entertainment':'yellowgreen','City Interactive':'slategrey','Rare':'cornsilk','Square':'cadetblue','Warner Bros. Interactive':'pink'}
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Publisher',
                y='Global_Sales',
                palette = DICT_PUBLISHER,
                flierprops=flierprops,
                data=df5)
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig3 = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig3, use_container_width=True)
        st.markdown("Analyse des outliers :  \n Nintendo : Wii Sport sorti en 2006 la saga Mario  \n Rockstar Games : La saga GTA (dont le V sorti en 2014)  \n Bethseda Softworks :The Elder Scrolls sorti en 2011 Fallout 4 sorti en 2015.  \n On remarque pour Sony que celui ci n'est pas concerné par des valeurs extrêmes. Cependant en regardant la distribution de ces modalités, nous constatons que 50% de celles ci sont en dessous de 1,6M de vente et que sa valeur max. est de 10,33M. D'ou un écart entre la moyenne et la médiane.")
        st.subheader("Analyse de la corrélation de la variable Publisher")
        comp_platform = df[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', "Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Publisher']).sum().sort_values(by = "Global_Sales", ascending=False).head(10)
        # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)
        
    if genre =="Genres":
        curdoc().theme = 'dark_minimal'
        liste=['Misc','Action','Shooter','Adventure','Sports']
        color=['violet','tomato','mediumaquamarine','hotpink','mediumpurple']
        df_bokeh = df.Year>1992
        p = figure(plot_width = 1000, plot_height = 600,x_axis_label='Year', y_axis_label='Genre')
        for index, i in enumerate(liste):
                dfsource =pd.DataFrame(df[df.Genre ==i].groupby(['Genre', 'Year']).count()).reset_index()
                source = ColumnDataSource(dfsource)
                p.line(x = "Year",
                    y = "Name",
                    line_width = 3,
                    color=color[index],
                    source = source,
                    legend_label=i)
                doc = curdoc()
                doc.theme = 'dark_minimal'
                doc.add_root(p)
                p.legend.click_policy="mute"
        st.bokeh_chart(p, use_container_width=True)
        st.markdown(
        "Role-Playing, Action, Shooter, Sports représentent la moitié des parts de marché.")
        st.subheader("Répartition des genres")
        # Remplacer les modalités peu nombreuse par Autre
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                                'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                                'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df2,
                    values=df2.value_counts(),
                    names=df2.value_counts().index,
                    color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        
        
        
        # Remplacer les modalités peu nombreuse par Autre
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
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
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 références")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Genre"].value_counts().head(10).index,
                    x=df["Genre"].value_counts().head(10).values, palette =DICT_GENRE );
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Ventes médianes")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher)
            st.pyplot(fig,theme="streamlit", use_container_width=True)
            
        st.markdown("Le top 6 reste le même, même si les genres s'interchange. Les genres de \"niche\" à savoir l'étiquette Autre, sont les genres les plus \"rentables\". L'écart entre les ventes médianes et moyennes est considérable, nous pouvons donc affirmer que cette variable est très sujette à des outliers.")
        with col2:
            st.subheader("Top 10 ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Ventes moyennes")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=False)
            df_publisher = pd.DataFrame(df_publisher).reset_index().head(10)
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse de la corrélation de la variable Genre")
        comp_genre = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
        comp_map = comp_genre.groupby(by=['Genre']).sum()
        plt.figure(figsize=(20, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True) 
        

        st.subheader("Analyse des valeurs extrêmes pour les genre")
        # Remplacer les modalités peu nombreuse par Autre
        fig = plt.figure(figsize=(20, 10))
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
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
        bx=sns.boxplot(x='Genre',
                y='Global_Sales',
                palette = DICT_GENRE,
                flierprops=flierprops,
                data=df[df.Genre.isin(list(df.Genre.value_counts().index))])
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="dark")
        plt.style.use("dark_background")
        fig = bx.get_figure()
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, use_container_width=True)        
    if genre == "Studios":
        st.markdown("### Zoom sur les Studios")
        st.markdown("Avant propos :  \n Cette variable possède plus de 1000 modalités. Pour pouvoir l'analyser, nous avons sélectionné des top selon la fréquence et les ventes.")
        DICT_STUDIO = {'Capcom': 'dodgerblue',
                        'Konami': 'tomato',
                        'Nintendo EAD': 'mediumaquamarine',
                        'EA Canada': 'mediumpurple',
                        'Square Enix': 'sandybrown',
                        'Ubisoft Montreal': 'lightskyblue',
                        'EA Tiburon': 'hotpink',
                        'Namco': 'palegreen',
                        'Ubisoft ': 'violet',
                        'Sonic Team': 'gold',
                        'Hudson Soft': 'lavender',
                        'Rare Ltd.': 'salmon',
                        'Atlus Co.': 'aquamarine',
                            'Ubisoft': "orchid",
                        'Game Freak': "plum",
                        'Rockstar North': "lavender",
                        'Infinity Ward':"magenta",
                        "Traveller's Tales":"blue",
                        'Treyarch':"mediumpurple",
                        'Good Science Studio':"turquoise",
                        'Nintendo SDD': "slateblue",
                        'Sledgehammer Games': "lightcoral",
                        'Dice': "peachpuff",
                        'Neversoft': "lemonchiffon",
                        'Nintendo EAD / Retro Studios': "mintcream",
                        'Rockstar Games': "powderblue",
                        'EA DICE': "navy",
                        'Bethesda Game Studios':"palegreen",
                        'Polyphony Digital': "blueviolet",
                        '4J Studios': "bisque",
                        'Nintendo EAD Tokyo':"azure",
                        'Bungie Studios': "steeblue",
                        'Project Sora':"chocolate",
                        'Naughty Dog': "mediumorchid",
                        'Team Bondi': "lightcyan",
                        'Level 5 / Armor Project': "tomato",
                        "Bungie":"deeppink"
            }
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 modalités")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Studio"].value_counts().head(10).index,
                        x=df["Studio"].value_counts().head(10).values, palette = DICT_STUDIO );
            st.pyplot(fig,theme="streamlit", use_container_width=True)


            st.subheader("Ventes médianes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 11))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].median().sort_values(ascending=False).head(15)
            df_studio = pd.DataFrame(df_studio).reset_index()
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO,data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            
        st.markdown("Nous remarquons que les répartitions sur les Studio n'est pas la même que sur les variables précédentes. En revanche le constat sur l'analyse du nombre de vente médian reste le même. On remarque que les échelles entre la médiane et la moyenne sont équivalentes.Infinity Ward perd 5M entre les 2 indicateurs et les 2 derniers Studios ne sont pas les mêmes. Nous pouvons en conclure que cette variable n'est pas très sensible aux valeurs extrêmes pour ces Studios. ")
        with col2:
            st.subheader("Top 10 des ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_studio = pd.DataFrame(df_studio).reset_index()
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO,data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        
            st.subheader("Ventes moyennes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 11))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].mean().sort_values(ascending=False).head(15)
            df_studio = pd.DataFrame(df_studio).reset_index().head(15)
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO, data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les studios")
        plt.figure(figsize=(15, 10))
        sns.set(style="dark", context="talk")
        plt.style.use("dark_background")
        df6= df.loc[(df['Studio'] == 'Infinity Ward') 
        | (df['Studio'] =='Sledgehammer Games')
        | (df['Studio'] =='Nintendo SDD')
        | (df['Studio'] =='Good Science Studio')
        | (df['Studio'] =='Nintendo EAD')
        | (df['Studio'] =='Treyarch')
        | (df['Studio'] =='4J Studios')
        | (df['Studio'] =='Nintendo EAD Tokyo')]
        # création du dico
        DICT_STUDIO = {'Capcom': 'dodgerblue',
        'Konami': 'tomato',
        'Nintendo EAD': 'mediumaquamarine',
        'EA Canada': 'mediumpurple',
        'Square Enix': 'sandybrown',
        'Ubisoft Montreal': 'lightskyblue',
        'EA Tiburon': 'hotpink',
        'Namco': 'palegreen',
        'Ubisoft ': 'violet',
        'Sonic Team': 'gold',
        'Hudson Soft': 'lavender',
        'Rare Ltd.': 'salmon',
        'Atlus Co.': 'aquamarine',
        'Ubisoft': "orchid",
        'Game Freak': "plum",
        'Rockstar North': "lavender",
        'Infinity Ward':"magenta",
        "Traveller's Tales":"blue",
        'Treyarch':"mediumpurple",
        'Good Science Studio':"turquoise",
        'Nintendo SDD': "slateblue",
        'Sledgehammer Games': "lightcoral",
        'Dice': "peachpuff",
        'Neversoft': "lemonchiffon",
        'Nintendo EAD / Retro Studios': "mintcream",
        'Rockstar Games': "powderblue",
        'EA DICE': "navy",
        'Bethesda Game Studios':"palegreen",
        'Polyphony Digital': "blueviolet",
        '4J Studios': "bisque",
        'Nintendo EAD Tokyo':"azure",
        'Bungie Studios': "steeblue",
        'Project Sora':"chocolate",
        'Naughty Dog': "mediumorchid",
        'Team Bondi': "lightcyan",
        'Level 5 / Armor Project': "tomato",
        "Bungie":"deeppink"}
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Studio',
                y='Global_Sales',
                palette = DICT_STUDIO,
                flierprops=flierprops,
                data=df6)
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig4 = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig4, use_container_width=True)
        st.markdown("Cette représentation graphique confirme les constats précédents.Pour Treyarch les outliers correspondent aux jeux :  \n Call of Duty: Black Ops sorti en 2010  \n Call of Duty: World at War : 2008   \n Pour Nintendo EAD : Wii Sport sorti en 2006  \n   -   La saga Mario constitue le reste des outliers")
        st.subheader("Analyse de la corrélation de la variable Studio")
        comp_platform = df[['Studio', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', "Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Studio']).sum().sort_values(by = "Global_Sales", ascending=False).head(10)
        # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)
        st.markdown("Nintendo EAD est le leader du marché. En excluant Nintendo, on constate qu'il y a un studio qui se démarque en fonction des régions :  \n -    NA : EA Tiburon\n -    EU : EA Canada\n -    JP : Game Freak et Capcom")
