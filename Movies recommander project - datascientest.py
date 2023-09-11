#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[354]:


df_ratings = pd.read_csv('movies/rating.csv',nrows=100001)
df_movies = pd.read_csv('movies/movie.csv')
df_genscore = pd.read_csv('movies/genome_scores.csv',nrows=100001)
df_gentags = pd.read_csv('movies/genome_tags.csv',nrows=100001)
df_link = pd.read_csv('movies/link.csv',nrows=100001)
df_tag = pd.read_csv('movies/tag.csv',nrows=100001)


# In[355]:


#Afficher les premières lignes du DataFrame pour avoir un aperçu initial.
print(df_ratings.head())


# In[356]:


#Obtenir des informations sur le jeu de données, telles que le nombre de lignes, le nombre de colonnes et les types de données.

print(df_ratings.info())


# - Obtenir des statistiques sommaires sur les colonnes numériques, comme la moyenne, l'écart-type, le minimum et le maximum.
# 
# 

# In[357]:


print(df_ratings.describe())


# In[358]:


# histogramme pour visualiser la distribution des évaluations attribuées aux films.
plt.hist(df_ratings['rating'], bins=10, edgecolor='black')
plt.xlabel('Évaluation')
plt.ylabel('Nombre d\'utilisateurs')
plt.title('Distribution des évaluations')
plt.show()


# - Identification des films les plus évalués en comptant le nombre d'avis par film.

# In[359]:


top_movies = df_ratings['movieId'].value_counts().head(10)
print(top_movies)


# - Les utilisateurs qui ont contribué le plus grand nombre d'évaluations.

# In[360]:


top_users = df_ratings['userId'].value_counts().head(10)
print(top_users)


# - Décomposition de timestamp

# In[361]:


df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], errors='coerce')
df_ratings['year'] = df_ratings['timestamp'].dt.year
df_ratings['month'] = df_ratings['timestamp'].dt.month
df_ratings['time'] = df_ratings['timestamp'].dt.time
df_ratings['date'] = df_ratings['timestamp'].dt.date


# In[362]:


df_ratings.head()


# In[363]:


print('Année :          ', df_ratings.year.unique())
print('Mois :         ', df_ratings.month.unique())
print('Note unique : ', df_ratings.rating.unique())
print('Note moyenne : ', round(df_ratings.rating.mean(), 2))
print('Fréquence de chaque valeur dévaluation:\n',df_ratings['rating'].value_counts())


# In[364]:


df_ratings = df_ratings.sort_values('movieId')


# - Le nombre de notes par année

# In[365]:


plt.figure(figsize= (30, 6))

sns.countplot(data = df_ratings, x=df_ratings.year,
              hue= df_ratings.rating)

plt.title('Nombre de note par année', size= 25)
plt.xlabel("Année")
plt.xticks(size= 16)

plt.show()


# - Le nombre de notes par mois

# In[366]:


plt.figure(figsize= (30, 6))

sns.countplot(data = df_ratings, x=df_ratings.month,
              hue= df_ratings.rating)

plt.title('Nombre de note par mois', size= 25)
plt.xlabel("Mois")
plt.xticks(size= 16)
plt.show()


# - les 10 meilleurs utilisateurs en fonction du nombre de notes. 

# In[367]:


fig, axes = plt.subplots(1,2)

ax = df_ratings['userId'].value_counts()[:10].plot.bar(ax= axes[0], figsize=(30,6))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height()/2),
                va="center", ha="center",
                size = 11, weight = 'bold', rotation = 0, color = 'black',
                bbox=dict(boxstyle="round,pad=0.5", fc='white'))

ax.set_xlabel('ID utilisateur', fontsize=14)
ax.set_ylabel('Nombre de notes', fontsize=14)
axes[0].set_title('Top 10 des utilisateurs en fonction du nombre de notes', size= 25);

df = df_ratings['userId'].value_counts()[:10].reset_index().rename(columns={'userId': 'count', 'index': 'userId'})
pie = plt.pie(
    df['count'], 
    wedgeprops=dict(width=0.6, alpha=0.9),
    autopct='%1.0f%%',
    pctdistance=1.12, 
    textprops={
        'fontsize': 12, 
        'fontweight': 'bold'});

axes[1].legend(df['userId'], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), borderpad=1, fontsize=12)
axes[1].set_title('Le Ratio entre les tops utilisateurs', size= 25);

plt.tight_layout(rect=[0,0,0.9,1])
plt.show()


# - Nombre d'utilisateurs uniques par an

# In[368]:


df_temp = df_ratings[['year', 'userId']].groupby(['year']).nunique().reset_index()

plt.figure(figsize= (30, 6))
ax = sns.barplot(x = 'year', y = 'userId', data = df_temp);

for i in ax.patches:    
    ax.text(x = i.get_x() + i.get_width()/2, y = i.get_height()/2,
            s = int(i.get_height()), 
            ha = 'center', size = 14, weight = 'bold', rotation = 0, color = 'black')
ax.set_xlabel('Année', fontsize=14)
ax.set_ylabel('total', fontsize=14)
plt.title('Nombre utilisateurs uniques chaque année' , fontsize=16);
plt.show()


# - Movies

# In[369]:


df_movies.head()


# In[370]:


print(f"Shape : {df_movies.shape} \nSize  :  {df_movies.size}")


# In[371]:


df_movies.nunique()


# In[372]:


def preprocess_movie_csv(movie_df):
    ''' 
   Extraire des années en trouvant 4 nombres consécutifs. (d+ ne fonctionne pas correctement si le titre contient des chiffres) 
   Trouver tous les genres uniques
   '''
    df = movie_df.copy()
    df['release_year'] = df.title.str.extract("\((\d{4})\)", expand=True).astype(str)
    df['title'] = df.title.str[:-7]
    df = df.join(df['genres'].str.get_dummies().astype(bool))
    df.drop('genres', inplace=True, axis=1)
    return df


# In[373]:


processed_movie_df = preprocess_movie_csv(df_movies)
processed_movie_df.rename(columns={"(no genres listed)": "No Genre"}, inplace=True)
processed_movie_df


# - Les titres des films

# In[374]:


df_movie_titles = processed_movie_df[['movieId','title']]
df_movie_titles


# - Le nombre de films sortis par année

# In[375]:


years = processed_movie_df.release_year.unique()
years.sort()
print(years)


# In[376]:


df_temp = processed_movie_df.groupby(['release_year'])['title'].count()
df_temp.plot.bar(x='year', y='movies', title='Nombre de films diffusés par an', figsize=(30, 5))


# - Le nombre de films sortis au cours des 10 dernières années

# In[377]:


df_temp = processed_movie_df.groupby(['release_year'])['title'].count().tail(10)
plt.figure(figsize=(25,8))
ax= sns.barplot(x=df_temp.index,y=df_temp.values, data=processed_movie_df,palette='rainbow')

for i in ax.patches:    
    ax.text(x = i.get_x() + i.get_width()/2, y = i.get_height()/2,
            s = int(i.get_height()), 
            ha = 'center', va='center', size = 14, weight = 'bold', rotation = 0, color = 'white',
            bbox=dict(boxstyle="round,pad=0.5", fc='pink', ec="pink", lw=2))

plt.xlabel('Years',fontsize=14)
plt.xticks(rotation=90)
plt.ylabel('Total films',fontsize=14)
plt.title("Nombre de films sortis au cours des 10 dernières années", fontsize=18)
plt.show()


# - Le nombre de films sortis dans différents genres au cours des 10 dernières années.

# In[378]:


genres_unique = pd.DataFrame(pd.DataFrame(df_movies.genres.str.split('|').tolist()).stack().unique(), columns=['genre'])
genres_unique[-1:]= "No Genre"
genres_unique


# In[379]:


plt.figure(figsize=(30,10)) 
for genre in genres_unique.genre:
    df_temp = processed_movie_df[processed_movie_df[genre]==True][['release_year', 'movieId']]

    #remplir les nan avec la moyenne des années
    df_temp['release_year'] = pd.to_numeric(df_temp['release_year'], errors='coerce')
    df_temp['release_year'].fillna(int(df_temp['release_year'].mean()), inplace=True)
    df_temp['release_year'] =df_temp['release_year'].astype(int)  
    df_temp = df_temp.groupby(['release_year']).count().reset_index().tail(10)
    plt.plot(df_temp['release_year'], df_temp['movieId'], label=genre)
plt.title('Nombre de films sortis chaque année dans différents genres')
plt.legend()
plt.show()


# - Les différents genres de films 

# In[380]:


df_genres = pd.DataFrame(columns=['genre', 'num_movies'])

for genre in genres_unique.genre:
    row = [genre, processed_movie_df[processed_movie_df[genre]==True][['movieId']].count()]
    df_genres.loc[len(df_genres)] = row
    
df_genres['num_movies'] = df_genres['num_movies'].astype(np.int32)
df_genres = df_genres.sort_values('num_movies', ascending=False).set_index('genre')
plot_fig = df_genres['num_movies'].plot(kind='bar', figsize=(15,7))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
plot_fig.set_title('Nombre de films par genre')
plot_fig.set_xlabel('Genre');


# - Le pourcentage par genres

# In[381]:


df_genres = pd.DataFrame(columns=['genre', 'num_movies'])

for genre in genres_unique.genre:
    row = [genre, processed_movie_df[processed_movie_df[genre]==True][['movieId']].count()]
    df_genres.loc[len(df_genres)] = row
plt.figure(figsize=(30,10)) 
plt.axis('equal');
plt.pie(df_genres['num_movies'], labels=df_genres['genre'], autopct='%.1f%%', labeldistance=1.04, rotatelabels=True)
plt.title("Pourcentage de genres par films")
plt.show()


# ## 1 - Collaborative Filtering (CF)
# Le filtrage collaboratif est le processus de filtrage d'informations ou de modèles à l'aide de techniques impliquant la collaboration entre plusieurs utilisateurs, points de vue et sources de données.
# 
# Il existe deux approches de la CF -->
# 
# 1) **CF basée sur la mémoire** - Il s'agit d'une approche qui trouve des similitudes entre les utilisateurs ou entre les articles afin de recommander des articles similaires. Les exemples incluent la FC basée sur le voisinage et les recommandations top-N basées sur l'article ou l'utilisateur.
# 
# 2) **CF basée sur un modèle** - Dans cette approche, nous utilisons différents algorithmes d'exploration de données et d'apprentissage automatique pour prédire l'évaluation par les utilisateurs d'articles non évalués. Les exemples incluent la décomposition en valeurs singulières (SVD), l'analyse en composantes principales (PCA), etc.

# Nous créons d’abord la colonne Nouvelle Année dans le dataframe du film

# In[382]:


df_ratings = pd.read_csv('movies/rating.csv',nrows=100001)
df_movies = pd.read_csv('movies/movie.csv')


# In[383]:


df_movies.head()


# In[384]:


df_movies['year'] = df_movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
df_movies['year'] = df_movies.year.str.extract('(\d\d\d\d)',expand=False)


# In[385]:


#Suppression des années de la colonne "titre".
df_movies['title'] = df_movies.title.str.replace('(\(\d\d\d\d\))', '')
#L'application de la fonction "strip" permet de se débarrasser de tout caractère d'espace
df_movies['title'] = df_movies['title'].apply(lambda x: x.strip())


# In[386]:


df_movies.head()


# Le filtrage collaboratif ne fait pas de recommandations basées sur les caractéristiques du film. La recommandation est basée sur les appréciations ou les évaluations des voisins ou d'autres utilisateurs. Nous allons donc supprimer la colonne du genre, puisqu'elle n'est pas utile.

# In[387]:


df_movies.drop(columns=['genres'], inplace=True)


# In[388]:


df_movies.head()


# En ce qui concerne le cadre de données des évaluations, nous avons la colonne movieId qui est commune avec le cadre de données des films. Chaque utilisateur a donné plusieurs évaluations pour différents films. La colonne Timestamp n'est pas nécessaire pour le système de recommandation. Nous pouvons donc la supprimer.

# In[389]:


df_ratings.head()


# In[390]:


df_ratings.drop(columns=['timestamp'], inplace = True)


# In[391]:


df_ratings.head()


# ## 1.1 ) User based Collaborative Filtering¶
# 

# Cette technique s'appelle le filtrage collaboratif, également connu sous le nom de filtrage utilisateur-utilisateur. Comme son nom l'indique, cette technique fait appel à d'autres utilisateurs pour recommander des articles à l'utilisateur. Elle tente de trouver des utilisateurs qui ont des préférences et des opinions similaires à celles de l'utilisateur, puis recommande à ce dernier des articles qu'ils ont aimés. Il existe plusieurs méthodes pour trouver des utilisateurs similaires (certaines font même appel à l'apprentissage automatique), et celle que nous utiliserons ici sera basée sur la fonction de corrélation de Pearson.
# 
# Le processus de création d'un système de recommandation basé sur l'utilisateur est le suivant :
# 
# - Sélectionner un utilisateur et les films qu'il a regardés.
# - Sur la base de son évaluation des films, trouver les X meilleurs voisins.
# - Obtenir la liste des films regardés par l'utilisateur pour chaque voisin.
# - Calculer un score de similarité à l'aide d'une formule
# - Recommander les éléments ayant le score le plus élevé

# In[392]:


user = [
            {'title':'Breakfast Club, The', 'rating':4},
            {'title':'Toy Story', 'rating':2.5},
            {'title':'Jumanji', 'rating':3},
            {'title':"Pulp Fiction", 'rating':4.5},
            {'title':'Akira', 'rating':5}
         ] 
inputMovie = pd.DataFrame(user)
inputMovie


# Nous devons maintenant ajouter la colonne movieId du dataframe movie dans le dataframe inputMovie. 
# 
# Nous filtrons d'abord les lignes qui contiennent le titre des films en entrée, puis nous fusionnons ce sous-ensemble avec le cadre de données en entrée. Nous supprimons également les colonnes inutiles pour l'entrée afin d'économiser de l'espace mémoire.

# In[393]:


#Filtrer les films par titres
Id = df_movies[df_movies['title'].isin(inputMovie['title'].tolist())]
#Puis nous les fusionnons afin d'obtenir l'identifiant du film. Il s'agit d'une fusion implicite par titre.
inputMovie = pd.merge(Id, inputMovie)
#Suppression des informations que nous n'utiliserons pas dans le cadre de données d'entrée
inputMovie = inputMovie.drop('year', 1)
inputMovie


# Trouvons les utilisateurs qui ont vu les mêmes films à partir de la base de données d'évaluation Avec les identifiants des films en entrée, nous pouvons maintenant obtenir le sous-ensemble d'utilisateurs qui ont regardé et critiqué les films en entrée.

# In[394]:


#Filtrer les utilisateurs qui ont regardé des films que l'entrée a regardés et les stocker
users = df_ratings[df_ratings['movieId'].isin(inputMovie['movieId'].tolist())]
users.head()


# In[395]:


users.shape


# In[396]:


#Groupby crée plusieurs sous dataframes de données qui ont toutes la même valeur dans la colonne spécifiée en tant que paramètre.
userSubsetGroup = users.groupby(['userId'])


# In[397]:


#un exemple de groupe en obtenant tous les utilisateurs d'un uderId particulier
userSubsetGroup.get_group(500)


# In[398]:


#trions de manière à ce que les utilisateurs ayant le film le plus en commun avec l'entrée aient la priorité.
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


# In[399]:


userSubsetGroup[0:3]


# Similitude des utilisateurs avec l'utilisateur d'entrée. Ensuite, nous allons comparer tous les utilisateurs à l'utilisateur spécifié et trouver celui qui est le plus similaire. Nous allons déterminer le degré de similitude de chaque utilisateur avec l'entrée à l'aide du coefficient de corrélation de Pearson. Il est utilisé pour mesurer la force d'une association linéaire entre deux variables. La formule pour trouver ce coefficient entre les ensembles X et Y avec N valeurs est présentée dans l'image ci-dessous.
# 
# Pourquoi la corrélation de Pearson ?
# 
# La corrélation de Pearson est invariante à l'échelle, c'est-à-dire en multipliant tous les éléments par une constante non nulle ou en ajoutant une constante quelconque à tous les éléments. Par exemple, si vous avez deux vecteurs X et Y, alors, pearson(X, Y) == pearson(X, 2 * Y + 3). Il s'agit d'une propriété très importante dans les systèmes de recommandation car, par exemple, deux utilisateurs pourraient évaluer deux séries d'articles de manière totalement différente en termes de taux absolus, mais il s'agirait d'utilisateurs similaires (c'est-à-dire ayant des idées similaires) avec des taux similaires dans différentes échelles.![image.png](attachment:image.png)
# 
# Les valeurs données par la formule varient de r = -1 à r = 1, où 1 forme une corrélation directe entre les deux entités (il s'agit d'une corrélation positive parfaite) et -1 forme une corrélation négative parfaite.
# 
# Dans notre cas, un 1 signifie que les deux utilisateurs ont des goûts similaires, tandis qu'un -1 signifie le contraire.

# In[400]:


userSubsetGroup = userSubsetGroup[0:100]


# In[401]:


from numpy import sqrt 

#Stocker la corrélation de Pearson dans un dictionnaire, où la clé est l'identifiant de l'utilisateur et 
#a valeur est le coefficient.
pearsonCorDict = {}

#Pour chaque groupe d'utilisateurs de notre sous-ensemble
for name, group in userSubsetGroup:
    #Commençons par trier les données d'entrée et le groupe d'utilisateurs actuel afin de ne pas mélanger les valeurs par la suite.
    group = group.sort_values(by='movieId')
    inputMovie = inputMovie.sort_values(by='movieId')
    #Obtenir la valeur N de la formule
    n = len(group)
    #Obtenez les notes des critiques des films qu'ils ont en commun
    temp = inputMovie[inputMovie['movieId'].isin(group['movieId'].tolist())]
    #Puis les stocker dans une variable tampon temporaire sous forme de liste afin de faciliter les calculs ultérieurs.
    tempRatingList = temp['rating'].tolist()
    #afficher les revues des groupes d'utilisateurs actuels sous forme de liste
    tempGroupList = group['rating'].tolist()
    #Calculons maintenant la corrélation de Pearson entre deux utilisateurs, appelés x et y.
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)
    
    #Si le dénominateur est différent de zéro, on divise, sinon on corrige par 0..
    if Sxx != 0 and Syy != 0:
        pearsonCorDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorDict[name] = 0


# In[402]:


pearsonCorDict.items()


# In[403]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()


# In[404]:


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


# Note des utilisateurs sélectionnés par rapport à tous les films, nous allons procéder en prenant la moyenne pondérée des notations des films en utilisant la corrélation de Pearson comme poids. Mais pour ce faire, nous devons d'abord obtenir les films regardés par les utilisateurs dans notre pearsonDF à partir du dataframe des évaluations, puis stocker leur corrélation dans une nouvelle colonne appelée _similarityIndex". Nous y parvenons ci-dessous en fusionnant ces deux tables.

# In[53]:


topUsersRating=topUsers.merge(df_ratings, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# multiplication de la note du film par son poids (l'indice de similarité), puis additionner les nouvelles notes et les diviser par la somme des poids.
# 
# Nous pouvons facilement faire cela en multipliant simplement deux colonnes, puis en regroupant le dataframe par movieId et en divisant ensuite deux colonnes :

# In[54]:


#Multiplie la similarité par les notes attribuées par l'utilisateur
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()


# In[55]:


#Applique une somme aux topUsers après les avoir regroupés par userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()


# In[56]:


#Crée un cadre de données vide
recommendation_df = pd.DataFrame()
#Nous prenons maintenant la moyenne pondérée
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()


# In[57]:


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)


# In[58]:


df_movies.loc[df_movies['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]


# ## 2 - Model Based Collaborative Filtering
# Nous utiliserons ici des méthodes de réduction de la dimensionnalité pour améliorer la robustesse et la précision de la FC basée sur la mémoire. Fondamentalement, nous compressons la matrice des éléments de l'utilisateur en une matrice de faible dimension. Nous utilisons des techniques telles que le SVD, qui est une méthode de factorisation à faible rang, l'ACP, qui est utilisée pour la réduction de la dimensionnalité, etc.
# 
# Les méthodes fondées sur un modèle sont basées sur la factorisation des matrices et sont plus efficaces pour gérer la rareté.
# 
# Nous utiliserons une bibliothèque "Surprise" pour implémenter SVD, KNN et NMF.
# 
# bibliothèque Surprise contient presque tous les algorithmes requis pour les systèmes de recommandation basés sur des modèles.
# 
# Pour charger un jeu de données à partir d'un cadre de données pandas, vous aurez besoin de la méthode load_from_df(). Vous aurez également besoin d'un objet Reader, mais seul le paramètre rating_scale doit être spécifié.
# 
# La classe Reader est utilisée pour analyser un fichier contenant des classements.
# 
# ![image.png](attachment:image.png)![image-2.png](attachment:image-2.png)

# In[433]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.models import Model


# In[406]:


df_ratings = pd.read_csv('movies/rating.csv')
df_movies = pd.read_csv('movies/movie.csv')


# In[407]:


df = df_ratings[['userId', 'movieId', 'rating']]


# In[408]:


df.shape


# ## 2.1 - Filtrage collaboratif avec factorisation matricielle
# 
# Le filtrage collaboratif a un problème avec le démarrage à froid de l'utilisateur, quel modèle pourrait ne pas être en mesure de fournir une liste de recommandations décente à ceux qui ont donné un faible nombre d'évaluations, donc le modèle manque d'informations sur les préférences de l'utilisateur de démarrage à froid calculer une table masive).

# In[409]:


n_interacted = 2000
user_movie_data_temp = pd.pivot_table(df, index = ['userId'], values='movieId', aggfunc='count')
user_movie_data_temp[user_movie_data_temp.movieId>=n_interacted]
selected_user_ids = user_movie_data_temp[user_movie_data_temp.movieId>=n_interacted].index
print('nombre utilisateur: ', str(len(selected_user_ids)))

n_rated = 1000
get_rated_movie = pd.pivot_table(df, index=['movieId'], values='userId', aggfunc='count')
get_rated_movie[get_rated_movie.userId>=n_rated]
selected_movie_ids = get_rated_movie[get_rated_movie.userId>=n_rated].index

print('nombre id film: ', str(len(selected_movie_ids)))

filtered_rating_data = df[(df['userId'].isin(selected_user_ids)) &(df['movieId'].isin(selected_movie_ids))]
filtered_rating_data['movieId'] = filtered_rating_data['movieId'].apply(lambda x: 'm_'+str(x))

print('taille des données .  : ',str(filtered_rating_data.shape))


# In[410]:


filtered_rating_data = filtered_rating_data[['userId','movieId','rating']]
filtered_rating_data


# - Comme d'habitude, nous diviserons les données en deux groupes : un ensemble d'apprentissage et un ensemble de test — en utilisant la méthode train_test_split.

# In[414]:


from sklearn.model_selection import train_test_split, cross_validate


# In[415]:


train_df, test_df =  train_test_split(filtered_rating_data, 
                                   stratify = filtered_rating_data['userId'],
                                   test_size = 0.2,
                                   random_state = 42)

print('train_df size:{}'.format(len(train_df)))
print('test_df size:{}'.format(len(test_df)))


# - Bien que les informations dont nous avons besoin soient présentes, elles ne sont pas présentées de manière à ce que les humains puissent les comprendre. Cependant, nous avons créé un tableau qui présente les mêmes données dans un format plus facile à comprendre pour les humains.

# In[416]:


user_movie_data_train = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0.0)
user_movie_data_train


# # 1 : Factorisation matricielle en Python à partir de zéro
# Implémentation de la factorisation matricielle avec la descente de gradient. 
# 
# La matrix_factorizationfonction renvoie 2 matrices : nP (matrice utilisateur) et nQ (matrice film)

# In[417]:


def matrix_factorization(R, K, steps=5, alpha=0.002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter
    
    '''
 
    P = np.random.rand(len(R),K)
    Q = np.random.rand(len(R[0]),K)
    Q = Q.T

    for step in range(steps):
        print('Processing epoch {}'.format(step))
        
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T


# - Ensuite, ajustons l'ensemble de données d'apprentissage au modèle et ici, nous fixons n_factor K = 5.
# 
# - Ensuite, les prédictions peuvent être calculées en multipliant nP et la transposition de nQ à l'aide de la méthode du produit scalaire, comme illustré dans l'extrait de code ci-dessous.

# In[419]:


R = np.array(user_movie_data_train)
nP, nQ = matrix_factorization(R, K=5)


# In[429]:


model = matrix_factorization(R, K=5)


# In[420]:


pred_R = np.dot(nP, nQ.T)

# Transformer la prédiction en matrice reconstruite en une trame de données Pandas au format croisé
user_movie_pred = pd.DataFrame(pred_R, columns=user_movie_data_train.columns, index=list(user_movie_data_train.index))
print(user_movie_pred.shape)
user_movie_pred.head(10)


# In[421]:


# User Matrix

Pu = pd.DataFrame(nP, index=list(user_movie_data_train.index))

# Movie Matrix

Qu = pd.DataFrame(nQ, index=user_movie_data_train.columns)


# In[422]:


def predict_rating(data):
    try:
        pred_rating = np.dot(Pu.loc[data.userId], Qu.loc[data.movieId].T)
    except Exception as e:
        pred_rating = np.nan
        print('Unknown user: {} or movieId: {}'.format(data.userId,data.movieId))
    return pred_rating


# In[423]:


test_df['pred_rating'] = test_df.apply(predict_rating, axis=1)


# In[424]:


test_df.head(10)


# - Évaluation des performances de prédiction.
# 
# Bien qu'il existe diverses mesures d'évaluation pour les systèmes de recommandation, telles que Precision, Recall, MAP, et la liste continue. nous utiliserons une métrique de précision de base, à savoir RMSE.

# In[425]:


rmse_test = mean_squared_error(test_df['rating'], test_df['pred_rating'], squared=False)
rmse_test


# In[435]:


mae_test = mean_absolute_error(test_df['rating'], test_df['pred_rating'])
mae_test


# # 2 - Factorisation matricielle avec package surprise

# In[436]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD, NMF, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split


# In[437]:


filtered_rating_data


# In[444]:


import surprise as sp


# In[447]:


reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(filtered_rating_data[['userId','movieId','rating']], reader)

benchmark = []
# Itération sur tous les algorithmes
for algorithm in [sp.SVD(n_epochs = 10, lr_all = 0.005), sp.SVDpp(), sp.SlopeOne(), sp.NMF(), sp.NormalPredictor(), sp.KNNBaseline(), sp.KNNBasic(n_neighbours=10, mink=6, min_support=2,sim_options=sim_options, verbose=False, random_state=1), sp.KNNWithMeans(), sp.KNNWithZScore(), sp.BaselineOnly(), sp.CoClustering()]:
    # Performance cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    
    # Obtenir les résultats et ajouter le nom de l'algorithme
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    
    # Stockage des données
    benchmark.append(tmp)


# In[448]:


# Pour créer des graphiques interactifs
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# In[449]:


# Stockage resultats
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse', ascending=False)

# obtenir les données
data = surprise_results[['test_rmse', 'test_mae']]
grid = data.values

# Création d'axe d'étiquettes 
x_axis = [label.split('_')[1].upper() for label in data.columns.tolist()]
y_axis = data.index.tolist()

x_label = 'Mesure des performances'
y_label = 'Algorithm'


# Obtenir des annotations et du texte en surimpression
hovertexts = []
annotations = []
for i, y_value in enumerate(y_axis):
    row = []
    for j, x_value in enumerate(x_axis):
        annotation = grid[i, j]
        row.append('Error: {:.3f}<br>{}: {}<br>{}: {}<br>Fit Time: {:.3f}s<br>Test Time: {:.3f}s'.format(annotation, y_label, y_value ,x_label, x_value, surprise_results.loc[y_value]['fit_time'], surprise_results.loc[y_value]['test_time']))
        annotations.append(dict(x=x_value, y=y_value, text='{:.3f}'.format(annotation), ax=0, ay=0, font=dict(color='#000000')))
    hovertexts.append(row)

# Graphique
trace = go.Heatmap(x = x_axis,
                   y = y_axis,
                   z = data.values,
                   text = hovertexts,
                   hoverinfo = 'text',
                   colorscale = 'Picnic',
                   colorbar = dict(title = 'Error'))

# Create layout
layout = go.Layout(title = 'Comparaison croisée des algorithmes de surprise',
                   xaxis = dict(title = x_label),
                   yaxis = dict(title = y_label,
                                tickangle = -40),
                   annotations = annotations)

# plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[450]:


sdvpp = sp.SVDpp()


# In[456]:


reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(filtered_rating_data[['userId','movieId','rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

# Nous utiliserons le célèbre algorithme SVD (celui de la factorisation matricielle)..
svdpp = sp.SVDpp()

# Entraînement de l'algorithme sur la rating et prédisons les notes de l'ensemble de test
svdpp.fit(trainset)
svdpp_pred = sdv.test(testset)

# Then compute RMSE
evaluation_svdpp = cross_validate(svdpp, data, measures=['RMSE','MAE'], cv= 5, verbose=True)


# - Fonction de prédiction des top 

# In[457]:


from collections import defaultdict


# In[458]:


def get_top_n(predictions, n=5):
    # Il s'agit d'abord d'associer les prédictions à chaque utilisateur.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Trier ensuite les prédictions pour chaque utilisateur et extraire les k prédictions les plus élevées.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# **- Top 10 des prédictions des notes par utilisateurs**

# In[459]:


top_n = get_top_n(svd_pred, 10)

top_nn = pd.DataFrame(top_n)
top_nn.head(10)


# In[ ]:





# In[465]:


from surprise import AlgoBase

class HybridAlgorithm(AlgoBase):

    def __init__(self, models, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.models = models
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        for model in self.models:
            model.fit(trainset)
                
        return self

    def estimate(self, user_id, item_id):
        
        scores_sum = 0
        weights_sum = 0
        
        for i in range(len(self.models)):
            scores_sum += self.models[i].estimate(user_id, item_id) * self.weights[i] # 3*1/4+4*3/4 laga ra
            weights_sum += self.weights[i] # always becomes one
            
        return scores_sum / weights_sum


# In[461]:


# Construire un modèle hybride en combinant les trois modèles
hybrid_model = HybridAlgorithm([svd, svdpp], [0.6, 0.5])

# Entraîner le modèle hybride
hybrid_model.fit(trainset)
hybrid_predictions = hybrid_model.test(testset)

evaluation_hybrid = cross_validate(hybrid_model, data, measures=['RMSE','MAE'], cv= 5, verbose=True)


# In[490]:


testset


# In[470]:


reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(filtered_rating_data[['userId','movieId','rating']], reader)

benchmark = []
# Itération sur tous les algorithmes
for algorithm in [HybridAlgorithm([svd, svdpp], [0.6, 0.5]), sp.SVD(n_epochs = 10, lr_all = 0.005), sp.SVDpp()]:
    # Performance cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    
    # Obtenir les résultats et ajouter le nom de l'algorithme
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    
    # Stockage des données
    benchmark.append(tmp)


# In[472]:


# Stockage resultats
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse', ascending=False)

# obtenir les données
data = surprise_results[['test_rmse', 'test_mae']]
grid = data.values

# Création d'axe d'étiquettes 
x_axis = [label.split('_')[1].upper() for label in data.columns.tolist()]
y_axis = data.index.tolist()

x_label = 'Mesure des performances'
y_label = 'Algorithm'


# Obtenir des annotations et du texte en surimpression
hovertexts = []
annotations = []
for i, y_value in enumerate(y_axis):
    row = []
    for j, x_value in enumerate(x_axis):
        annotation = grid[i, j]
        row.append('Error: {:.3f}<br>{}: {}<br>{}: {}<br>Fit Time: {:.3f}s<br>Test Time: {:.3f}s'.format(annotation, y_label, y_value ,x_label, x_value, surprise_results.loc[y_value]['fit_time'], surprise_results.loc[y_value]['test_time']))
        annotations.append(dict(x=x_value, y=y_value, text='{:.3f}'.format(annotation), ax=0, ay=0, font=dict(color='#000000')))
    hovertexts.append(row)

# Graphique

trace = go.Heatmap(x = x_axis,
                   y = y_axis,
                   z = data.values,
                   text = hovertexts,
                   hoverinfo = 'text',
                   colorscale = 'Picnic',
                   colorbar = dict(title = 'Error'))

# Create layout
layout = go.Layout(title = 'Comparaison croisée des algorithmes de surprise',
                   xaxis = dict(title = x_label),
                   yaxis = dict(title = y_label,
                                tickangle = -40),
                   annotations = annotations)

# plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# **- Top 10 des prédictions des notes par utilisateurs**

# In[473]:


top_n = get_top_n(hybrid_predictions, 10)

top_nn = pd.DataFrame(top_n)
top_nn.head(10)


# - Enregistrons chaque modèle individuel utilisé dans notre modèle hybride :

# In[495]:


import joblib

# Enregistrez les modèles individuels
svd_model_filename = "svd_model.pkl"
joblib.dump(svd, svd_model_filename)

svdpp_model_filename = "svdpp_model.pkl"
joblib.dump(svdpp, svdpp_model_filename)


# - Enregistrons les poids ou les informations nécessaires pour combiner ces modèles dans votre modèle hybride :

# In[496]:


# Enregistrez les informations nécessaires pour combiner les modèles
hybrid_info = {
    'weights': hybrid_model.weights,
    'models_filenames': [svd_model_filename, svdpp_model_filename]
}
joblib.dump(hybrid_info, "hybrid_info.pkl")


# Pour charger le modèle en production, nous allons charger d'abord les modèles individuels, puis les combinez avec les informations enregistrées pour obtenir le modèle hybride complet :

# In[497]:


# Chargez les modèles individuels
loaded_svd = joblib.load(svd_model_filename)
loaded_svdpp = joblib.load(svdpp_model_filename)


# Chargez les informations pour combiner les modèles
loaded_hybrid_info = joblib.load("hybrid_info.pkl")

# Créez le modèle hybride en utilisant les modèles individuels et les informations
loaded_hybrid_model = HybridAlgorithm(
    models=[loaded_svd, loaded_svdpp],
    weights=loaded_hybrid_info['weights']
)


# In[512]:


# Remplacez par l'ID de l'utilisateur pour lequel vous voulez faire une recommandation
new_user_id = 138493 
# Remplacez par l'ID du film que vous voulez recommander à l'utilisateur
new_movie_id = 689


# In[513]:


prediction = loaded_hybrid_model.estimate(new_user_id, new_movie_id)


# In[516]:


prediction


# In[506]:


df_ratings


# In[ ]:


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




