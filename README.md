
# XNAP-Project 5: Music Genre Classification
Write here a short summary about your project. The text must include a short introduction and the targeted goals</br>
Aquest projecte busca classificar diferents audios de musica en el genere musical al que pertanyen.

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.
Arrel del starting point original tenim tres elements de moment no canivats:</br>
-audio</br>
-weights</br>
-predict_example.py</br>
Quant executem el arxiu .py carrega un model ja fet de la carpeta weights i l'utilitza per fer una predicció de un dels arxius de la carpeta audio.</br>
Els arxius GenreFeatureData son els dataloaders, contenen el codi per un objecte que carrega les dades de la carpeta gtzan. La diferencia entre els dos arxius es que la versio "m" fa servir espectrograma per extreure les caracteristiques.</br>
La resta dels codis son les diferentes proves que anem fent.</br>
Cada arxiu conte el model i l'entrenament tot i que en un futur es separara ja que els entrenaments tenen la mateixa estructura.
L'arxiu requeriments.txt esta desactualitzat.</br>

# PyTorch LSTM Classificació de Gènere - README

Aquest repositori conté una implementació en PyTorch d'un model LSTM de 2 capes per a la classificació de gènere de música. El model LSTM s'entrena amb diverses característiques d'àudio com ara el centroid espectral, el contrast, el cromagrama i els coeficients cepstral de Mel (MFCC).

## Taula de continguts

- [Introducció](#introducció)
- [Dependències](#dependències)
- [Conjunt de dades](#conjunt-de-dades)
- [Arquitectura del model](#arquitectura-del-model)
- [Entrenament](#entrenament)
- [Avaluació](#avaluació)
- [Resultats](#resultats)
- [Ús](#ús)
- [Llicència](#llicència)

## Introducció

L'objectiu d'aquest projecte és classificar àudio musical en diferents gèneres utilitzant una aproximació de deep learning. El model LSTM s'entrena amb un conjunt de dades amb característiques d'àudio extretes utilitzant la llibreria librosa. Les característiques inclouen el centroid espectral, el contrast, el cromagrama i els coeficients cepstral de Mel, resultant en un total de 33 dimensions d'entrada.

L'objectiu d'aquest projecte es probar diferents models i mètodes per tal d'arribar a trobar diferents alternatives al projecte inicial.

## Dependències

Les següents dependències són necessàries per executar el codi:

- Python 3.x
- PyTorch
- NumPy
- librosa
- matplotlib

Podeu instal·lar les dependències requerides executant la següent comanda:

```bash
pip install torch numpy librosa matplotlib
```

## Conjunt de dades

El conjunt de dades utilitzat per a l'entrenament i avaluació consisteix en mostres d'àudio de diferents gèneres. Els fitxers d'àudio es pre-processen per extreure les característiques desitjades utilitzant la classe GenreFeatureData, que fa servir la llibreria librosa. Les dades pre-processades es guarden en format NumPy per a una càrrega eficient durant l'entrenament i l'avaluació.

El conjunt de dades es divideix en tres subconjunts: entrenament, validació i prova. El conjunt d'entrenament s'utilitza per entrenar el model, el conjunt de validació es fa servir per a monitorar el rendiment del model durant l'entrenament i el conjunt de prova es fa servir per avaluar el model final.

## Arquitectura del model

El model LSTM consisteix en dues capes LSTM seguides d'una capa de sortida lineal. La dimensió d'entrada de les capes LSTM és 33, que correspon al nombre de característiques d'àudio. La dimensió oculta s'estableix a 64 per a cada capa LSTM. El model s'entrena utilitzant l'algorisme de propagació endavant (forward propagation) i retropropagació (backpropagation) per ajustar els pesos dels paràmetres del model.

## Entrenament

Per entrenar el model, seguiu els següents passos:

1. Assegureu-vos d'haver instal·lat totes les dependències esmentades a la secció [Dependències](#dependències).
2. Baixeu el conjunt de dades o prepareu el vostre propi conjunt de dades amb mostres d'àudio i les seves etiquetes de gènere corresponents.
3. Pre-processeu les mostres d'àudio i extreu les característiques d'àudio utilitzant la classe GenreFeatureData. Desa les dades pre-processades en format NumPy.
4. Modifiqueu la funció `main()` en el codi per proporcionar la ruta de les dades pre-processades i ajustar altres hiperparàmetres si cal.
5. Executeu el codi utilitzant la següent comanda:

```bash
python lstm_genre_classification.py
```

6. Seguiu el progrés de l'entrenament a través de les mètriques de pèrdua (loss) i precisió (accuracy) que es mostren.
7. Després de l'entrenament, avalueu el rendiment del model en el conjunt de proves observant la mètrica de precisió (accuracy).
8. Visualitzeu els gràfics de pèrdua (loss) i precisió (accuracy) de l'entrenament per analitzar el rendiment del model durant l'entrenament.

## Avaluació

Per avaluar el model entrenat en el conjunt de proves, executeu el codi com es descriu a la secció [Entrenament](#entrenament). El codi calcularà la precisió (accuracy) del model en el conjunt de proves i mostrarà el resultat.

## Resultats

Els resultats del model són la precisió (accuracy) obtinguda en el conjunt de proves. Aquesta mètrica indica la capacitat del model per classificar correctament les mostres d'àudio en els gèneres corresponents. Els resultats també es poden visualitzar en forma de gràfics de pèrdua (loss) i precisió (accuracy) durant l'entrenament.

## Ús

Podeu utilitzar aquest codi com a punt de partida per a la vostra pròpia classificació de gènere d'àudio utilitzant PyTorch. Feu les modificacions necessàries per adaptar-lo al vostre conjunt de dades i requeriments específics.

## Agraïments

- Els canvis fets estàn inspirats amb les pràctiques i exemples fets durant l'assignatura.
- El conjunt de dades utilitzat per a l'entrenament i avaluació és una adaptació d'un conjunt de dades disponible públicament. Moltes gràcies als creadors del conjunt de dades original.



## Contributors
Write here the name and UAB mail of the group members:
</br>1568073@uab.cat
</br>1525973@uab.cat
</br>1605189@uab.cat

## Starting Point
Aquest projecte s'ha desenvolupat a partir del seguent setatring point:</br>
https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification</br>
El seu readme es pot trobar com el arxiu original_readme.md 

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades, 
UAB, 2023
