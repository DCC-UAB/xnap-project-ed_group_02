
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

Aquest repositori conté una implementació en PyTorch d'un model LSTM de 2 capes per a la classificació de gènere de música. El model LSTM s'entrena amb diverses característiques d'àudio com ara el centroid espectral, el contrast, el cromagrama i els coeficients cepstral de Mel (MFCC).</br>

## Taula de continguts

- [Introducció](#introducció)</br>
- [Dependències](#dependències)</br>
- [Conjunt de dades](#conjunt-de-dades)</br>
- [Models](#models)</br>
- [Entrenament](#entrenament)</br>
- [Avaluació](#avaluació)</br>
- [Resultats](#resultats)</br>
- [Ús](#ús)</br>
- [Llicència](#llicència)</br>

## Introducció

L'objectiu d'aquest projecte és classificar àudio musical en diferents gèneres utilitzant una aproximació de deep learning. El model LSTM s'entrena amb un conjunt de dades amb característiques d'àudio extretes utilitzant la llibreria librosa. Les característiques inclouen el centroid espectral, el contrast, el cromagrama i els coeficients cepstral de Mel, resultant en un total de 33 dimensions d'entrada.</br>

La idea d'aquest exercici pràctic es probar diferents models i mètodes per tal d'arribar a trobar diferents alternatives al projecte inicial.</br>

## Dependències

Les següents dependències són necessàries per executar el codi:</br>

- Python 3.x</br>
- PyTorch</br>
- NumPy</br>
- librosa</br>
- matplotlib</br>

Podeu instal·lar les dependències requerides executant la següent comanda:</br>

```bash
pip install torch numpy librosa matplotlib
```

## Conjunt de dades

El conjunt de dades utilitzat per a l'entrenament i avaluació consisteix en mostres d'àudio de diferents gèneres. Els fitxers d'àudio es pre-processen per extreure les característiques desitjades utilitzant la classe GenreFeatureData, que fa servir la llibreria librosa. Les dades pre-processades es guarden en format NumPy per a una càrrega eficient durant l'entrenament i l'avaluació.</br>

El conjunt de dades es divideix en tres subconjunts: entrenament, validació i prova. El conjunt d'entrenament s'utilitza per entrenar el model, el conjunt de validació es fa servir per a monitorar el rendiment del model durant l'entrenament i el conjunt de prova es fa servir per avaluar el model final.</br>

## Models

***Starting Point: lstm_pytorch_original***</br>

Aquest és un codi en Python que implementa un model LSTM (Long Short-Term Memory) de 2 capes per a la classificació de gènere de música utilitzant PyTorch. L'arquitectura del model consisteix en una pila de capes LSTM seguida d'una capa de sortida.</br>

Aquí hi ha una explicació de les parts importants del codi:</br>

1. La definició de la classe LSTM: Aquesta classe hereta de `nn.Module` de PyTorch i defineix les capes LSTM i la capa de sortida del model. S'inicialitzen els paràmetres com la dimensió d'entrada, la dimensió oculta, el tamany del lot, la dimensió de sortida i el nombre de capes LSTM.</br>

2. La funció `forward`: Aquesta funció defineix el pas endavant (forward pass) del model. Processa les dades d'entrada a través de les capes LSTM i retorna la sortida final del model. Aplica una funció de softmax a la sortida per obtenir una distribució de probabilitat sobre els gèneres musicals.</br>

3. La funció `get_accuracy`: Aquesta funció calcula l'exactitud (accuracy) del model per a una ronda d'entrenament. Compara les prediccions del model amb les etiquetes reals i calcula el percentatge d'instàncies classificades correctament.</br>

4. La funció `main`: Aquesta funció principal gestiona el flux principal del programa. Carrega les dades d'àudio pre-processades, defineix el model LSTM, especifica la funció de pèrdua (loss function) i l'optimitzador per a l'entrenament, i realitza el bucle d'entrenament durant diverses èpoques. També calcula la pèrdua i l'exactitud de la validació en certs intervals d'èpoques.</br>

En resum, aquest codi implementa i entrena un model LSTM per a la classificació de gènere de música utilitzant dades d'àudio pre-processades. Utilitza PyTorch com a framework per a la construcció i l'entrenament del model.</br>

***GPU/Visualització/Entrenament: lstm_pytorch_basic_gpu***</br>

Aquest model és una implementació en PyTorch d'un LSTM de 2 capes per a la classificació de gènere musical utilitzant característiques espectrals, cromàtiques i MFCC com a dades d'entrada. A continuació, es detallen les diferències d'aquest model respecte al model anterior:</br>

1. Arquitectura del model: En aquest model, s'utilitza una capa LSTM de 2 capes amb una dimensió d'entrada de 33. El model anterior només tenia una capa LSTM.</br>

2. Funció de pèrdua: En aquest model, s'utilitza la funció `nn.NLLLoss` (Negative Log Likelihood Loss) com a funció de pèrdua. Això és diferent de la funció `categorical_crossentropy` que s'utilitzava en el model anterior.</br>

3. Inicialització de l'estat ocult: En aquest model, l'estat ocult de la LSTM es reinicialitza a cada iteració de lot mitjançant `model.init_hidden(batch_size)`. Això és diferent del model anterior, on l'estat ocult no es reinicialitza a cada iteració de lot.</br>

4. Ús de GPU: En aquest model, es comprova si una GPU està disponible mitjançant `torch.cuda.is_available()`, i si és així, es mou el model i les dades a la GPU. En el model anterior, no es mostra com es comprova ni com es mouen les dades a la GPU.</br>

5. Bucle d'entrenament: En aquest model, hi ha un bucle d'entrenament que recorre diverses èpoques i minilots per a cada època. El model anterior també tenia un bucle d'entrenament similar, però amb algunes diferències en com es processen els minilots.</br>

6. Visualització de resultats: Aquest model inclou la visualització de la pèrdua i la precisió de validació en cada època utilitzant gràfics de línia. Això és diferent del model anterior, on no es mostra com es visualitzen els resultats.</br>

En resum, aquest model presenta canvis significatius en l'arquitectura de la LSTM, la funció de pèrdua, l'optimitzador i altres aspectes del codi respecte al model anterior. També inclou visualització de resultats.</br>


***Optimització de paràmetres, canvi en la funció loss i Dropout: lstm_pytorch_optim***</br>

En aquest model a diferència del anterior, trobem una capa adicional de dropout. També hem passat d'utilitzar el GenreFeatureData al GenreFeatureData_m (aquest GenreFeature ens retorna les característiques d'audio, però aquestes a diferencia del anterior GenreFeature provenen d'espectogrames de mel que han sigut pasats a una representació logarítimica per posteriorment extreure les seves característiques).</br>

Finalment, s'ha implementat un learning schedule i canviat la funció que utilitzavem per calcular la loss. A continuació els tres schedules utilitzats.</br>

**Lstm_pytorch_optim_ReduceLROnPlateau:** en aquest model s'ha utilitzat el schedule ReduceLROnPlateau que consisteix en monotoritzar el accuracy durant cada època d'entrenament. Si l'accuracy no millora durant un determinat nombre d'èpoques ( 10) consecutives, la taxa d'aprenentatge es redueix en un factor predeterminat (lr*0.1). La idea d'aquest schedule és ajustar el pas d'aprenentatge per aconseguir una millor convergència.</br>

**Lstm_pytorch_optim_CosineAnnealingLR:** el CosineAnnealingLR redueix gradualment la taxa d'aprenentatge durant el procés d'entrenament, seguint una funció de cosinus. Comença amb una taxa d'aprenentatge inicial alta i va disminuint fins a un valor mínim. A continuació, la taxa d'aprenentatge augmenta novament fins al seu valor inicial. Aquest procés es repeteix en cada cicle d'entrenament.<br>

La idea principal d'aquesta tècnica és permetre que el model "explori" diferents regions de l'espai de paràmetres, utilitzant una taxa d'aprenentatge més alta en les primeres etapes de l'entrenament per moure's més ràpidament i una taxa d'aprenentatge més baixa a mesura que es va aproximant a possibles òptims locals. Això pot ajudar a millorar el rendiment i evitar quedar-se estancat en òptims locals subòptims.<br>

Finalment el schedule utilitzat: <br>

**Lstm_pytorch_optim_StepLR:** El StepLR redueix la taxa d'aprenentatge en punts específics durant l'entrenament, anomenats "passos". Aquests passos estan determinats per un valor fix de nombre d'èpoques. Quan el nombre d'èpoques d'entrenament arriba a un d'aquests passos, la taxa d'aprenentatge es redueix segons un factor predefinit.<br>

L'objectiu principal d'utilitzar l'algorisme StepLR és ajustar la taxa d'aprenentatge per millorar el rendiment del model al llarg de l'entrenament. Reduir la taxa d'aprenentatge en moments específics pot ajudar a estabilitzar i refinar el model, especialment en situacions en què el model pot sobreajustar-se o quan s'acosta a un òptim local.<br>

**Lstm_pytorch_optim serà el nostre millor model**<br>

***Model CNN***

S’ha intentat implementar un nou model per la classificació de música segons el gènere on s'utilitzen les imatges MFCC, coeficients per la representació de la senyal d’audio.
MFCC ens permet obtenir informació sobre audios obviant el soroll.

S’ha creat un una llibreta per processar les dades de nou, anomenada *data_process_cnn.ipynb*, on es processa els audios ja existents, guardant en un diccionari dues llistes, una corresponent al gènere i l’altre una array amb els valors de la imatge MFCC.

Per processar els audios s’ha dividit en 5 segments, per tant els audios de 30 segons estaven representats en 5 imatges, augmentant lligerament la mida de les dades.
El diccionari es guarda en un fitxer JSON pel train, test i validation respectivament.

El model CNN que segueix la següent estructura:
3 Capes convulacionals, amb un kernel ( 3x3 ) i un padding de 1, per detectar patrón importants en el MFCC.
3 Capes d’agrupació MaxPooling que de les característiques extretes en les capes anteriors es queda amb el valor més gran per a cada regió..
3 Capes de normalització BatchNorm2d, que tenen com objectiu facilitar l’entrenament i una millor convergencia de la red.
A més s’ha utilitzat la funció d’activació ReLU per evitar linealitat després de cada capa de convolució i finalment unes capes Flatten, Linear, ReLU i DropOut que transformen la sortida de les capes convolucionals anteriors i ho transformen xarxa completament connectada.

Aquest model s'executa desde la llibreta *model_cnn.ipynb*, pero actualment no funciona degut en un error al prepara les dades.

## Entrenament

Per entrenar el model, seguiu els següents passos:</br>

1. Assegureu-vos d'haver instal·lat totes les dependències esmentades a la secció [Dependències](#dependències).</br>
2. Baixeu el conjunt de dades o prepareu el vostre propi conjunt de dades amb mostres d'àudio i les seves etiquetes de gènere corresponents.</br>
3. Pre-processeu les mostres d'àudio i extreu les característiques d'àudio utilitzant la classe GenreFeatureData. Desa les dades pre-processades en format NumPy.</br>
4. Modifiqueu la funció `main()` en el codi per proporcionar la ruta de les dades pre-processades i ajustar altres hiperparàmetres si cal.</br>
5. Executeu el codi utilitzant la següent comanda:</br>

```bash
python lstm_genre_classification.py
```
</br>

6. Seguiu el progrés de l'entrenament a través de les mètriques de pèrdua (loss) i precisió (accuracy) que es mostren.</br>
7. Després de l'entrenament, avalueu el rendiment del model en el conjunt de proves observant la mètrica de precisió (accuracy).</br>
8. Visualitzeu els gràfics de pèrdua (loss) i precisió (accuracy) de l'entrenament per analitzar el rendiment del model durant l'entrenament.</br>

## Avaluació

Per avaluar el model entrenat en el conjunt de proves, executeu el codi com es descriu a la secció [Entrenament](#entrenament). El codi calcularà la precisió (accuracy) del model en el conjunt de proves i mostrarà el resultat.</br>

## Resultats

Els resultats del model són la precisió (accuracy) obtinguda en el conjunt de proves. Aquesta mètrica indica la capacitat del model per classificar correctament les mostres d'àudio en els gèneres corresponents. Els resultats també es poden visualitzar en forma de gràfics de pèrdua (loss) i precisió (accuracy) durant l'entrenament.</br>

| Encabezado 1 | Encabezado 2 | Encabezado 3 | Encabezado 4 | Encabezado 5 |
|--------------|--------------|--------------|--------------|--------------|
| Celda 1      | Celda 2      | Celda 3      | Celda 4      | Celda 5      |
| Celda 6      | Celda 7      | Celda 8      | Celda 9      | Celda 10     |
| Celda 11     | Celda 12     | Celda 13     | Celda 14     | Celda 15     |


## Ús

Podeu utilitzar aquest codi com a punt de partida per a la vostra pròpia classificació de gènere d'àudio utilitzant PyTorch. Feu les modificacions necessàries per adaptar-lo al vostre conjunt de dades i requeriments específics.</br>

## Agraïments

- Els canvis fets estàn inspirats amb les pràctiques i exemples fets durant l'assignatura.</br>
- El conjunt de dades utilitzat per a l'entrenament i avaluació és una adaptació d'un conjunt de dades disponible públicament. Moltes gràcies als creadors del conjunt de dades original.</br>



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
