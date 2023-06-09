
# XNAP-Project 5: Music Genre Classification
Write here a short summary about your project. The text must include a short introduction and the targeted goals</br>
Aquest projecte busca classificar diferents audios de musica en el genere musical al que pertanyen.


# PyTorch LSTM Classificació de Gènere - README

Aquest repositori conté una implementació en PyTorch d'un model LSTM de 2 capes per a la classificació de gènere de música. El model LSTM s'entrena amb diverses característiques d'àudio com ara el centroid espectral, el contrast, el cromagrama i els coeficients cepstral de Mel (MFCC).</br>

## Taula de continguts

- [Introducció](#introducció)</br>
- [Dependències](#dependències)</br>
- [Conjunt de dades](#conjunt-de-dades)</br>
- [Dataloader](#dataloader)</br>
- [Models](#models)</br>
- [Entrenament](#entrenament)</br>
- [Avaluació](#avaluació)</br>
- [Resultats](#resultats)</br>
- [Altre Dataset](#altre-dataset)</br>
- [Ús](#ús)</br>

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
- seaborn</br>
- pandas</br>
- scikit-learn

Podeu instal·lar les dependències requerides executant la següent comanda:</br>

```bash
pip install torch numpy librosa matplotlib seaborn pandas scikit-learn
```


## Conjunt de dades

El conjunt de dades utilitzat per a l'entrenament i avaluació consisteix en mostres d'àudio de diferents gèneres. Els fitxers d'àudio es pre-processen per extreure les característiques desitjades utilitzant la classe *GenreFeatureData*, que fa servir la llibreria librosa. Les dades pre-processades es guarden en format NumPy per a una càrrega eficient durant l'entrenament i l'avaluació.</br>
El conjunt de dades es divideix en tres subconjunts: entrenament, validació i prova. El conjunt d'entrenament s'utilitza per entrenar el model, el conjunt de validació es fa servir per a monitorar el rendiment del model durant l'entrenament i el conjunt de prova es fa servir per avaluar el model final.</br>

### Gtzan

El dataset GTZAN es pot obtenir des de el repositoir del starting point o be del Kaggle</br>
Aquest dataset té un total de 600 arxius, separats en tres carpetes: train, validation i test.
La primera consta de 420 arxius, la segona de 120 i la darrera de 60.</br>
Els arxius es troben en format *genere.id.au* on genere te 6 possibles classes i id és únic fins i tot entre carpetes diferents. Per tant es tracta d'un dataset molt fàcil de tractar.

### Fma small

El dataset [FMA small](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?select=fma_small)
conte 8000 arxius. No els trobem com en l'altre datasets, tenim dues diferències molt importants:</br>
-No ho tenim dividit en test i train</br>
-No tenim la classe en el nom de l'aixiu</br>
Els arxius es troben en format *id.mp3* dintre de 155 carpetes diferents.</br>


## Dataloader

*GenreFeatureData* és la classe que utilitzem en tots els casos per carregar les dades, és el nostre dataloader. Ens extreu característiques d'àudio i ens carrega el conjunt d'entrenament (train,test). Quant utilitzem aquesta classe pot carregar les dades de dos formes.</br>
La primera és quan no troba els arxius preprocessats. Llavors utilitza la funció d'extreure característiques per cada un dels conjunts (test train validation) i hi genera sis objectes numpy, per les dades i el target de cada conjunt. Un cop fa això guarda aquests arrays en un arxiu .npy. Aquests són precisament els arxius preprocessats.</br>
La segona es quant els troba perquè ja s'ha executat abans. Quant es així el programa es molt mes ràpid ja que nomes ha de llegir els arxius i ja es pot posar a entrenar.</br>
Depenen de quin dels tres arxius: _GenreFeatureData.py_, _GenreFeatureData_m.py_ o _GenreFeatureDataFMA.py_ importem, obtindrem un objecte amb el mateix nombre i crides de funcions, però amb funcionaments interns diferent.</br>

**GenreFeatureData.py:**</br>
Aquest arxiu conte el dataloader original. </br>
Per agafar les caractistiques simplement ha de recorre la carpeta en questio i utilitzar librosa i numpy per estrure i guardar la informació.</br>
Per agafar el target simplement ha de mirar el nom del arxiu ja que en aquest dataset es troben amb el format *genere.id.au*. Aquests targets els codifica amb one-hot encoding. Aquest dataloader esta preparat per classificar cualsevol nombre de generes sempre i quan els definim en una llista al principi de l’objecte.</br>

**GenreFeatureData_m.py:**</br>
És igual que l'anterior però quan extreu les característiques fa servir espectrograma. En les proves hem vist que finalment les execucions de l'espectrograma són pitjors que l'original. Per l'espectograma agafem el mel espectrograma que ens representa la freqüència d'una forma més visual tot i que ho utilitzem com a dades i no com a imatge.</br>

**GenreFeatureDataFMA.py:**</br>
Aquests és el més diferent, serveix per carregar les dades del fma_small.
Bàsicament ha de carregar de 4 arxius el llistat d'arxius train, test i validation així com el gènere de cada cançó. D'on hem tret aquests 4 arxius es parlarà més endavant.</br>
Per poder carregar-los hi ha canvis en les dues funcions que extreuen les dades. L'estructura és igual però utilitza unes línies extra per llegir els 3 arxius i una funció per llegir el csv on tenim els generes.</br>

Tots tres arxius són capaços de fer data augmentacion, es tracta de canviar la variable self.augmentar de False a True. Però en totes les proves veiem que no ens servia per treure problemes, per tant no s'utilitza. Bàsicament el que fa és carregar el mateix arxiu tres cops, l'original, amb soroll i augmentant/reduint la senyal.</br>
L'execució d'aquests codis pot generar problemes. Si troben un arxiu preprocessat no poden mirar si és d'espectrograma o si està augmentat. Per tant, si hi ha dubtes sobre la procedència la millor practica es eliminar els .npy </br>


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

En aquest model a diferència del anterior, trobem una capa adicional de dropout. També hem passat d'utilitzar el *GenreFeatureData* al *GenreFeatureData_m* (aquest GenreFeature ens retorna les característiques d'audio, però aquestes a diferencia del anterior GenreFeature provenen d'espectogrames de mel que han sigut pasats a una representació logarítimica per posteriorment extreure les seves característiques).</br>

Finalment, s'ha implementat un learning schedule i canviat la funció que utilitzavem per calcular la loss. A continuació els tres schedules utilitzats.</br>

**Lstm_pytorch_optim_ReduceLROnPlateau:** en aquest model s'ha utilitzat el schedule ReduceLROnPlateau que consisteix en monotoritzar el accuracy durant cada època d'entrenament. Si l'accuracy no millora durant un determinat nombre d'èpoques ( 10) consecutives, la taxa d'aprenentatge es redueix en un factor predeterminat (lr*0.1). La idea d'aquest schedule és ajustar el pas d'aprenentatge per aconseguir una millor convergència.</br>

**Lstm_pytorch_optim_CosineAnnealingLR:** el CosineAnnealingLR redueix gradualment la taxa d'aprenentatge durant el procés d'entrenament, seguint una funció de cosinus. Comença amb una taxa d'aprenentatge inicial alta i va disminuint fins a un valor mínim. A continuació, la taxa d'aprenentatge augmenta novament fins al seu valor inicial. Aquest procés es repeteix en cada cicle d'entrenament.<br>

La idea principal d'aquesta tècnica és permetre que el model "explori" diferents regions de l'espai de paràmetres, utilitzant una taxa d'aprenentatge més alta en les primeres etapes de l'entrenament per moure's més ràpidament i una taxa d'aprenentatge més baixa a mesura que es va aproximant a possibles òptims locals. Això pot ajudar a millorar el rendiment i evitar quedar-se estancat en òptims locals subòptims.<br>

Finalment el schedule utilitzat: <br>

**Lstm_pytorch_optim_StepLR:** El StepLR redueix la taxa d'aprenentatge en punts específics durant l'entrenament, anomenats "passos". Aquests passos estan determinats per un valor fix de nombre d'èpoques. Quan el nombre d'èpoques d'entrenament arriba a un d'aquests passos, la taxa d'aprenentatge es redueix segons un factor predefinit.<br>

L'objectiu principal d'utilitzar l'algorisme StepLR és ajustar la taxa d'aprenentatge per millorar el rendiment del model al llarg de l'entrenament. Reduir la taxa d'aprenentatge en moments específics pot ajudar a estabilitzar i refinar el model, especialment en situacions en què el model pot sobreajustar-se o quan s'acosta a un òptim local.<br>



***Altres: Model CNN***

S’ha intentat implementar un nou model per la classificació de música segons el gènere on s'utilitzen les imatges MFCC, coeficients per la representació de la senyal d’audio.
MFCC ens permet obtenir informació sobre audios obviant el soroll.

S’ha creat un una llibreta per processar les dades de nou, anomenada *data_process_cnn.ipynb*, on es processa els audios ja existents, guardant en un diccionari dues llistes, una corresponent al gènere i l’altre una array amb els valors de la imatge MFCC.

Per processar els audios s’ha dividit en 5 segments, per tant els audios de 30 segons estaven representats en 5 imatges, augmentant lligerament la mida de les dades.
El diccionari es guarda en un fitxer JSON pel train, test i validation respectivament.

El model CNN que segueix la següent estructura:
3 Capes convulacionals, amb un kernel ( 3x3 ) i un padding de 1, per detectar patrons importants en el MFCC.
3 Capes d’agrupació MaxPooling que de les característiques extretes en les capes anteriors es queda amb el valor més gran per a cada regió.

3 Capes de normalització BatchNorm2d, que tenen com objectiu facilitar l’entrenament i una millor convergencia de la red.
A més s’ha utilitzat la funció d’activació ReLU per evitar linealitat després de cada capa de convolució i finalment unes capes Flatten, Linear, ReLU i DropOut que transformen la sortida de les capes convolucionals anteriors i ho transformen xarxa completament connectada.

Aquest model s'executa desde la llibreta *model_cnn.ipynb*, pero actualment no funciona degut en un error al prepara les dades.


## Entrenament

Per entrenar el model, seguiu els següents passos:</br>

1. Assegureu-vos d'haver instal·lat totes les dependències esmentades a la secció [Dependències](#dependències).</br>
2. Baixeu el conjunt de dades o prepareu el vostre propi conjunt de dades amb mostres d'àudio i les seves etiquetes de gènere corresponents.</br>
3. Pre-processeu les mostres d'àudio i extreu les característiques d'àudio utilitzant la classe *GenreFeatureData*. Desa les dades pre-processades en format NumPy.</br>
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

| Mètrica             |   StepLR     |   Cosine     |   Plateau    |
|---------------------|--------------|--------------|--------------|
| Accuracy train      | 79,77%       | 96.63%       | 78.70%       | 
| Loss train          | 0.5256       | 0.3907       | 0.5924       | 
| Accuracy validation | 73.44%       | 68.75%       | 71.88%       | 
| Loss validation     | 1.044        | 1.8855       | 0.8942       | 
| Accuracy test       | 58,33%       | 68.33%       | 60.00%       | 
| Loss test           | 1.055        | 2.11         | 1.2041       | 


## Altre Dataset

Volíem veure que passava amb els models si els entrenàvem amb un dataset diferent, per fer-ho vam utilitzar el dataset fma descrit anteriorment, en concret la versió small. L'objectiu original per tant era simplement canviar el dataloader, però finalment la funció per entrenar va haver de ser canviada, no per motius d'execució sinó per la impressió de la matriu de confusió.</br>
Com ja s'ha comentat abans aquest dataset es troba estructurat deifernement:</br>
-No ho tenim dividit en test i train</br>
-No tenim la classe en el nom de l'arxiu</br>

Per solucionar el segon problema haurem d'utilitzar un arxiu *tracks.csv* que es troba també a la pàgina web. En la capeta fma_metadada trobem aquest arxiu. Per poder fer les execucions necessitem tenir en la carpeta fma l'arxiu *tracks.csv* i la carpeta *fma_small* que conté les 155 carpetes, el checksum i un readme.</br>
Una vegada hem descarregat o pujat aquests arxius de la pàgina web necessitarem fer diversos canvis, això es deu a que hi ha tres arxius de música que estan buits i per tant han de ser eliminats per evitar errors. Per treure'ls ho farem manualment ja que només ho hem de fer un cop. En la terminal situant-nos a la carpeta *fma_small* executem:</br>
```bash
rm 099/099134.mp3
rm 108/108925.mp3
rm 133/133297.mp3
```
</br>
I tenim aquests tres arxius eliminats. Però per poder executar la resta del codi també els hem de treure de l'arxiu checksum. Entrem a l'arxiu i busquem aquests tres i esborrem la línia de cada un d'ells al complet, tant la sèrie de caràcters de l'esquerra com l'arxiu com a tal. Amb aquests canvis ja no tenim un dataset amb errors i podem passar a processar-lo.</br>

### Preprocesament

Hem fet un codi, _read_fma.py_, que s'encarrega de generar 4 arxius que ens tornen aquest dataset en algo amigable pel nostre dataloader. Aquest codi llegeix els arxius tracks.csv i els arxius checksum. Amb aquests genera un tracks_small.csv que només conte dues columnes, una per id i una altre per el gènere de l'àudio. I genera tres arxius que contenen un llistat de les cançons que s'utilitzaran per train, validation i test.</br>
Amb aquests quatre arxius simplement hem de fer canvis al *GenreFeatureData* i tindrem l'arxiu *GenreFeatureDataFMA.py* que actua a nivell extern exactament igual que amb l'altre dataset. Per tant no hem de canviar res dels arxius on definim el model per executar-ho. Simplement fem la importació del dataloader d'aquest nou arxiu.</br>
Com ja s'ha comentat al principi hem de canviar també l'entrenament simplement perquè el nombre de classes i quines són canvia. Ho fem només perquè la funció d'entrenament ens imprimeix una matriu de confusió. A part d'aquest detall no canvia res.</br>

### Problema

Si fem els passos de descarregar i ordenar els arxius, eliminar els 3 arxius del dataset i checksum, i finalment executem _read_fma.py_ ens trobem amb un problema. Dintre del programa per algun motiu quan decidim quins ids són del train (ids que hem extret del chacksum) i tornem a buscar aquests ids de la mateixa forma no ens els troba tots.</br>
És a dir, emplenem perfectament el llistat d'arxius test (800) i validation (800) però pel train només emplena 1890 dels 6400 que hauria d'emplenar. Per tant ens queda un training més petit del que hi hauria. Desconeixem perquè fa això tot i que sabem perfectament en quina part del codi passa això (línia 63 bàsicament). Tampoc tenim cap teoria perquè per memòria no te cap problema.</br>
Si seguim endavant tenim un problema i és que tenim el train des-balancejat. Si executem el següent codi un cop preprocessem les dades amb el *GenreFeaturesData*:</br>
```python
import numpy as np
import torch
train_y=np.load("./fma/data_train_target.npy")
train_y=torch.from_numpy(train_y).type(torch.Tensor)
print(np.histogram(torch.max(train_y, 1)[1].numpy(),bins=7))
```
</br>
Ens informa que efectivament les classes estan desequilibrades

### Possibles Solucions

A partir d'aquí es pot entrenar igualment o intentar equilibrar. Si fem el primer els resultats es poden veure a la capeta de figures amb els noms fma_sense_juntar.</br>
Per intentar equilibrar podem concatenar training i validation, utilitzar test per tant la funció de validation com per la de test. Això no ens comporta molts problemes, test i validation en el nostre entrenament són intercanviables, i aconseguim que hi hagui una mica més de dades equilibrades. Els resultats d'aquest mètode es poden observar a figures amb els noms fma_junt</br>

### Resultats 

Podem observar que tots dos resultats són bastant negatius. Ambues obtenen 100% al train però 23% al test. I en la matriu de confusió veiem que te favoritismes cap a predir certes classes.</br>
Hem obtingut el nostre objectiu d'utilitzar un dataset diferent però el nostre model no es capaç d'adaptar-se. Cap la possibilitat que això es degui principalment a l'error explicat anteriorment. En tot cas no tenim resultats conclusions per afirmar que tenim un bon model per aquest dataset.


## Ús

Podeu utilitzar aquest codi com a punt de partida per a la vostra pròpia classificació de gènere d'àudio utilitzant PyTorch. Feu les modificacions necessàries per adaptar-lo al vostre conjunt de dades i requeriments específics.</br>

## Agraïments

- Els canvis fets estàn inspirats amb les pràctiques i exemples fets durant l'assignatura.</br>
- El conjunt de dades utilitzat per a l'entrenament i avaluació és una adaptació d'un conjunt de dades disponible públicament. Moltes gràcies als creadors del conjunt de dades original.</br>


## Contributors
Write here the name and UAB mail of the group members:
</br> Daniel Paulí - 1568073@uab.cat
</br> Bernat Planelles - 1525973@uab.cat
</br> Marti Torrents - 1605189@uab.cat

## Starting Point
Aquest projecte s'ha desenvolupat a partir del seguent setatring point:</br>
https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification</br>
El seu readme es pot trobar com el arxiu original_readme.md 

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades, 
UAB, 2023
