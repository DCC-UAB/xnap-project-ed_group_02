
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
