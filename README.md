# annotator
Ferramentas úteis para anotações de datasets

git clone por:
```
https://github.com/salocinrevenge/Grafica.git
```

Antes de tudo é necessário instalar as dependências. Eu particularmente recomendo o venv realizando:
```
python3 -m venv .venv
source .venv/bin/activate

```

Para utilizar a ferramenta de visualização de anotações use:
```
python viewVideoLabel.py  --video PATH_TO_VIDEO --csvs PATH_TO_FOLDER_WITH_CSVS
```
Para isso, organize todas as anotacoes em uma pasta com esses csvs e passe o caminho dela para o algoritmo. Ele lerá todos os arquivos csvs.