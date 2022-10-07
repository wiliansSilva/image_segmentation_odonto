# DeepRAD: Rede neural para segmentação de imagens para odontologia

<div align="center">
 	<img src="./Divulgacao/Imagens/icone_ufpel_192.png" width="100" height="100" style="border-radius:100px"> 
	<img src="./Divulgacao/Imagens/icone_gaia_192.jpg" width="100" height="100">
</div>
<!--![](/Divulgacao/Imagens/icone_gaia_192.jpg)
![](/Divulgacao/Imagens/icone_ufpel_192.png)-->

## Apresentação

O projeto **DeepRAD** do [Grupo de Pesquisa em Inteligência Artificial](https://wp.ufpel.edu.br/gaia/) da [Universidade Federal de Pelotas](https://portal.ufpel.edu.br/) tem como objetivo é a identificação dos elementos dentários presentes, anomalias dentárias e no diagnóstico de doenças bucais (lesões periapicais e lesões de cárie), utilizando radiografias panorâmicas e bitewing.

Atualmente está sendo desenvolvido uma Rede Neural Convolucional (CNN) para identificar os seguintes:

- **Restaurações**
- **Coroa dentária**
- **Tratamento de canal**
- **Implante dental**
- **Dente**
- **Polpa dentária**

Importante ressalter que durante o projeto podem vir a serem adicionados novos pontos de interesse que irão ser detectados pelo algoritmo.

No campo odontológico, as CNNs já foram utilizadas para detectar lesões de cárie em radiografias de dentes permanentes, para detecção de perda óssea periodontal em radiografias periapicais ou para diagnosticar tumores na mandíbula em radiografias panorâmicass. **Portanto o uso das CNNs para esses fins pode reduzir esse esforço e facilitar a rotina do cirurgião-dentista.**

Serão analisadas cerca de 100000 radiografias (panorâmicas, periapicais, bitewing) e 8000 tomografias anônimas digitais com seus respectivos laudos digitais radiológicos cedidas por uma clínica radiológica privada. 

Para cada desfecho, o teste de referência será baseado na informação contida nos laudos radiológicos digitais (padrão-ouro) associado às respectivas imagens utilizando processamento de linguagem natural. Será possível utilizar ferramentas de magnificação e aprimoramento das imagens  (contraste e brilho). Será utilizada uma CNN customizada e pré-treinada  e um grid search será desenvolvido.

## Exemplos

A seguir são apresentados alguns exemplos dos resultados já obtidos:

## Modelos disponíveis

Quando estiverem disponíveis colocaremos aqui para que seja possível fazer o download deles.

## Eventos

### Posters Enviados

- **12º IEEE CASS (2022):**

<div align="center" >
	<img alt="Poster da 12º IEEE CASS de Guilherme Peglow" src="./Divulgacao/Posters/CASSW-RS_Poster_X-RaySegmentation_2022.png" width="500">
</div>

## Tecnologias aplicadas

Algumas das tecnologias que foram aplicadas no projeto:

- Python (linguagem de programação utilizada)
- Tensorflow e Keras (biblioteca para treinamento da rede neural)
- Segmentation Models (biblioteca que contém alguns modelos populares de CNN implementados)
- Google Colab (ambiente para treinamento dos modelos)
- LabelMe (anotação dos pontos de interesse nas imagens)

<div align="center">
	<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="60px"/>
	<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="60px"/>
</div>

## Membros

- [Anderson Priebe Ferrugem](https://github.com/MFerrugem)
- [Mauricio Braga de Paula](https://github.com/maubrapa)
- Jonas Almeida Rodrigues
- Dante Augusto Couto Barone
- Guilherme Nunes Peglow
- [Wilians Donizete Da Silva Júnior](https://github.com/wiliansSilva)
- [Felipe Dias Lopes](https://github.com/fdloopes)
- Alessandro Bof de Oliveira
- [Gabriel Leite Bessa](https://github.com/glbessa)


## Metodologia

<br>

**Materiais e métodos**

Para o desenvolvimento da arquitetura do modelo, será avaliado o desempenho do modelo candidato em relação à área abaixo da curva ROC (AUC). Serão avaliados diferentes números de unidades neurais (16 a 2048 em 2n) e números de filtros (16 a 2048 em 2n) para cada específica camada  convulacional. Posteriormente serão aplicados diferentes tamanhos de núcleo (2x2 a 5x5) e as configurações da camada max-pooling serão avaliadas (2x2 a 4x4). Como funções ativadoras serão usadas unidades lineares retificadas (ReLUs) e sigmoides. O desenvolvimento métrico primário será AUC, que está relacionada com a habilidade de um teste (no caso desse estudo, um modelo) classificar de maneira  correta (saudável/doente). As métricas secundárias serão a sensibilidade e a especificidade, juntamente com os valores preditivos positivo e negativo (VPP e VPN). 

<br>

**Banco de imagens e pré-processamento**

Serão cedidas cerca de 100000 radiografias (panorâmicas e periapicais, e bitewings) e 8000 tomografias anônimas com seus respectivos laudos radiológicos digitais numerados e também anonimizados cedidos por uma clínica radiológica privada (termos anexados). A partir deste banco bruto, dois examinadores experientes selecionarão as radiografias que não apresentarem alterações de brilho, cor, contraste e posição, dividindo-as de acordo com a técnica (periapical ou bitewing). As imagens serão padronizadas em tamanho (pixels) e cor (RGB).

<br>

**Desfechos analisados**

Nas panorâmicas e laudos:
- Identificação dos dentes presentes (decíduos e permanentes) em panorâmicas
de pacientes em fase de dentição decídua, mista e permanente;
- Anomalias dentárias (tumores malignos e benignos, dentes supranumerários,
agenesias, dentes impactados, anquilose, fusão, geminação)
- Lesões periapicais em dentes decíduos e permanentes
- Lesões de cárie (proximais).
Nas radiografias bitewings e laudos:
- Lesões de cárie (dentes decíduos e permanentes)Nas radiografias periapicais e laudos:
- Lesões periapicais em dentes decíduos e permanentes
Nas tomografias e laudos:
- Anomalias dentárias (tumores malignos e benignos, dentes supranumerários,
agenesias, dentes impactados, anquilose, fusão, geminação)

<br>

**Teste de Referência (aprendizado de máquina)**

Para cada desfecho, o teste de referência será baseado na informação
contida nos laudos radiológicos digitais (padrão-ouro) associado às respectivas
imagens utilizando processamento de linguagem natural. Será possível utilizar
ferramentas de magnificação e aprimoramento das imagens (contraste e brilho).

<br>

**Indicadores, metas e resultados esperados**

Será calculado a reprodutibilidade interexaminador através do teste Kappa de Fleiss (FLEISS JL., 1971), que é uma extensão do pi de Scott (SCOTT, 1955) e avalia a reprodutibilidade de acordo com uma escala nominal entre mais de dois examinadores (FLEISS JL., 1971).

Diferentes modelos de desenvolvimento métrico serão utilizados, tais como AUC, sensibilidade, especificidade, valor preditivo positivo (VPP, também conhecido como precisão) e o valor preditivo negativo (VPN) (SOKOLOVA e LAPALME, 2009).

A AUC está relacionada com a habilidade de evitar uma falsa classificação. A sensibilidade é a capacidade de identificar rótulos positivos enquanto que a especificidade é a capacidade de identificar rótulos negativos. O VPP é responsável pelo acordo de classe dos rótulos de dados com os rótulos positivos fornecidos pelo classificador, já o VPN contabiliza o acordo de classe dos rótulos de dados com os rótulos negativos fornecidos pelo classificador (SOKOLOVA AND LAPALME, 2009).

O desenvolvimento métrico primário será AUC, que está relacionada com a habilidade de um teste (no caso desse estudo, um modelo) classificar de maneira correta (saudável/doente).

As métricas secundárias serão a sensibilidade e a especificidade, juntamente com os valores preditivos positivo e negativo (VPP e VPN) (EKERT, T. et.al., 2019).