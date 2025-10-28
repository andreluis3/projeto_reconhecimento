#COMO FAZER A DETECÇÃO DE IMAGEM E RECONHECIMENTO FACIAL

#primeiro passo identificar o rosto na imagem
# segundo passo colocar imagens em preto e branco

#usa tecnica de histograma para gradientes
#usa o método HOG (Histogram of Oriented Gradients) para localizar faces em imagens,
#usa landmarks para pegar e identificar pontos e mapear a imagem 

#usa o método DeepLearning para identificar os melhores 128 pontos para identificar o indivíduo 
#Essa representação é então comparada por meio de uma métrica de distância euclidiana,
# permitindo identificar se o rosto detectado corresponde a um indivíduo conhecido.

#a partir dessas medidas dos pontos a gente usa um método de comparação

#proximo passo é encontrar a pessoa baseado no código 


#BIBLIOTECAS PARA BAIXAR
bibliotecas: [c-make
dlip
face-recognition
opencv-python 
]
