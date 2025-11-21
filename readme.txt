Reconhecimento Facial – Documentação do Projeto

Esse projeto foi desenvolvido com foco em estudo, pesquisa e experimentação prática. O processo de treinar a máquina, ajustar as métricas e validar os resultados foi extremamente enriquecedor. A experiência de programar e pesquisar sobre reconhecimento facial, landmarks, geração de encodings, monitoramento e análise de desempenho trouxe aprendizados valiosos e resultados muito satisfatórios.

1. Ambiente Virtual (venv) e Instalação das Bibliotecas
Antes de iniciar, é necessário ativar o ambiente virtual e instalar todas as bibliotecas utilizadas no projeto.
Criar o ambiente virtual:
python -m venv venv
Ativar o ambiente:
Windows:
venv\Scripts\activate
Linux/Mac:
source venv/bin/activate
Instalar as dependências:
pip install -r requirements.txt
Certifique-se de ter todas as bibliotecas usadas no projeto, como: opencv-python, dlib, face_recognition, numpy, entre outras.

2. Estrutura do Dataset
O projeto utiliza um dataset localizado na pasta:
 /dataset
Cada pasta interna representa uma pessoa, contendo suas imagens faciais.

3. Arquivo de Encodings
encodings.pkl é gerado automaticamente pelo script de criação de encodings.

4. Ordem Recomendada de Execução
4.1. Validar o Dataset
Execute: ValidateDataset.py
4.2. Gerar Encodings
Execute: gerar_encodings.py
4.3. Treinamento
Execute: main.py

5. Módulos de Execução e Monitoramento
detecting.py captura screenshots da tela.
monitorar_tela.py monitora em segundo plano.
monitoring_continua.py monitora a webcam.
eficiencia.py gera métricas e CSVs.

6. Referências Utilizadas
https://stackoverflow.com/questions/50316600/training-a-model-to-achieve-dlibs-facial-landmarks-like-feature-points-for-hand
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826
https://face-recognition.readthedocs.io/en/latest/readme.html#installation
https://datahacker.rs/009-how-to-detect-facial-landmarks-using-dlib-and-opencv/
