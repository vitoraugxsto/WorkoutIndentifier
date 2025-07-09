import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Para salvar e carregar o modelo

# 1. Carregar os Dados
file_path = 'dataset_rosca_completo.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Dataset '{file_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no diretório correto.")
    exit()

# Inspecionar as colunas para garantir que temos o que precisamos
required_columns = ['video', 'repeticao', 'angulo', 'velocidade', 'ombro_y', 'cotovelo_y', 'punho_y', 'desvio_5_frames', 'rotulo']
if not all(col in df.columns for col in required_columns):
    print(f"\nErro: Uma ou mais colunas necessárias ({required_columns}) não foram encontradas no dataset.")
    print("Colunas disponíveis: ", df.columns.tolist())
    exit()

# 2. Engenharia de Características por Repetição
# Agrupamos por 'video' e 'repeticao' para criar uma única linha de características por repetição
print("\nCriando dataset de características por repetição...")
features_per_repetition = df.groupby(['video', 'repeticao']).agg(
    # Para angulo
    min_angulo=('angulo', 'min'),
    max_angulo=('angulo', 'max'),
    mean_angulo=('angulo', 'mean'),
    std_angulo=('angulo', 'std'),
    
    # Para velocidade
    min_velocidade=('velocidade', 'min'),
    max_velocidade=('velocidade', 'max'),
    mean_velocidade=('velocidade', 'mean'),
    std_velocidade=('velocidade', 'std'),
    
    # Para coordenadas Y (considerando variação vertical como característica principal)
    range_ombro_y=('ombro_y', lambda x: x.max() - x.min()),
    mean_ombro_y=('ombro_y', 'mean'),
    range_cotovelo_y=('cotovelo_y', lambda x: x.max() - x.min()),
    mean_cotovelo_y=('cotovelo_y', 'mean'),
    range_punho_y=('punho_y', lambda x: x.max() - x.min()),
    mean_punho_y=('punho_y', 'mean'),
    
    # Para desvio_5_frames
    mean_desvio_5_frames=('desvio_5_frames', 'mean'),
    max_desvio_5_frames=('desvio_5_frames', 'max'),
    
    # Duração da repetição em frames
    duration_frames=('frame', 'count'),
    
    # O rótulo da repetição (qualidade)
    rotulo=('rotulo', lambda x: x.iloc[0]) # Pega o primeiro rótulo, assumindo que é consistente por repetição
).reset_index()

# Lidar com NaNs que podem surgir (ex: std de uma repetição de 1 frame)
# Preenchemos com 0; outra estratégia seria preencher com a média ou mediana das colunas
features_per_repetition = features_per_repetition.fillna(0)

print("\nPrimeiras 5 linhas do dataset de características por repetição:")
print(features_per_repetition.head())
print(f"\nTotal de repetições únicas encontradas: {len(features_per_repetition)}")

# 3. Preparação para o ML
# Separar características (X) e variável alvo (y)
X = features_per_repetition.drop(['video', 'repeticao', 'rotulo'], axis=1)
y = features_per_repetition['rotulo']

# Codificar a variável alvo (y): 'correto' -> 1, 'incorreto' -> 0
y_encoded = y.map({'correto': 1, 'incorreto': 0})

# Verificar a distribuição das classes antes do split
print(f"\nDistribuição das classes no dataset completo:\n{y_encoded.value_counts()}")

# Dividir os dados em conjuntos de treinamento e teste
# test_size: 20% para teste
# random_state: para reprodutibilidade
# stratify: importante para manter a proporção de classes (correto/incorreto) em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nShape do X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape do X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Distribuição das classes no treino:\n{y_train.value_counts()}")
print(f"Distribuição das classes no teste:\n{y_test.value_counts()}")

# 4. Treinamento do Modelo (RandomForestClassifier)
print("\nTreinando o modelo RandomForestClassifier...")
# n_estimators: número de árvores na floresta
# random_state: para reprodutibilidade
# class_weight='balanced': importante se as classes 'correto'/'incorreto' estiverem desbalanceadas
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Treinamento concluído.")

# 5. Avaliação do Modelo
print("\nAvaliando o modelo no conjunto de teste...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

print("\nRelatório de Classificação:")
# target_names: Nomes das classes para o relatório (0='incorreto', 1='correto')
# labels: Garante que ambas as classes sejam consideradas no relatório, mesmo que uma esteja ausente no y_test (cenários de dataset pequeno)
print(classification_report(y_test, y_pred, target_names=['incorreto', 'correto'], labels=[0, 1], zero_division=0)) # zero_division=0 para evitar warnings em classes sem predições

# 6. Salvar o Modelo Treinado
model_filename = 'rosca_quality_classifier_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModelo salvo como '{model_filename}'")

print("\n--- Próximos Passos ---")
print("1. Para usar este modelo em tempo real, você precisará carregar o arquivo '.joblib' em seu algoritmo de validação em tempo real.")
print("2. Quando uma repetição for detectada e os dados forem coletados, você deverá calcular as mesmas características resumidas (min, max, mean, std, ranges, duration) para essa nova repetição.")
print("3. Alimente essas características (como um DataFrame de 1 linha) para o modelo carregado para obter a previsão ('correto' ou 'incorreto').")
print("Exemplo de uso:")
print(f"   import joblib")
print(f"   loaded_model = joblib.load('{model_filename}')")
print(f"   # features_nova_rep deve ser um DataFrame/array 2D com as mesmas colunas (na mesma ordem) que X_train")
print(f"   # prediction = loaded_model.predict(features_nova_rep)")
print(f"   # probability_correct = loaded_model.predict_proba(features_nova_rep)[:, 1]")