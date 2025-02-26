# Testando o modelo treinado com os dados de 2024-01 para  
# realizar previsões a partir dos dados de 2024-03
import cloudpickle
import uuid

# Carregar o pipeline ELT referente a 2024-01
with open('./models/python_model.pkl', 'rb') as file:
    pipELT = cloudpickle.load(file)

# Carregar os dados de 2024-03
data_march = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet'

# Aplicando as etapas Extract e Load do pipeline: 'data_loader_cleaner'
loaded_data = pipELT.named_steps['data_loader_cleaner'].transform(data_march)

# Separar X (atributos) e y (alvo)
if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
    X, y = loaded_data
else:
    X = loaded_data
    y = None

# Verificar se 'rideID' existe em X
if 'rideID' not in X.columns:
    X['rideID'] = [str(uuid.uuid4()) for _ in range(len(X))]  # Criar valores únicos para cada linha


# Aplicando a etapa Transform do pipeline: 'preprocessor'
X_processed = pipELT.named_steps['preprocessor'].transform(X)

# Carregar o modelo treinado
with open('./models/model.pkl', 'rb') as file:
    trained_model = cloudpickle.load(file)

# Fazer previsões usando o modelo treinado
predictions = trained_model.predict(X_processed)
print(predictions)

