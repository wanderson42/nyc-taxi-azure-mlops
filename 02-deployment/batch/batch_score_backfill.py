from datetime import datetime
from dateutil.relativedelta import relativedelta
import asyncio

from prefect import flow
import batch_score

@flow
async def ride_duration_prediction_backfill():
    """
    Executa previsões em lote da duração das viagens de táxi
    considerando multiplos conjuntos de dados.

    Este script chama `batch_score.predictions` para gerar previsões 
    e um intervalo de meses estipulado, processando um conjunto de dados
    mensal por vez.

    Args:
        Nenhum argumento externo é necessário.

    Returns:
        Nenhum retorno explícito. Os resultados são processados e armazenados via 'batch_score.predictions'.
    """

    start_date = datetime(year=2024, month=2, day=1)
    end_date = datetime(year=2024, month=11, day=1)
    
    current_date = start_date
    
    while current_date <= end_date:
        print(f"Iniciando previsão para {current_date.strftime('%Y-%m')}")
        
        await batch_score.predictions(
            taxi_type='green',
            run_id='2c834544a1ac49cd979583e75c1eb3f6',
            run_date=current_date
        )
        
        current_date += relativedelta(months=1)
    
    print("Processo de backfill concluído.")

if __name__ == '__main__':
    asyncio.run(ride_duration_prediction_backfill())
