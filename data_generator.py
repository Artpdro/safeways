import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

def load_locations():
    """Carrega dados de UFs e municípios"""
    with open('uf_options.json', 'r', encoding='utf-8') as f:
        uf_options = json.load(f)
    
    with open('municipios_por_uf.json', 'r', encoding='utf-8') as f:
        municipios_por_uf = json.load(f)
    
    return uf_options, municipios_por_uf

def generate_accident_data(num_records=1000):
    """Gera dados simulados de acidentes para demonstração"""
    uf_options, municipios_por_uf = load_locations()
    
    # Coordenadas aproximadas dos centros dos estados brasileiros
    uf_coordinates = {
        'AC': (-9.0238, -70.8120),  # Acre
        'AL': (-9.5713, -36.7820),  # Alagoas
        'AP': (1.4144, -51.7865),   # Amapá
        'AM': (-4.1431, -69.8578),  # Amazonas
        'BA': (-12.5797, -41.7007), # Bahia
        'CE': (-5.4984, -39.3206),  # Ceará
        'DF': (-15.7998, -47.8645), # Distrito Federal
        'ES': (-19.1834, -40.3089), # Espírito Santo
        'GO': (-15.827, -49.8362),  # Goiás
        'MA': (-4.9609, -45.2744),  # Maranhão
        'MT': (-12.6819, -56.9211), # Mato Grosso
        'MS': (-20.7722, -54.7852), # Mato Grosso do Sul
        'MG': (-18.5122, -44.5550), # Minas Gerais
        'PA': (-3.9014, -52.4774),  # Pará
        'PB': (-7.2399, -36.7819),  # Paraíba
        'PR': (-24.89, -51.55),     # Paraná
        'PE': (-8.8137, -36.9541),  # Pernambuco
        'PI': (-8.5569, -42.7401),  # Piauí
        'RJ': (-22.9099, -43.2095), # Rio de Janeiro
        'RN': (-5.4026, -36.9541),  # Rio Grande do Norte
        'RS': (-30.0346, -51.2177), # Rio Grande do Sul
        'RO': (-10.9472, -62.8182), # Rondônia
        'RR': (1.99, -61.33),       # Roraima
        'SC': (-27.2423, -50.2189), # Santa Catarina
        'SP': (-23.5505, -46.6333), # São Paulo
        'SE': (-10.5741, -37.3857), # Sergipe
        'TO': (-10.184, -48.3336)   # Tocantins
    }
    
    data = []
    
    for _ in range(num_records):
        # Selecionar UF aleatória
        uf = random.choice(uf_options)
        
        # Selecionar município aleatório da UF
        municipio = random.choice(municipios_por_uf[uf])
        
        # Gerar coordenadas próximas ao centro do estado
        base_lat, base_lon = uf_coordinates[uf]
        lat = base_lat + random.uniform(-2, 2)
        lon = base_lon + random.uniform(-2, 2)
        
        # Gerar data aleatória nos últimos 365 dias
        days_ago = random.randint(0, 365)
        date = datetime.now() - timedelta(days=days_ago)
        
        # Gerar horário com distribuição realista (mais acidentes em horários de pico)
        hour_weights = [2, 1, 1, 1, 2, 4, 8, 12, 10, 8, 6, 8, 10, 12, 15, 18, 20, 18, 15, 12, 8, 6, 4, 3]
        hour = random.choices(range(24), weights=hour_weights)[0]
        
        # Número de acidentes (1-5, com peso maior para 1)
        num_accidents = random.choices([1, 2, 3, 4, 5], weights=[60, 25, 10, 3, 2])[0]
        
        # Condição meteorológica
        weather_conditions = ["Bom", "Chuva", "Nublado", "Vento", "Nevoeiro/Neblina", "Outro"]
        weather_weights = [50, 20, 15, 8, 5, 2]
        weather = random.choices(weather_conditions, weights=weather_weights)[0]
        
        data.append({
            'data': date.strftime('%Y-%m-%d'),
            'hora': f"{hour:02d}:00",
            'uf': uf,
            'municipio': municipio,
            'latitude': lat,
            'longitude': lon,
            'num_acidentes': num_accidents,
            'condicao_meteorologica': weather,
            'dia_semana': date.strftime('%A'),
            'mes': date.month,
            'ano': date.year
        })
    
    return pd.DataFrame(data)

def get_hourly_accidents():
    """Retorna dados de acidentes por horário para gráfico"""
    df = generate_accident_data(500)
    hourly = df.groupby('hora')['num_acidentes'].sum().reset_index()
    
    # Garantir que temos todas as 24 horas
    all_hours = [f"{h:02d}:00" for h in range(24)]
    hourly_complete = pd.DataFrame({'hora': all_hours})
    hourly_complete = hourly_complete.merge(hourly, on='hora', how='left')
    hourly_complete['num_acidentes'] = hourly_complete['num_acidentes'].fillna(0)
    
    return hourly_complete

def get_daily_trend():
    """Retorna dados de tendência diária dos últimos 30 dias"""
    df = generate_accident_data(300)
    
    # Filtrar últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df['data'] = pd.to_datetime(df['data'])
    df_recent = df[df['data'] >= start_date]
    
    daily = df_recent.groupby('data')['num_acidentes'].sum().reset_index()
    daily['data'] = daily['data'].dt.strftime('%Y-%m-%d')
    
    # Preencher dias faltantes com 0
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    complete_dates = pd.DataFrame({'data': date_range.strftime('%Y-%m-%d')})
    daily_complete = complete_dates.merge(daily, on='data', how='left')
    daily_complete['num_acidentes'] = daily_complete['num_acidentes'].fillna(0)
    
    return daily_complete

def get_heatmap_data():
    """Retorna dados para mapa de calor"""
    df = generate_accident_data(200)
    
    # Agrupar por coordenadas aproximadas
    df['lat_rounded'] = df['latitude'].round(1)
    df['lon_rounded'] = df['longitude'].round(1)
    
    heatmap = df.groupby(['lat_rounded', 'lon_rounded'])['num_acidentes'].sum().reset_index()
    heatmap = heatmap[heatmap['num_acidentes'] > 0]
    
    return heatmap[['lat_rounded', 'lon_rounded', 'num_acidentes']].values.tolist()

def get_statistics():
    """Retorna estatísticas gerais para o dashboard"""
    df = generate_accident_data(1000)
    
    # Acidentes este mês
    current_month = datetime.now().month
    current_year = datetime.now().year
    accidents_this_month = df[(df['mes'] == current_month) & (df['ano'] == current_year)]['num_acidentes'].sum()
    
    # Predições hoje (simulado)
    predictions_today = random.randint(400, 800)
    
    # Alertas ativos (simulado)
    active_alerts = random.randint(15, 35)
    
    # Precisão do modelo (fixo)
    model_accuracy = 89
    
    return {
        'accidents_this_month': int(accidents_this_month),
        'model_accuracy': model_accuracy,
        'predictions_today': predictions_today,
        'active_alerts': active_alerts
    }

if __name__ == "__main__":
    # Teste das funções
    print("Gerando dados de teste...")
    df = generate_accident_data(100)
    print(f"Gerados {len(df)} registros de acidentes")
    print(df.head())
    
    print("\nEstatísticas:")
    stats = get_statistics()
    print(stats)
    
    print("\nDados por horário:")
    hourly = get_hourly_accidents()
    print(hourly.head(10))
