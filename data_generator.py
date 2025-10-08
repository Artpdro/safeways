
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import LabelEncoder
import holidays
import warnings

warnings.filterwarnings("ignore")

# Carregar o modelo e os encoders
try:
    with open("modelo_acidentes.pkl", "rb") as f:
        model_data = pickle.load(f)
    modelo = model_data["modelo"]
    encoders = model_data["encoders"]
    feature_names = model_data["features"]
    holidays_br = holidays.Brazil()
except FileNotFoundError:
    print("Erro: modelo_acidentes.pkl não encontrado. Certifique-se de que o modelo foi treinado e salvo.")
    # Fallback ou raise error, dependendo do comportamento desejado
    modelo = None
    encoders = {}
    feature_names = []
    holidays_br = holidays.Brazil()

def load_locations():
    """Carrega dados de UFs e municípios"""
    with open("uf_options.json", "r", encoding="utf-8") as f:
        uf_options = json.load(f)
    
    with open("municipios_por_uf.json", "r", encoding="utf-8") as f:
        municipios_por_uf = json.load(f)
    
    with open("condicoes_metereologicas_options.json", "r", encoding="utf-8") as f:
        condicoes_options = json.load(f)
    
    return uf_options, municipios_por_uf, condicoes_options

def _simplificar_clima(cond):
    if any(k in cond for k in ["Chuva", "Garoa"]):
        return "Chuva"
    if "Nublado" in cond:
        return "Nublado"
    if any(k in cond for k in ["Céu Claro", "Sol"]):
        return "Bom"
    if "Vento" in cond:
        return "Vento"
    if any(k in cond for k in ["Nevoeiro", "Neblina"]):
        return "Nevoeiro/Neblina"
    return "Outro"

def _criar_features_para_previsao(df_input):
    df = df_input.copy()
    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="coerce") # Ajustado formato da data
    df["hora"] = pd.to_datetime(df["horario"], format="%H:%M", errors="coerce").dt.hour # Ajustado formato do horário
    df["condicao_metereologica"] = df["condicao_metereologica"].apply(_simplificar_clima)

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia_semana"] = df["data"].dt.dayofweek
    df["dia_ano"] = df["data"].dt.dayofyear
    df["semana"] = df["data"].dt.isocalendar().week.astype(int)
    df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["feriado"] = df["data"].apply(lambda x: int(x in holidays_br))
    df["feriado_fim_semana"] = df["feriado"] * df["fim_semana"]

    # Lags e médias móveis serão 0 para dados de previsão sem histórico
    for lag in [1, 2, 7, 14]:
        df[f"lag_{lag}"] = 0
    for w in [7, 14, 28]:
        df[f"media_{w}"] = 0
        df[f"std_{w}"] = 0

    df.fillna(0, inplace=True)

    for col in ["uf", "municipio", "tipo_acidente", "condicao_metereologica"]:
        if col in df.columns and col in encoders:
            enc = encoders[col]
            df.loc[:, f"{col}_enc"] = df[col].apply(lambda x: enc.transform([x])[0] if x in enc.classes_ else -1)
        else:
            df.loc[:, f"{col}_enc"] = 0 # Valor padrão se encoder ou coluna não existirem

    # Renomear condicao_metereologica_enc para clima_enc para corresponder ao modelo
    if "condicao_metereologica_enc" in df.columns:
        df.rename(columns={"condicao_metereologica_enc": "clima_enc"}, inplace=True)

    # Garantir que todas as features esperadas pelo modelo estejam presentes
    X_prever = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df.columns:
            X_prever[col] = df[col]
        else:
            X_prever[col] = 0 # Preencher com 0 para features ausentes

    return X_prever

def generate_accident_data(num_records=1000):
    """Gera dados simulados de acidentes para demonstração usando o modelo de ML"""
    if modelo is None:
        print("Modelo de ML não carregado. Gerando dados de fallback.")
        # Implementar um fallback simples ou retornar um DataFrame vazio
        return pd.DataFrame()

    uf_options, municipios_por_uf, condicoes_options = load_locations()
    
    # Coordenadas aproximadas dos centros dos estados brasileiros (mantidas para lat/lon)
    uf_coordinates = {
        "AC": (-9.0238, -70.8120), "AL": (-9.5713, -36.7820), "AP": (1.4144, -51.7865), 
        "AM": (-4.1431, -69.8578), "BA": (-12.5797, -41.7007), "CE": (-5.4984, -39.3206),  
        "DF": (-15.7998, -47.8645), "ES": (-19.1834, -40.3089), "GO": (-15.827, -49.8362),  
        "MA": (-4.9609, -45.2744), "MT": (-12.6819, -56.9211), "MS": (-20.7722, -54.7852), 
        "MG": (-18.5122, -44.5550), "PA": (-3.9014, -52.4774), "PB": (-7.2399, -36.7819),  
        "PR": (-24.89, -51.55), "PE": (-8.8137, -36.9541), "PI": (-8.5569, -42.7401),  
        "RJ": (-22.9099, -43.2095), "RN": (-5.4026, -36.9541), "RS": (-30.0346, -51.2177), 
        "RO": (-10.9472, -62.8182), "RR": (1.99, -61.33), "SC": (-27.2423, -50.2189), 
        "SP": (-23.5505, -46.6333), "SE": (-10.5741, -37.3857), "TO": (-10.184, -48.3336)   
    }
    
    data_for_prediction = []
    
    # Gerar uma amostra de inputs para o modelo
    # Para simular dados "verídicos", vamos iterar sobre algumas combinações possíveis
    # e usar o modelo para prever o número de acidentes.
    # A diversidade dos dados gerados dependerá da diversidade dos inputs aqui.
    
    # Exemplo: gerar dados para os últimos 7 dias para algumas UFs e condições
    today = datetime.now()
    sample_ufs = uf_options[:5] # Pegar as primeiras 5 UFs como exemplo
    sample_condicoes = condicoes_options[:3] # Pegar as primeiras 3 condições

    for i in range(num_records):
        # Selecionar UF de forma cíclica ou aleatória (ainda precisamos de alguma variação)
        uf = sample_ufs[i % len(sample_ufs)]
        
        # Selecionar município (o primeiro da UF para simplificar, ou um de forma cíclica)
        municipio = municipios_por_uf[uf][0] if municipios_por_uf[uf] else "N/A"
        
        # Gerar data nos últimos 30 dias
        date_obj = today - timedelta(days=i % 30)
        
        # Gerar horário (a cada 4 horas para ter alguma variação)
        hour = (i * 4) % 24
        
        # Condição meteorológica
        weather = sample_condicoes[i % len(sample_condicoes)]

        # Gerar coordenadas próximas ao centro do estado (ainda com um pequeno offset para variação)
        base_lat, base_lon = uf_coordinates.get(uf, (0,0))
        lat = base_lat + (i % 10) * 0.1 - 0.5 # Pequena variação
        lon = base_lon + (i % 10) * 0.1 - 0.5 # Pequena variação

        data_for_prediction.append({
            "data": date_obj.strftime("%Y-%m-%d"), # Manter a coluna 'data' como string para o input do modelo
            "horario": f"{hour:02d}:00",
            "uf": uf,
            "municipio": municipio,
            "tipo_acidente": "Colisão", # Assumir um tipo de acidente padrão para previsão
            "condicao_metereologica": weather,
            "latitude": lat,
            "longitude": lon,
            "dia_semana": date_obj.strftime("%A"),
            "mes": date_obj.month,
            "ano": date_obj.year
        })
    
    df_input_for_model = pd.DataFrame(data_for_prediction)
    
    # Prever o número de acidentes usando o modelo
    X_prever = _criar_features_para_previsao(df_input_for_model)
    previsoes = np.clip(np.round(modelo.predict(X_prever)), 0, None).astype(int)
    
    df_result = df_input_for_model.copy()
    df_result["num_acidentes"] = previsoes
    
    # Converter a coluna 'data' para datetime APÓS a previsão para uso nas funções de agregação
    df_result["data"] = pd.to_datetime(df_result["data"])

    # Manter a coluna 'horario' para uso em get_hourly_accidents
    # df_result.drop(columns=["tipo_acidente"], inplace=True, errors="ignore")

    return df_result

def get_hourly_accidents():
    """Retorna dados de acidentes por horário para gráfico"""
    df = generate_accident_data(500)
    if df.empty: return pd.DataFrame(columns=["hora", "num_acidentes"])
    
    # Extrair a hora da coluna 'horario' que agora é mantida em df_result
    df["hora"] = pd.to_datetime(df["horario"], format="%H:%M").dt.hour.apply(lambda x: f"{x:02d}:00")

    hourly = df.groupby("hora")["num_acidentes"].sum().reset_index()
    
    # Garantir que temos todas as 24 horas
    all_hours = [f"{h:02d}:00" for h in range(24)]
    hourly_complete = pd.DataFrame({"hora": all_hours})
    hourly_complete = hourly_complete.merge(hourly, on="hora", how="left")
    hourly_complete["num_acidentes"] = hourly_complete["num_acidentes"].fillna(0)
    
    return hourly_complete

def get_daily_trend():
    """Retorna dados de tendência diária dos últimos 30 dias"""
    df = generate_accident_data(300)
    if df.empty: return pd.DataFrame(columns=["data", "num_acidentes"])
    
    # Filtrar últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # df["data"] já é datetime devido ao ajuste em generate_accident_data
    df_recent = df[df["data"] >= start_date]
    
    daily = df_recent.groupby("data")["num_acidentes"].sum().reset_index()
    daily["data"] = daily["data"].dt.strftime("%Y-%m-%d")
    
    # Preencher dias faltantes com 0
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    complete_dates = pd.DataFrame({"data": date_range.strftime("%Y-%m-%d")})
    daily_complete = complete_dates.merge(daily, on="data", how="left")
    daily_complete["num_acidentes"] = daily_complete["num_acidentes"].fillna(0)
    
    return daily_complete

def get_heatmap_data():
    """Retorna dados para mapa de calor"""
    df = generate_accident_data(200)
    if df.empty: return []
    
    # Agrupar por coordenadas aproximadas
    df["lat_rounded"] = df["latitude"].round(1)
    df["lon_rounded"] = df["longitude"].round(1)
    
    heatmap = df.groupby(["lat_rounded", "lon_rounded"])["num_acidentes"].sum().reset_index()
    heatmap = heatmap[heatmap["num_acidentes"] > 0]
    
    return heatmap[["lat_rounded", "lon_rounded", "num_acidentes"]].values.tolist()

def get_statistics():
    """Retorna estatísticas gerais para o dashboard"""
    df = generate_accident_data(1000)
    if df.empty: 
        return {
            "accidents_this_month": 0,
            "model_accuracy": 0,
            "predictions_today": 0,
            "active_alerts": 0
        }
    
    # Acidentes este mês
    current_month = datetime.now().month
    current_year = datetime.now().year
    accidents_this_month = df[(df["mes"] == current_month) & (df["ano"] == current_year)]["num_acidentes"].sum()
    
    # Predições hoje (simulado com base nos dados gerados)
    today_str = datetime.now().strftime("%Y-%m-%d")
    predictions_today = df[df["data"].dt.strftime("%Y-%m-%d") == today_str]["num_acidentes"].sum()
    
    # Alertas ativos (simulado, pode ser baseado em um limiar de previsões)
    active_alerts = (df["num_acidentes"] > 5).sum() # Exemplo: mais de 5 acidentes é um alerta
    
    # Precisão do modelo (fixo, pois vem do modelo treinado)
    model_accuracy = 89 # Este valor deve vir do modelo_acidentes.pkl se disponível
    if modelo is not None and "r2" in model_data:
        model_accuracy = round(model_data["r2"] * 100)

    return {
        "accidents_this_month": int(accidents_this_month),
        "model_accuracy": model_accuracy,
        "predictions_today": int(predictions_today),
        "active_alerts": int(active_alerts)
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

