from fasthtml.common import *
import plotly.graph_objects as go
import plotly.express as px
import json
from data_generator import get_hourly_accidents, get_daily_trend, get_heatmap_data, get_statistics
import folium
from folium.plugins import HeatMap
import base64
from io import BytesIO

def create_hourly_chart():
    """Cria gráfico de acidentes por horário"""
    hourly_data = get_hourly_accidents()
    
    fig = go.Figure(data=[
        go.Bar(
            x=hourly_data['hora'],
            y=hourly_data['num_acidentes'],
            marker_color='#4A90E2',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>Acidentes: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='',
        xaxis_title='Horário',
        yaxis_title='Número de Acidentes',
        margin=dict(t=20, r=20, b=40, l=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=300
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="hourly-chart")

def create_trend_chart():
    """Cria gráfico de tendência diária"""
    trend_data = get_daily_trend()
    
    fig = go.Figure(data=[
        go.Scatter(
            x=trend_data['data'],
            y=trend_data['num_acidentes'],
            mode='lines+markers',
            line=dict(color='#4A90E2', width=3),
            marker=dict(color='#4A90E2', size=6),
            hovertemplate='<b>%{x}</b><br>Acidentes: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='',
        xaxis_title='Data',
        yaxis_title='Número de Acidentes',
        margin=dict(t=20, r=20, b=40, l=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=300
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="trend-chart")

def create_heatmap():
    """Cria mapa de calor usando Folium"""
    try:
        # Criar mapa centrado no Brasil
        m = folium.Map(
            location=[-14.2350, -51.9253],  # Centro do Brasil
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Obter dados do mapa de calor
        heat_data = get_heatmap_data()
        
        # Adicionar camada de calor
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # Converter para HTML
        map_html = m._repr_html_()
        
        return f'<div style="height: 400px; border-radius: 10px; overflow: hidden;">{map_html}</div>'
    
    except Exception as e:
        # Fallback para visualização simples
        return '''
        <div style="background: linear-gradient(45deg, #e3f2fd, #bbdefb); 
                    height: 400px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    border-radius: 10px;
                    color: #1976d2;
                    font-size: 18px;
                    font-weight: 600;
                    text-align: center;">
            🗺️ Mapa de Calor Interativo<br>
            <small style="font-size: 14px; font-weight: normal;">
                Visualização de acidentes por região
            </small>
        </div>
        '''

def create_weather_distribution_chart():
    """Cria gráfico de distribuição por condição meteorológica"""
    # Dados simulados de distribuição por clima
    weather_data = {
        'Bom': 45,
        'Chuva': 25,
        'Nublado': 15,
        'Vento': 8,
        'Nevoeiro': 5,
        'Outro': 2
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(weather_data.keys()),
            values=list(weather_data.values()),
            hole=0.4,
            marker_colors=['#4A90E2', '#2C5282', '#63B3ED', '#90CDF4', '#BEE3F8', '#E6F3FF']
        )
    ])
    
    fig.update_layout(
        title='Distribuição por Condição Meteorológica',
        margin=dict(t=40, r=20, b=20, l=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=300
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="weather-chart")

def create_uf_ranking_chart():
    """Cria gráfico de ranking de UFs com mais acidentes"""
    # Dados simulados de ranking por UF
    uf_data = {
        'SP': 180,
        'RJ': 145,
        'MG': 120,
        'RS': 95,
        'PR': 85,
        'BA': 75,
        'SC': 65,
        'GO': 55,
        'PE': 50,
        'CE': 45
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(uf_data.values()),
            y=list(uf_data.keys()),
            orientation='h',
            marker_color='#4A90E2',
            opacity=0.8
        )
    ])
    
    fig.update_layout(
        title='Top 10 UFs - Acidentes este Mês',
        xaxis_title='Número de Acidentes',
        yaxis_title='UF',
        margin=dict(t=40, r=20, b=40, l=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=400
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="uf-ranking-chart")

def get_dashboard_stats():
    """Retorna estatísticas para os cards do dashboard"""
    return get_statistics()

def dashboard_content():
    """Retorna o conteúdo completo do dashboard"""
    stats = get_dashboard_stats()
    
    return Div(
        # Cards de estatísticas
        Div(
            Div(
                Div(
                    Div(f"{stats['accidents_this_month']:,}", cls="stats-number"),
                    Div("Acidentes este mês", cls="stats-label"),
                    cls="stats-card"
                ),
                cls="col-md-3"
            ),
            Div(
                Div(
                    Div(f"{stats['model_accuracy']}%", cls="stats-number"),
                    Div("Precisão do modelo", cls="stats-label"),
                    cls="stats-card"
                ),
                cls="col-md-3"
            ),
            Div(
                Div(
                    Div(f"{stats['predictions_today']:,}", cls="stats-number"),
                    Div("Predições hoje", cls="stats-label"),
                    cls="stats-card"
                ),
                cls="col-md-3"
            ),
            Div(
                Div(
                    Div(f"{stats['active_alerts']}", cls="stats-number"),
                    Div("Alertas ativos", cls="stats-label"),
                    cls="stats-card"
                ),
                cls="col-md-3"
            ),
            cls="row mb-4"
        ),
        
        # Primeira linha de gráficos
        Div(
            Div(
                Div(
                    Div(
                        H5("Mapa de Calor - Acidentes por Região", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        Div(create_heatmap(), id="heatmap-container"),
                        cls="card-body p-0"
                    ),
                    cls="card"
                ),
                cls="col-md-8"
            ),
            Div(
                Div(
                    Div(
                        H5("Acidentes por Horário", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        Div(id="hourly-chart-container"),
                        cls="card-body"
                    ),
                    cls="card"
                ),
                cls="col-md-4"
            ),
            cls="row mb-4"
        ),
        
        # Segunda linha de gráficos
        Div(
            Div(
                Div(
                    Div(
                        H5("Tendência de Acidentes - Últimos 30 dias", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        Div(id="trend-chart-container"),
                        cls="card-body"
                    ),
                    cls="card"
                ),
                cls="col-md-8"
            ),
            Div(
                Div(
                    Div(
                        H5("Condições Meteorológicas", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        Div(id="weather-chart-container"),
                        cls="card-body"
                    ),
                    cls="card"
                ),
                cls="col-md-4"
            ),
            cls="row mb-4"
        ),
        
        # Terceira linha - Ranking UFs
        Div(
            Div(
                Div(
                    Div(
                        H5("Ranking de Estados", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        Div(id="uf-ranking-container"),
                        cls="card-body"
                    ),
                    cls="card"
                ),
                cls="col-12"
            ),
            cls="row"
        ),
        
        # Scripts para renderizar gráficos
        Script(f"""
            // Renderizar gráfico de acidentes por horário
            document.getElementById('hourly-chart-container').innerHTML = `{create_hourly_chart()}`;
            
            // Renderizar gráfico de tendência
            document.getElementById('trend-chart-container').innerHTML = `{create_trend_chart()}`;
            
            // Renderizar gráfico de condições meteorológicas
            document.getElementById('weather-chart-container').innerHTML = `{create_weather_distribution_chart()}`;
            
            // Renderizar ranking de UFs
            document.getElementById('uf-ranking-container').innerHTML = `{create_uf_ranking_chart()}`;
        """),
        
        cls="content-area"
    )
