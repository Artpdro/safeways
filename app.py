from fasthtml.common import *
from dashboard import create_heatmap 
import json
import random
from datetime import datetime, timedelta

# Configuração básica sem CDNs externos
app, rt = fast_app()

def load_json_data(filename):
    """Carrega dados de arquivo JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_full_css():
    """Retorna todo o CSS necessário"""
    return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8f9fa;
            line-height: 1.6;
        }
        
        /* Sidebar */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: 250px;
            background: linear-gradient(180deg, #4A90E2 0%, #2C5282 100%);
            color: white;
            z-index: 1000;
            overflow-y: auto;
        }
        
        .sidebar-header {
            padding: 25px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            text-align: center;
        }
        
        .sidebar-header h3 {
            color: white;
            font-weight: 700;
            font-size: 24px;
            margin: 0;
        }
        
        .sidebar-subtitle {
            color: rgba(255,255,255,0.8);
            font-size: 12px;
            margin-top: 8px;
        }
        
        .sidebar-nav {
            padding: 20px 0;
        }
        
        .sidebar-nav a {
            display: block;
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            padding: 15px 25px;
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
            font-size: 14px;
        }
        
        .sidebar-nav a:hover {
            background-color: rgba(255,255,255,0.1);
            color: white;
            border-left-color: white;
        }
        
        .sidebar-nav a.active {
            background-color: rgba(255,255,255,0.2);
            color: white;
            border-left-color: white;
            font-weight: 600;
        }
        
        /* Main content */
        .main-content {
            margin-left: 250px;
            min-height: 100vh;
        }
        
        /* Header */
        .top-header {
            background: white;
            padding: 25px 30px;
            border-bottom: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }
        
        .welcome-text {
            color: #2C5282;
            font-size: 28px;
            font-weight: 700;
            margin: 0 0 15px 0;
        }
        
        .user-info {
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 12px;
            display: flex;
            align-items: center;
        }
        
        .user-avatar {
            width: 45px;
            height: 45px;
            background: #4A90E2;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            margin-right: 15px;
        }
        
        .user-details h4 {
            margin: 0;
            color: #2C5282;
            font-size: 16px;
        }
        
        .user-details p {
            margin: 2px 0 0 0;
            color: #28a745;
            font-size: 12px;
            font-weight: 600;
        }
        
        /* Content area */
        .content-area {
            padding: 30px;
        }
        
        /* Grid system */
        .row {
            display: flex;
            flex-wrap: wrap;
            margin: -10px;
        }
        
        .chart-row {
            display: flex;
            flex-wrap: wrap;
            margin: 0;
            width: 100%;
            margin-top: 30px;
        }
        
        .col-md-3 {
            flex: 0 0 25%;
            padding: 10px;
        }
        
        .col-md-6 {
            flex: 0 0 50%;
            padding: 5px;
        }
        
        .chart-col {
            flex: 0 0 50%;
            padding: 0 5px;
            max-width: 50%;
        }
        
        .col-md-8 {
            flex: 0 0 66.666%;
            padding: 10px;
        }
        
        .col-md-4 {
            flex: 0 0 33.333%;
            padding: 10px;
        }
        
        .col-12 {
            flex: 0 0 100%;
            padding: 10px;
        }
        
        /* Cards */
        .stats-card {
            background: white;
            border-radius: 15px;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        
        .stats-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .stats-number {
            font-size: 2.8rem;
            font-weight: 700;
            color: #4A90E2;
            margin-bottom: 10px;
            line-height: 1;
        }
        
        .stats-label {
            color: #6c757d;
            font-size: 14px;
            font-weight: 500;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .card-header {
            background: white;
            border-bottom: 1px solid #e9ecef;
            padding: 20px 25px;
        }
        
        .card-header h5 {
            color: #2C5282;
            font-weight: 600;
            margin: 0;
            font-size: 18px;
        }
        
        .card-body {
            padding: 25px;
        }
        
        /* Forms */
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            color: #2C5282;
            font-weight: 600;
            margin-bottom: 8px;
            display: block;
            font-size: 14px;
        }
        
        .form-control, .form-select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s ease;
            background: white;
        }
        
        .form-control:focus, .form-select:focus {
            outline: none;
            border-color: #4A90E2;
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
        }
        
        /* Buttons */
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #4A90E2 0%, #2C5282 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-lg {
            padding: 15px 40px;
            font-size: 16px;
        }
        
        /* Chart containers */
        .chart-container {
            position: relative;
            height: 400px;
            width: 100%;
            min-height: 400px;
        }
        
        .chart-container canvas {
            width: 100% !important;
            height: 400px !important;
            max-width: 100% !important;
        }
        
        .chart-col .card {
            height: 100%;
            margin: 0;
        }
        
        .chart-col .card-body {
            padding: 15px;
        }
        
        /* Heatmap container adjustments for folium */
        #heatmap-container iframe {
            width: 100% !important;
            height: 400px !important;
            border: none !important;
        }
        

        
        /* Prediction result */
        .prediction-result {
            margin-top: 20px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid;
        }
        
        .prediction-result.bg-success {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        
        .prediction-result.bg-warning {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }
        
        .prediction-result.bg-danger {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        
        .result-number {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 15px 0;
            line-height: 1;
        }
        
        /* Lists */
        ul {
            padding-left: 20px;
        }
        
        li {
            margin-bottom: 8px;
            color: #495057;
        }
        
        /* Utilities */
        .mb-3 { margin-bottom: 1rem; }
        .mb-4 { margin-bottom: 1.5rem; }
        .mt-3 { margin-top: 1rem; }
        .text-primary { color: #4A90E2; }
        .text-success { color: #28a745; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .col-md-3, .col-md-4, .col-md-8 {
                flex: 0 0 100%;
            }
            
            .content-area {
                padding: 15px;
            }
            
            .stats-number {
                font-size: 2.2rem;
            }
            
            .welcome-text {
                font-size: 24px;
            }
        }
    </style>
    """

def sidebar(active_page="dashboard"):
    """Cria a sidebar de navegação"""
    return f"""
    <div class="sidebar">
        <div class="sidebar-header">
            <h3>SafeWay</h3>
            <div class="sidebar-subtitle">Sistema de rotas seguras em construção</div>
        </div>
        <div class="sidebar-nav">
            <a href="/" class="{'active' if active_page == 'dashboard' else ''}">🏠 Dashboard</a>
            <a href="/prediction" class="{'active' if active_page == 'prediction' else ''}">📊 Predição</a>
        </div>
    </div>
    """

def top_header():
    """Cria o header superior"""
    return """
    <div class="top-header">
        <h1 class="welcome-text">Bem-vindo ao SafeWay</h1>
        <div class="user-info">
            <div class="user-avatar">U</div>
            <div class="user-details">
                <h4>exemplo@gmail.com</h4>
                <p>Administrador</p>
            </div>
        </div>
    </div>
    """

def create_stats_cards():
    """Cria os cards de estatísticas"""
    return f"""
    <div class="row">
        <div class="col-md-3">
            <div class="stats-card">
                <div class="stats-number">13,81</div>
                <div class="stats-label">Desvio-Padrão</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stats-card">
                <div class="stats-number">82,27%</div>
                <div class="stats-label">Precisão do modelo</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stats-card">
                <div class="stats-number">380.851</div>
                <div class="stats-label">Registros</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stats-card">
                <div class="stats-number">LightGBM</div>
                <div class="stats-label">Biblioteca de Machine Learning</div>
            </div>
        </div>
    </div>
    """

# Carregar dados de municípios por UF do arquivo JSON
MUNICIPIOS_POR_UF = load_json_data('municipios_por_uf.json')

# API para retornar municípios por UF
@rt("/api/municipios/{uf}")
def get_municipios(uf: str):
    municipios = MUNICIPIOS_POR_UF.get(uf.upper(), [])
    return {"municipios": municipios}

# Rota principal - Dashboard (com heatmap integrado)
@rt("/")
def get():
    return Html(
        Head(
            Title("SafeWay - Dashboard"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            NotStr(get_full_css()),
            # Chart.js CDN
            Script(src="https://cdn.jsdelivr.net/npm/chart.js")
        ),
        Body(
            NotStr(sidebar("dashboard")),
            Div(
                NotStr(top_header()),
                Div(
                    NotStr(create_stats_cards()),
                    
                    # Bloco do Heatmap (usa create_heatmap do dashboard.py)
                    Div(
                        Div(
                            Div(
                                Div(
                                    H5("Mapa de Calor - Acidentes por Região"),
                                    cls="card-header"
                                ),
                                Div(
                                    # Insere diretamente o HTML retornado pela função create_heatmap()
                                    NotStr(create_heatmap()),
                                    cls="card-body p-0",
                                    id="heatmap-container"
                                ),
                                cls="card"
                            ),
                            cls="col-md-8"
                        ),
                        Div(
                            Div(
                                Div(
                                    H5("Informações Rápidas", cls="card-header"),
                                    Div(
                                        Div(
                                            Div("Última atualização", cls="stats-label"),
                                            Div(datetime.now().strftime("%d/%m/%Y %H:%M"), cls="stats-number")
                                        ),
                                        cls="card-body"
                                    ),
                                    cls="card"
                                ),
                                cls="col-md-4"
                            ),
                            cls="row mb-4"
                        ),
                        
                        cls="row mb-4"
                    ),

                    # Gráficos lado a lado ocupando largura total (Chart.js)
                    Div(
                        Div(
                            Div(
                                Div(
                                    H5("Acidentes por Horário"),
                                    cls="card-header"
                                ),
                                Div(
                                    Div(
                                        Canvas(id="hourlyChart", style="height: 400px; max-height: 400px; width: 100%;"),
                                        cls="chart-container"
                                    ),
                                    cls="card-body"
                                ),
                                cls="card"
                            ),
                            cls="chart-col"
                        ),
                        Div(
                            Div(
                                Div(
                                    H5("Acidentes por Dia da Semana"),
                                    cls="card-header"
                                ),
                                Div(
                                    Div(
                                        Canvas(id="weekChart", style="height: 400px; max-height: 400px; width: 100%;"),
                                        cls="chart-container"
                                    ),
                                    cls="card-body"
                                ),
                                cls="card"
                            ),
                            cls="chart-col"
                        ),
                        cls="chart-row"
                    ),
                    
                    cls="content-area"
                ),
                cls="main-content"
            ),
            
            # Scripts para gráficos
            Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    // Aguardar um pouco para garantir que Chart.js carregou
                    setTimeout(function() {
                        // Verificar se Chart.js está carregado
                        if (typeof Chart === 'undefined') {
                            console.error('Chart.js não carregado');
                            return;
                        }
                        
                        // Gráfico de acidentes por horário
                        const hourlyCtx = document.getElementById('hourlyChart');
                        if (hourlyCtx) {
                            new Chart(hourlyCtx, {
                                type: 'bar',
                                data: {
                                    labels: ['0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', 
                                            '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h'],
                                    datasets: [{
                                        label: 'Acidentes',
                                        data: [12, 8, 5, 3, 2, 4, 8, 15, 25, 30, 28, 32, 35, 38, 42, 45, 48, 52, 45, 38, 32, 28, 22, 18],
                                        backgroundColor: '#4A90E2',
                                        borderColor: '#2C5282',
                                        borderWidth: 1,
                                        borderRadius: 4
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            display: false
                                        }
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                            grid: {
                                                color: '#e9ecef'
                                            }
                                        },
                                        x: {
                                            grid: {
                                                display: false
                                            }
                                        }
                                    }
                                }
                            });
                        }
                        
                        // Gráfico de dias da semana
                        const weekCtx = document.getElementById('weekChart');
                        if (weekCtx) {
                            // Dados dos dias da semana
                            const diasSemana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'];
                            const dadosAcidentes = [45, 38, 42, 48, 52, 65, 58]; // Dados simulados realistas
                            
                            new Chart(weekCtx, {
                                type: 'line',
                                data: {
                                    labels: diasSemana,
                                    datasets: [{
                                        label: 'Acidentes por dia da semana',
                                        data: dadosAcidentes,
                                        borderColor: '#4A90E2',
                                        backgroundColor: 'rgba(74, 144, 226, 0.1)',
                                        borderWidth: 3,
                                        fill: true,
                                        tension: 0.4,
                                        pointBackgroundColor: '#4A90E2',
                                        pointBorderColor: '#2C5282',
                                        pointRadius: 6,
                                        pointHoverRadius: 8
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            display: false
                                        }
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                            grid: {
                                                color: '#e9ecef'
                                            },
                                            title: {
                                                display: true,
                                                text: 'Número de Acidentes'
                                            }
                                        },
                                        x: {
                                            grid: {
                                                display: false
                                            },
                                            title: {
                                                display: true,
                                                text: 'Dia da Semana'
                                            }
                                        }
                                    },
                                    interaction: {
                                        intersect: false,
                                        mode: 'index'
                                    }
                                }
                            });
                        }
                    }, 1000); // Aguardar 1 segundo
                });
            """)
        )
    )

# Rota de predição
@rt("/prediction")
def get():
    # Carregar dados dos arquivos JSON
    uf_options = load_json_data('uf_options.json')
    condicoes = load_json_data('condicoes_metereologicas_options.json')
    
    return Html(
        Head(
            Title("SafeWay - Predição"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            NotStr(get_full_css())
        ),
        Body(
            NotStr(sidebar("prediction")),
            Div(
                NotStr(top_header()),
                Div(
                    Div(
                        Div(
                            Div(
                                H5("Formulário de Predição de Acidentes"),
                                cls="card-header"
                            ),
                            Div(
                                Form(
                                    Div(
                                        Label("Data:", cls="form-label"),
                                        Input(type="date", name="data", cls="form-control", required=True),
                                        cls="form-group"
                                    ),
                                    Div(
                                        Label("Horário:", cls="form-label"),
                                        Input(type="time", name="horario", cls="form-control", required=True),
                                        cls="form-group"
                                    ),
                                    Div(
                                        Label("UF (Estado):", cls="form-label"),
                                        Select(
                                            Option("Selecione um estado...", value=""),
                                            *[Option(uf, value=uf) for uf in uf_options],
                                            name="uf", cls="form-select", required=True, id="uf-select"
                                        ),
                                        cls="form-group"
                                    ),
                                    Div(
                                        Label("Município:", cls="form-label"),
                                        Select(
                                            Option("Primeiro selecione um estado", value=""),
                                            name="municipio", cls="form-select", required=True, id="municipio-select"
                                        ),
                                        cls="form-group"
                                    ),
                                    Div(
                                        Label("Condição Meteorológica:", cls="form-label"),
                                        Select(
                                            Option("Selecione uma condição...", value=""),
                                            *[Option(cond, value=cond) for cond in condicoes],
                                            name="condicao_meteorologica", cls="form-select", required=True
                                        ),
                                        cls="form-group"
                                    ),
                                    Button("🔮 Fazer Predição", type="submit", cls="btn btn-primary btn-lg"),
                                    method="post", action="/predict"
                                ),
                                cls="card-body"
                            ),
                            cls="card"
                        ),
                        cls="col-md-8"
                    ),
                    Div(
                        Div(
                            Div(
                                H5("Informações sobre a Predição"),
                                cls="card-header"
                            ),
                            Div(
                                H6("Como funciona?", cls="text-primary mb-3"),
                                P("Nosso modelo de machine learning analisa diversos fatores para prever a probabilidade de acidentes:"),
                                Ul(
                                    Li("📅 Data e horário"),
                                    Li("📍 Localização (UF e município)"),
                                    Li("🌤️ Condições meteorológicas"),
                                    Li("📊 Dados históricos de acidentes"),
                                    Li("🎯 Padrões sazonais e temporais")
                                ),

                                cls="card-body"
                            ),
                            cls="card"
                        ),
                        cls="col-md-4"
                    ),
                    cls="row"
                ),
                cls="content-area"
            ),
            cls="main-content"
        ),
        
        # JavaScript para carregar municípios dinamicamente
        Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const ufSelect = document.getElementById('uf-select');
                const municipioSelect = document.getElementById('municipio-select');
                
                ufSelect.addEventListener('change', function() {
                    const uf = this.value;
                    
                    if (!uf) {
                        municipioSelect.innerHTML = '<option value="">Primeiro selecione um estado</option>';
                        return;
                    }
                    
                    // Mostrar carregando
                    municipioSelect.innerHTML = '<option value="">Carregando municípios...</option>';
                    
                    // Fazer requisição para API de municípios
                    fetch(`/api/municipios/${uf}`)
                        .then(response => response.json())
                        .then(data => {
                            municipioSelect.innerHTML = '<option value="">Selecione um município...</option>';
                            
                            data.municipios.forEach(municipio => {
                                const option = document.createElement('option');
                                option.value = municipio;
                                option.textContent = municipio;
                                municipioSelect.appendChild(option);
                            });
                        })
                        .catch(error => {
                            console.error('Erro ao carregar municípios:', error);
                            municipioSelect.innerHTML = '<option value="">Erro ao carregar municípios</option>';
                        });
                });
            });
        """)
    )

# Rota para processar predição
@rt("/predict")
def post(data: str, horario: str, uf: str, municipio: str, condicao_meteorologica: str):
    try:
        # Simular predição
        prediction_value = random.uniform(0.1, 0.9)
        
        # Determinar nível de risco
        if prediction_value < 0.3:
            confidence = "Baixo risco"
            level = "success"
        elif prediction_value < 0.6:
            confidence = "Risco moderado"
            level = "warning"
        else:
            confidence = "Alto risco"
            level = "danger"
        
        return Html(
            Head(
                Title("SafeWay - Resultado da Predição"),
                Meta(name="viewport", content="width=device-width, initial-scale=1"),
                NotStr(get_full_css())
            ),
            Body(
                NotStr(sidebar("prediction")),
                Div(
                    NotStr(top_header()),
                    Div(
                        Div(
                            Div(
                                Div(
                                    H5("Resultado da Predição"),
                                    cls="card-header"
                                ),
                                Div(
                                    Div(
                                        H6(f"Dados da Consulta:", cls="text-primary"),
                                        P(f"📅 Data: {data}"),
                                        P(f"🕐 Horário: {horario}"),
                                        P(f"📍 Local: {municipio}, {uf}"),
                                        P(f"🌤️ Clima: {condicao_meteorologica}"),
                                        Hr(),
                                        Div(
                                            Div(f"{prediction_value * 100:.1f}%", cls="result-number"),
                                            P(f"Probabilidade de {prediction_value * 100:.1f}% de ocorrência de acidentes"),
                                            Div(
                                                Strong(f"Nível de Risco: {confidence}"),
                                                style="margin-top: 10px; font-size: 1.1rem;"
                                            ),
                                            cls=f"prediction-result bg-{level}"
                                        )
                                    ),
                                    A("← Fazer Nova Predição", href="/prediction", cls="btn btn-secondary mt-3"),
                                    cls="card-body"
                                ),
                                cls="card"
                            ),
                            cls="col-12"
                        ),
                        cls="row"
                    ),
                    cls="content-area"
                ),
                cls="main-content"
            )
        )
    except Exception as e:
        return f"Erro na predição: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    print("🚀 SafeWay iniciado!")
    print("📱 Acesse: http://localhost:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)
