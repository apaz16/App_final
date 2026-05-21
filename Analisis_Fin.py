## RUTA MAC 
##   Primero --->  cd '/Users/alejandropaz/Desktop/UP/8° Semestre/Seminario Finanzas/Proyecto final'
###  Segundo ---> "/opt/anaconda3/bin/python" -m streamlit run Analisis_Fin.py
############################# ----------- ##############
import yfinance as yf
import streamlit as st
import ta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

st.set_page_config(
    page_title="Análisis Financiero con Auto ARIMA",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>📈 Análisis Financiero con Yahoo Finance</h1>",
    unsafe_allow_html=True
)
### AUTORES ###
st.markdown(
    "<h6 style='text-align: center; color: orange'>Mariana Murillo, Joseph Orta y Alejandro Paz</h6>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Precios, MACD y pronóstico con Auto ARIMA</h3>",
    unsafe_allow_html=True
)


## Sugerido probar ^VIX, ^GSPC, LIVEPOLC-1.MX

ticker = st.text_input("Ingresa el Ticker deseado", "CUERVO.MX")

periodo = st.text_input("Ingresa el periodo deseado", "1y")

Text_Granularity = st.text_input("Inserta la Granularidad deseada:", "1d")

Ticker = yf.Ticker(ticker)
data = Ticker.history(periodo, interval = Text_Granularity)



## Informacion del ticker

st.subheader(f"Información de {ticker.upper()}")

col1, col2 = st.columns([2, 1])

with col1:
    st.metric("Último precio de cierre", f"${data['Close'].iloc[-1]:.2f}")

with col2:
    rendimiento = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100
    st.metric("Rendimiento efectivo del periodo", f"{rendimiento:.2f}%")

left, space, right = st.columns([3,1,3])

with left:
    
    ## Grafica de precios normal
    
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna()
    
    fig_precio = px.line(
        data,
        x="Date",
        y="Close",
        title=f"Precio de cierre de {ticker.upper()}",
        markers=False
    )
    
    st.plotly_chart(fig_precio)
    
    ## Noticia relevante 1
    
    News = Ticker.news[0]
    Title = News["content"]["title"]
    Summary = News["content"]["summary"]
    st.header(Title)
    st.write(Summary)
    
with right:
    
    ## Grafica de MACD
    
    MACD = ta.trend.MACD(data["Close"])
    data["MACD_macd"]  = MACD.macd().dropna()
    data["MACD_signal"] = MACD.macd_signal().dropna()
    ## Graficar dos series en un mismo gráfico ##
    st.write("Gráfico de MACD de ", ticker)
    st.line_chart(data[["MACD_macd", "MACD_signal"]])
    
   ## Noticia relevante 2
    
    News = Ticker.news[1]
    Title = News["content"]["title"]
    Summary = News["content"]["summary"]
    st.header(Title)
    st.write(Summary)
    
## Pronóstico utilizando Auto ARIMA


serie = data.set_index("Date")["Close"].dropna()
serie = pd.to_numeric(serie, errors="coerce").dropna()
serie = serie.astype(float)

if len(serie) < 30:
    st.warning("Se necesitan al menos 30 observaciones para hacer un pronóstico más confiable.")

else:

    # =========================
    # TRAIN / TEST
    # =========================

    corte = int(len(serie) * 0.80)

    train = serie.iloc[:corte].dropna()
    test = serie.iloc[corte:].dropna()

    # =========================
    # AUTO ARIMA TRAIN
    # =========================

    modelo_test = auto_arima(
        train,
        start_p=0,
        start_q=0,
        max_p=4,
        max_q=4,
        d=None,
        seasonal=False,
        stationary=False,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
        with_intercept=True
    )



    # Pronóstico test
    pred_test = modelo_test.predict(n_periods=len(test))

    pred_test = pd.Series(
        np.asarray(pred_test, dtype=float),
        index=test.index
    )

    # =========================
    # MÉTRICAS
    # =========================

    comparacion = pd.DataFrame({
        "real": test.values,
        "predicho": pred_test.values
    })

    comparacion = comparacion.replace([np.inf, -np.inf], np.nan).dropna()

    if len(comparacion) == 0:

        mse = 0
        rmse = 0
        mape = 0

    else:

        mse = mean_squared_error(
            comparacion["real"],
            comparacion["predicho"]
        )

        rmse = np.sqrt(mse)

        comparacion_mape = comparacion[
            comparacion["real"] != 0
        ]

        if len(comparacion_mape) == 0:

            mape = 0

        else:

            mape = np.mean(
                np.abs(
                    (
                        comparacion_mape["real"]
                        - comparacion_mape["predicho"]
                    )
                    /
                    comparacion_mape["real"]
                )
            ) * 100

    # =========================
    # AUTO ARIMA FINAL
    # =========================

    modelo_final = auto_arima(
        serie,
        start_p=0,
        start_q=0,
        max_p=4,
        max_q=4,
        d=None,
        seasonal=False,
        stationary=False,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
        with_intercept=True
    )

    # =========================
    # PRONÓSTICO FUTURO
    # =========================

    pronostico = modelo_final.predict(n_periods=5)

    pronostico = np.asarray(
        pronostico,
        dtype=float
    )

    fechas_futuras = pd.bdate_range(
        start=serie.index[-1] + pd.Timedelta(days=1),
        periods=5
    )

    pronostico = pd.Series(
        pronostico,
        index=fechas_futuras
    )

    # =========================
    # AJUSTE HISTÓRICO
    # =========================

    fitted_values = modelo_final.predict_in_sample()

    fitted_values = np.asarray(
        fitted_values,
        dtype=float
    )

    fitted_values = pd.Series(
        fitted_values,
        index=serie.index
    ).tail(15)

    # =========================
    # TABLA PRONÓSTICO
    # =========================

    tabla_pronostico = pd.DataFrame({
        "Fecha": pronostico.index.strftime("%Y-%m-%d"),
        "Precio pronosticado": pronostico.values.round(2)
    })

    # =========================
    # SEÑAL
    # =========================

    precio_hoy = serie.iloc[-1]

    precio_manana = pronostico.iloc[0]

    variacion = (
        (precio_manana / precio_hoy) - 1
    ) * 100

    # =========================
    # LAYOUT
    # =========================

    izq, der = st.columns([4, 2])

    # =========================
    # GRÁFICA
    # =========================

    with izq:

        ultimos_15 = serie.tail(15)

        fig_forecast = go.Figure()

        # Observados

        fig_forecast.add_trace(
            go.Scatter(
                x=ultimos_15.index,
                y=ultimos_15.values,
                mode="lines+markers",
                name="Valores observados",
                line=dict(
                    color="blue",
                    width=3
                )
            )
        )

        # Ajuste

        fig_forecast.add_trace(
            go.Scatter(
                x=fitted_values.index,
                y=fitted_values.values,
                mode="lines",
                name="Ajuste del modelo",
                line=dict(
                    color="green",
                    width=3,
                    dash="dot"
                )
            )
        )

        # Pronóstico

        fig_forecast.add_trace(
            go.Scatter(
                x=pronostico.index,
                y=pronostico.values,
                mode="lines+markers",
                name="Pronóstico 5 días",
                line=dict(
                    color="orange",
                    width=3
                )
            )
        )

        fig_forecast.update_layout(
            title="Zoom: valores observados, ajuste del modelo y pronóstico",
            xaxis_title="Fecha",
            yaxis_title="Precio de cierre",
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )
        
        
    # Mostrar modelo elegido
    st.markdown(f"Modelo Auto ARIMA elegido {modelo_final.order}")
    
    
    # =========================
    # PANEL DERECHO
    # =========================

    with der:

        st.markdown("### Tabla de pronóstico")

        st.dataframe(
            tabla_pronostico,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Evaluación del modelo")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div style='text-align: center;'>
                <h5>MSE</h5>
                <h3>{mse:.2f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div style='text-align: center;'>
                <h5>RMSE</h5>
                <h3>$ +/-{rmse:.2f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div style='text-align: center;'>
                <h5>MAPE</h5>
                <h3>{mape:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================
    # RECOMENDACIÓN
    # =========================

    if precio_manana > precio_hoy:

        recomendacion = "Comprar"

    elif precio_manana < precio_hoy:

        recomendacion = "Vender"

    else:

        recomendacion = "Mantener"

    if recomendacion == "Comprar":

        st.markdown(
            f"""
            <div style="
                background-color: rgba(40, 167, 69, 0.25);
                padding: 18px;
                border-radius: 12px;
                font-size: 24px;
                font-weight: 600;
                color: #4ade80;
                text-align: center;
            ">
                📈 Señal: Comprar. El modelo estima una subida de {variacion:.2f}% para mañana.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif recomendacion == "Vender":

        st.markdown(
            f"""
            <div style="
                background-color: rgba(220, 53, 69, 0.25);
                padding: 18px;
                border-radius: 12px;
                font-size: 24px;
                font-weight: 600;
                color: #f87171;
                text-align: center;
            ">
                📉 Señal: Vender. El modelo estima una caída de {abs(variacion):.2f}% para mañana.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="
                background-color: rgba(108, 117, 125, 0.25);
                padding: 18px;
                border-radius: 12px;
                font-size: 24px;
                font-weight: 600;
                color: #d1d5db;
                text-align: center;
            ">
                ➖ Señal: Mantener. El modelo estima estabilidad en el precio.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Señal académica basada en un modelo estadístico simple; no representa asesoría financiera."
    )