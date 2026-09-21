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


# ==========================================================
# CONFIGURACIÓN DE LA APP
# ==========================================================

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
    "<h6 style='text-align: center; color: orange'>"
    "Mariana Murillo, Joseph Orta y Alejandro Paz"
    "</h6>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>"
    "Precios, MACD y pronóstico con Auto ARIMA"
    "</h3>",
    unsafe_allow_html=True
)


# ==========================================================
# INPUTS
# ==========================================================

## Sugerido probar ^VIX, ^GSPC, LIVEPOLC-1.MX

ticker = st.text_input(
    "Ingresa el Ticker deseado",
    "CUERVO.MX"
)

periodo = st.text_input(
    "Ingresa el periodo deseado",
    "1y"
)

Text_Granularity = st.text_input(
    "Inserta la Granularidad deseada:",
    "1d"
)

ticker = ticker.strip()
periodo = periodo.strip()
Text_Granularity = Text_Granularity.strip()


# ==========================================================
# DESCARGA DE DATOS
# ==========================================================

try:

    Ticker = yf.Ticker(ticker)

    data = Ticker.history(
        period=periodo,
        interval=Text_Granularity
    )

except Exception as e:

    st.error(
        f"No fue posible descargar los datos de {ticker.upper()}."
    )

    st.caption(str(e))

    st.stop()


# Validar que Yahoo Finance regresó información
if data.empty:

    st.error(
        f"No se encontraron datos para {ticker.upper()} "
        f"con periodo {periodo} y granularidad {Text_Granularity}."
    )

    st.stop()


if "Close" not in data.columns:

    st.error(
        "Yahoo Finance no regresó una columna de precios de cierre."
    )

    st.stop()


# ==========================================================
# PREPARACIÓN DE DATOS
# ==========================================================

data = data.reset_index()


# Yahoo utiliza Date para datos diarios y Datetime
# para algunas granularidades intradía.
if "Date" in data.columns:

    columna_fecha = "Date"

elif "Datetime" in data.columns:

    columna_fecha = "Datetime"

else:

    st.error(
        "No se encontró una columna de fecha en los datos."
    )

    st.stop()


data[columna_fecha] = pd.to_datetime(
    data[columna_fecha],
    errors="coerce"
)


# Eliminar timezone para evitar problemas posteriores
if data[columna_fecha].dt.tz is not None:

    data[columna_fecha] = (
        data[columna_fecha]
        .dt
        .tz_localize(None)
    )


# Convertir Close a numérico
data["Close"] = pd.to_numeric(
    data["Close"],
    errors="coerce"
)


# Eliminar únicamente filas sin fecha o Close
data = data.dropna(
    subset=[
        columna_fecha,
        "Close"
    ]
)


if data.empty:

    st.error(
        "Después de limpiar los datos no quedaron observaciones válidas."
    )

    st.stop()


# ==========================================================
# INFORMACIÓN DEL TICKER
# ==========================================================

st.subheader(
    f"Información de {ticker.upper()}"
)

col1, col2 = st.columns([2, 1])


with col1:

    st.metric(
        "Último precio de cierre",
        f"${data['Close'].iloc[-1]:.2f}"
    )


with col2:

    rendimiento = (
        (
            data["Close"].iloc[-1]
            /
            data["Close"].iloc[0]
        )
        - 1
    ) * 100

    st.metric(
        "Rendimiento efectivo del periodo",
        f"{rendimiento:.2f}%"
    )


# ==========================================================
# FUNCIÓN PARA MOSTRAR NOTICIAS DE FORMA SEGURA
# ==========================================================

def mostrar_noticia(numero):

    try:

        noticias = Ticker.news

        if noticias is not None and len(noticias) > numero:

            noticia = noticias[numero]

            contenido = noticia.get(
                "content",
                {}
            )

            titulo = contenido.get(
                "title",
                "Noticia relevante"
            )

            resumen = contenido.get(
                "summary",
                "No hay resumen disponible."
            )

            st.header(titulo)

            st.write(resumen)

        else:

            st.info(
                "No hay una noticia disponible en este momento."
            )

    except Exception:

        st.info(
            "No fue posible cargar esta noticia."
        )


# ==========================================================
# GRÁFICAS PRINCIPALES
# ==========================================================

left, space, right = st.columns([3, 1, 3])


# ==========================================================
# PRECIO
# ==========================================================

with left:

    fig_precio = px.line(
        data,
        x=columna_fecha,
        y="Close",
        title=f"Precio de cierre de {ticker.upper()}",
        markers=False
    )

    st.plotly_chart(
        fig_precio,
        use_container_width=True
    )

    mostrar_noticia(0)


# ==========================================================
# MACD
# ==========================================================

with right:

    if len(data) >= 26:

        MACD = ta.trend.MACD(
            data["Close"]
        )

        data["MACD_macd"] = MACD.macd()

        data["MACD_signal"] = MACD.macd_signal()

        st.write(
            "Gráfico de MACD de",
            ticker
        )

        st.line_chart(
            data[
                [
                    "MACD_macd",
                    "MACD_signal"
                ]
            ]
        )

    else:

        st.warning(
            "No existen suficientes observaciones para calcular el MACD."
        )

    mostrar_noticia(1)


# ==========================================================
# PRONÓSTICO UTILIZANDO AUTO ARIMA
# ==========================================================

serie = (
    data
    .set_index(columna_fecha)["Close"]
    .copy()
)


# Convertir explícitamente a número
serie = pd.to_numeric(
    serie,
    errors="coerce"
)


# Limpiar NaN e infinitos
serie = serie.replace(
    [np.inf, -np.inf],
    np.nan
)

serie = serie.dropna()

serie = serie.astype(float)

serie = serie.sort_index()


# Eliminar fechas duplicadas, por seguridad
serie = serie[
    ~serie.index.duplicated(
        keep="last"
    )
]


# ==========================================================
# VALIDACIÓN DE OBSERVACIONES
# ==========================================================

if len(serie) < 30:

    st.warning(
        "Se necesitan al menos 30 observaciones "
        "para hacer un pronóstico más confiable."
    )


else:

    # ======================================================
    # TRAIN / TEST
    # ======================================================

    corte = int(
        len(serie) * 0.80
    )

    train = (
        serie
        .iloc[:corte]
        .dropna()
    )

    test = (
        serie
        .iloc[corte:]
        .dropna()
    )


    # ======================================================
    # CORRECCIÓN IMPORTANTE
    # ======================================================
    #
    # Auto ARIMA NO recibe el DatetimeIndex.
    #
    # Las fechas permanecen en train y test para graficar,
    # pero el modelo únicamente recibe números.
    # ======================================================

    train_arima = np.asarray(
        train,
        dtype=np.float64
    )

    test_arima = np.asarray(
        test,
        dtype=np.float64
    )


    # Validar que no existan NaN o infinitos
    train_arima = train_arima[
        np.isfinite(train_arima)
    ]

    test_arima = test_arima[
        np.isfinite(test_arima)
    ]


    # ======================================================
    # AUTO ARIMA TRAIN
    # ======================================================

    try:

        modelo_test = auto_arima(
            train_arima,
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

    except Exception as e:

        st.error(
            "No fue posible entrenar el modelo Auto ARIMA."
        )

        st.caption(str(e))

        st.stop()


    # ======================================================
    # PRONÓSTICO DEL TEST
    # ======================================================

    try:

        pred_test = modelo_test.predict(
            n_periods=len(test_arima)
        )

    except Exception as e:

        st.error(
            "El modelo fue entrenado, pero ocurrió un error "
            "al realizar el pronóstico del conjunto de prueba."
        )

        st.caption(str(e))

        st.stop()


    pred_test = np.asarray(
        pred_test,
        dtype=np.float64
    )


    # Le devolvemos las fechas DESPUÉS de que ARIMA
    # realizó el pronóstico
    pred_test = pd.Series(
        pred_test,
        index=test.index[:len(pred_test)]
    )


    # ======================================================
    # MÉTRICAS
    # ======================================================

    comparacion = pd.DataFrame(
        {
            "real": test.iloc[
                :len(pred_test)
            ].values,

            "predicho": pred_test.values
        }
    )


    comparacion = comparacion.replace(
        [np.inf, -np.inf],
        np.nan
    )

    comparacion = comparacion.dropna()


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
                        -
                        comparacion_mape["predicho"]
                    )
                    /
                    comparacion_mape["real"]
                )
            ) * 100


    # ======================================================
    # AUTO ARIMA FINAL
    # ======================================================

    # Nuevamente pasamos únicamente valores numéricos
    serie_arima = np.asarray(
        serie,
        dtype=np.float64
    )


    serie_arima = serie_arima[
        np.isfinite(serie_arima)
    ]


    try:

        modelo_final = auto_arima(
            serie_arima,
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

    except Exception as e:

        st.error(
            "No fue posible entrenar el modelo Auto ARIMA final."
        )

        st.caption(str(e))

        st.stop()


    # ======================================================
    # PRONÓSTICO FUTURO
    # ======================================================

    try:

        pronostico = modelo_final.predict(
            n_periods=5
        )

    except Exception as e:

        st.error(
            "No fue posible realizar el pronóstico futuro."
        )

        st.caption(str(e))

        st.stop()


    pronostico = np.asarray(
        pronostico,
        dtype=np.float64
    )


    # ======================================================
    # FECHAS FUTURAS
    # ======================================================
    #
    # Para granularidad 1d usamos días hábiles.
    # Esto evita sábado y domingo.
    # ======================================================

    ultima_fecha = pd.Timestamp(
        serie.index[-1]
    )


    fechas_futuras = pd.bdate_range(
        start=ultima_fecha
        + pd.Timedelta(days=1),
        periods=5
    )


    pronostico = pd.Series(
        pronostico,
        index=fechas_futuras
    )


    # ======================================================
    # AJUSTE HISTÓRICO
    # ======================================================

    fitted_values = (
        modelo_final
        .predict_in_sample()
    )


    fitted_values = np.asarray(
        fitted_values,
        dtype=np.float64
    )


    # Asegurar misma longitud
    longitud_fitted = min(
        len(fitted_values),
        len(serie)
    )


    fitted_values = pd.Series(
        fitted_values[-longitud_fitted:],
        index=serie.index[-longitud_fitted:]
    )


    fitted_values = (
        fitted_values
        .tail(15)
    )


    # ======================================================
    # TABLA PRONÓSTICO
    # ======================================================

    tabla_pronostico = pd.DataFrame(
        {
            "Fecha":
                pronostico.index.strftime(
                    "%Y-%m-%d"
                ),

            "Precio pronosticado":
                pronostico.values.round(2)
        }
    )


    # ======================================================
    # SEÑAL
    # ======================================================

    precio_hoy = serie.iloc[-1]

    precio_manana = pronostico.iloc[0]


    variacion = (
        (
            precio_manana
            /
            precio_hoy
        )
        - 1
    ) * 100


    # ======================================================
    # LAYOUT
    # ======================================================

    izq, der = st.columns(
        [4, 2]
    )


    # ======================================================
    # GRÁFICA
    # ======================================================

    with izq:

        ultimos_15 = serie.tail(15)

        fig_forecast = go.Figure()


        # --------------------------------------
        # OBSERVADOS
        # --------------------------------------

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


        # --------------------------------------
        # AJUSTE DEL MODELO
        # --------------------------------------

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


        # --------------------------------------
        # PRONÓSTICO
        # --------------------------------------

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

            title=(
                "Zoom: valores observados, "
                "ajuste del modelo y pronóstico"
            ),

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


    # ======================================================
    # MODELO ELEGIDO
    # ======================================================

    st.markdown(
        f"Modelo Auto ARIMA elegido: "
        f"**ARIMA{modelo_final.order}**"
    )


    # ======================================================
    # PANEL DERECHO
    # ======================================================

    with der:

        st.markdown(
            "### Tabla de pronóstico"
        )


        st.dataframe(
            tabla_pronostico,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "### Evaluación del modelo"
        )


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


    # ======================================================
    # RECOMENDACIÓN
    # ======================================================

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
                📈 Señal: Comprar.
                El modelo estima una subida de
                {variacion:.2f}% para mañana.
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
                📉 Señal: Vender.
                El modelo estima una caída de
                {abs(variacion):.2f}% para mañana.
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
                ➖ Señal: Mantener.
                El modelo estima estabilidad en el precio.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.caption(
        "Señal académica basada en un modelo estadístico simple; "
        "no representa asesoría financiera."
    )
