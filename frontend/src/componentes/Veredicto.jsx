import { useState, useEffect } from "react";
import { api, porcentaje } from "../api.js";

// Balanza entre negativo (izquierda) y positivo (derecha): la aguja marca P(positivo).
export default function Veredicto({ resultado }) {
  const pPositivo = resultado.probabilidades.positivo ?? 0;
  const positivo = resultado.sentimiento === "positivo";
  const [feedbackEnviado, setFeedbackEnviado] = useState(null);
  const [enviando, setEnviando] = useState(false);

  // Reiniciar estado si cambia el resultado analizado
  useEffect(() => {
    setFeedbackEnviado(null);
  }, [resultado?.id, resultado?.texto]);

  async function enviarFeedback(acertado) {
    if (!resultado?.id || enviando) return;
    setEnviando(true);
    try {
      await api.feedback(resultado.id, acertado);
      setFeedbackEnviado(acertado);
    } catch (err) {
      console.error("Error al registrar feedback:", err);
    } finally {
      setEnviando(false);
    }
  }

  return (
    <div className={`veredicto ${positivo ? "veredicto--pos" : "veredicto--neg"}`}>
      <p className="veredicto__clase">{positivo ? "Positiva" : "Negativa"}</p>
      <p className="veredicto__confianza">
        con {porcentaje(resultado.confianza)} de probabilidad
      </p>

      <div
        className="balanza"
        role="img"
        aria-label={`Probabilidad positiva ${porcentaje(pPositivo)}, negativa ${porcentaje(1 - pPositivo)}`}
      >
        <div className="balanza__pista">
          <div className="balanza__aguja" style={{ left: `${pPositivo * 100}%` }} />
        </div>
        <div className="balanza__extremos">
          <span>negativa {porcentaje(1 - pPositivo, 0)}</span>
          <span>positiva {porcentaje(pPositivo, 0)}</span>
        </div>
      </div>

      {resultado.id && (
        <div className="veredicto__feedback">
          <span className="veredicto__feedback-titulo">¿Acertó el modelo?</span>
          {feedbackEnviado !== null ? (
            <span className="veredicto__feedback-gracias">
              {feedbackEnviado ? "👍 ¡Gracias por confirmar!" : "👎 Anotado para mejorar."}
            </span>
          ) : (
            <div className="veredicto__feedback-acciones">
              <button
                type="button"
                className="boton-feedback"
                title="Sí, acertó el veredicto"
                onClick={() => enviarFeedback(true)}
                disabled={enviando}
              >
                👍 Sí
              </button>
              <button
                type="button"
                className="boton-feedback"
                title="No acertó el veredicto"
                onClick={() => enviarFeedback(false)}
                disabled={enviando}
              >
                👎 No
              </button>
            </div>
          )}
        </div>
      )}

      {resultado.terminos_clave.length > 0 && (
        <div className="claves">
          <p className="claves__titulo">Lo que más pesó en la decisión</p>
          <ul className="claves__lista">
            {resultado.terminos_clave.map((t) => (
              <li key={t.termino} className={t.aporte > 0 ? "clave clave--pos" : "clave clave--neg"}>
                <span>{t.termino}</span>
                <span className="clave__valor">
                  {t.aporte > 0 ? "+" : "−"}
                  {Math.abs(t.aporte).toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
