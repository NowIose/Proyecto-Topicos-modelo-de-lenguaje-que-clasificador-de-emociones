import { porcentaje } from "../api.js";

// Balanza entre negativo (izquierda) y positivo (derecha): la aguja marca P(positivo).
export default function Veredicto({ resultado }) {
  const pPositivo = resultado.probabilidades.positivo ?? 0;
  const positivo = resultado.sentimiento === "positivo";

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
