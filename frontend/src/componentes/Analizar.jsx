import { useEffect, useState } from "react";
import { api } from "../api.js";
import Veredicto from "./Veredicto.jsx";
import ResenaMarcada from "./ResenaMarcada.jsx";
import Pasos from "./Pasos.jsx";

const EJEMPLOS = [
  "Llegó antes de lo previsto y la calidad es excelente, lo recomiendo.",
  "No funciona, se apagó a los dos días y el vendedor no responde.",
  "No está mal, pero esperaba más por el precio.",
  "Pésimo embalaje, la caja llegó rota aunque el producto sí funciona.",
];

export default function Analizar() {
  const [texto, setTexto] = useState("");
  const [resultado, setResultado] = useState(null);
  const [pasos, setPasos] = useState(null);
  const [error, setError] = useState("");
  const [cargando, setCargando] = useState(false);

  async function analizar(valor = texto) {
    setCargando(true);
    setError("");
    try {
      // Las dos llamadas usan el mismo pipeline serializado en el backend.
      const [prediccion, preprocesamiento] = await Promise.all([
        api.predecir(valor),
        api.preprocesar(valor),
      ]);
      setResultado(prediccion);
      setPasos(preprocesamiento);
    } catch (e) {
      setError(e.message);
      setResultado(null);
      setPasos(null);
    } finally {
      setCargando(false);
    }
  }

  // Permite abrir la app con una reseña ya analizada: /?texto=...
  useEffect(() => {
    const inicial = new URLSearchParams(window.location.search).get("texto");
    if (inicial) usarEjemplo(inicial);
  }, []);

  function usarEjemplo(ejemplo) {
    setTexto(ejemplo);
    analizar(ejemplo);
  }

  return (
    <section className="analizar">
      <form
        className="entrada"
        onSubmit={(e) => {
          e.preventDefault();
          analizar();
        }}
      >
        <label htmlFor="resena" className="entrada__titulo">
          Escribe o pega una reseña de producto
        </label>
        <textarea
          id="resena"
          className="entrada__texto"
          value={texto}
          onChange={(e) => setTexto(e.target.value)}
          placeholder="Por ejemplo: la batería dura poco y el cargador vino roto."
          rows={5}
          maxLength={5000}
        />
        <div className="entrada__pie">
          <span className="contador">{texto.length} / 5000</span>
          <button className="boton" type="submit" disabled={cargando || !texto.trim()}>
            {cargando ? "Analizando…" : "Analizar reseña"}
          </button>
        </div>
        <div className="ejemplos">
          <span className="ejemplos__titulo">Probar con:</span>
          {EJEMPLOS.map((e) => (
            <button key={e} type="button" className="ejemplo" onClick={() => usarEjemplo(e)}>
              {e}
            </button>
          ))}
        </div>
        {error && (
          <p className="aviso aviso--error" role="alert">
            {error}
          </p>
        )}
      </form>

      {resultado && pasos && (
        <>
          <div className="resultado">
            <Veredicto resultado={resultado} />
            <ResenaMarcada pasos={pasos} terminos={resultado.terminos_clave} />
          </div>
          <Pasos pasos={pasos} />
        </>
      )}
    </section>
  );
}
