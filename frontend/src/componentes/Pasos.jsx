// Preprocesamiento paso a paso, en el mismo orden que PipelineNLP.procesar_texto.

function Fichas({ items, variante = "" }) {
  if (!items.length) return <p className="vacio">Ninguno.</p>;
  return (
    <ul className="fichas">
      {items.map((t, i) => (
        <li key={`${t}-${i}`} className={`ficha ${variante}`}>
          {t}
        </li>
      ))}
    </ul>
  );
}

export default function Pasos({ pasos }) {
  const reduccion = pasos.metodo_morfologico === "lematizacion" ? "Lematización" : "Stemming";
  const etiquetaRaiz = pasos.metodo_morfologico === "lematizacion" ? "lema" : "raíz";

  return (
    <section className="pasos" aria-labelledby="titulo-pasos">
      <h2 id="titulo-pasos" className="seccion__titulo">Cómo ve el modelo esta reseña</h2>
      <ol className="pasos__lista">
        <li className="paso">
          <h3>Normalización</h3>
          <p className="paso__ayuda">Minúsculas, sin tildes (se conserva la ñ), sin URLs, números, signos ni emojis.</p>
          <p className="paso__texto">{pasos.texto_normalizado || <span className="vacio">Texto vacío tras limpiar.</span>}</p>
        </li>
        <li className="paso">
          <h3>Tokenización</h3>
          <p className="paso__ayuda">{pasos.tokens.length} tokens.</p>
          <Fichas items={pasos.tokens} />
        </li>
        <li className="paso">
          <h3>Stopwords</h3>
          <p className="paso__ayuda">
            Se quitan {pasos.stopwords_eliminadas.length}. Las negaciones e intensificadores
            (no, nunca, muy…) se conservan porque cambian el sentimiento.
          </p>
          <Fichas items={pasos.stopwords_eliminadas} variante="ficha--stop" />
        </li>
        <li className="paso">
          <h3>{reduccion}</h3>
          <p className="paso__ayuda">Cada palabra queda reducida a su {etiquetaRaiz}.</p>
          <ul className="fichas">
            {pasos.tokens_sin_stopwords.map((t, i) => (
              <li key={`${t}-${i}`} className="ficha ficha--par">
                {t} <span aria-hidden="true">›</span> <strong>{pasos.tokens_finales[i]}</strong>
              </li>
            ))}
          </ul>
        </li>
        <li className="paso">
          <h3>TF-IDF</h3>
          <p className="paso__ayuda">
            Términos (y bigramas) que existen en el vocabulario del modelo; lo que no está en el
            vocabulario no influye.
          </p>
          <Fichas items={pasos.terminos_en_vocabulario} variante="ficha--vocab" />
        </li>
      </ol>
    </section>
  );
}
