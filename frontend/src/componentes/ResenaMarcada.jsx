// La reseña "corregida" como una prueba de imprenta: stopwords tachadas,
// sufijo recortado por el stemmer atenuado y cada término coloreado según
// cuánto empujó hacia positivo o negativo.

function alinear(pasos) {
  const conservados = pasos.tokens_sin_stopwords;
  const finales = pasos.tokens_finales;
  let j = 0;
  return pasos.tokens.map((token) => {
    if (j < conservados.length && conservados[j] === token) {
      const raiz = finales[j];
      j += 1;
      return { token, raiz, stopword: false };
    }
    return { token, raiz: null, stopword: true };
  });
}

function Palabra({ token, raiz, stopword, aporte }) {
  if (stopword) {
    return <s className="palabra palabra--stop" title="Stopword eliminada">{token}</s>;
  }
  const tono = aporte > 0 ? "pos" : aporte < 0 ? "neg" : "neutra";
  const intensidad = Math.min(Math.abs(aporte ?? 0) / 1.5, 1);
  const esPrefijo = raiz && token.startsWith(raiz) && raiz !== token;
  return (
    <span
      className={`palabra palabra--${tono}`}
      style={{ "--intensidad": intensidad }}
      title={aporte ? `Raíz «${raiz}», aporte ${aporte > 0 ? "+" : ""}${aporte.toFixed(2)}` : `Raíz «${raiz}»`}
    >
      {esPrefijo ? (
        <>
          {raiz}
          <span className="palabra__sufijo">{token.slice(raiz.length)}</span>
        </>
      ) : (
        token
      )}
    </span>
  );
}

export default function ResenaMarcada({ pasos, terminos }) {
  const aportes = Object.fromEntries(terminos.map((t) => [t.termino, t.aporte]));
  const palabras = alinear(pasos);

  return (
    <figure className="marcada">
      <blockquote className="marcada__texto">
        {palabras.map((p, i) => (
          <span key={i}>
            <Palabra {...p} aporte={p.raiz ? aportes[p.raiz] : undefined} />{" "}
          </span>
        ))}
      </blockquote>
      <figcaption className="leyenda">
        <span><s>tachada</s> stopword que se quita</span>
        <span>
          raíz<span className="palabra__sufijo">sufijo</span> lo que recorta el stemming
        </span>
        <span className="leyenda__pos">empuja a positiva</span>
        <span className="leyenda__neg">empuja a negativa</span>
      </figcaption>
    </figure>
  );
}
